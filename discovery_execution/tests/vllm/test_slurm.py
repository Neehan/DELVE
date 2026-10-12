"""Exercise job ordering, argument forwarding, and cleanup with fake cluster commands."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from discovery_execution.cli.audit import parse_arguments as parse_audit

SCRIPT = Path("discovery_execution/scripts/experiment.sbatch").resolve()


@pytest.mark.parametrize("mode,expected", [
    ("success", ["serve", "run", "audit"]),
    ("generation_failure", ["serve", "run"]),
    ("audit_failure", ["serve", "run", "audit"]),
    ("startup_failure", ["serve"]),
    ("invalid", []),
])
def test_job_lifecycle(mode: str, expected: list[str], slurm_environment: dict[str, str], tmp_path: Path) -> None:
    """Do not generate before readiness or audit failed generation; always stop a live server."""
    slurm_environment["TEST_MODE"] = mode
    arguments = ["--model", "qwen3-1.7b", "--competition", "aime", "--year", "2024",
                 "--experiment", "unaided", "--compute-multiplier-k", "2", "--seeds", "0", "3"]
    result = subprocess.run(["bash", str(SCRIPT), *arguments], env=slurm_environment,
                            capture_output=True, text=True, timeout=15)
    assert (result.returncode == 0) == (mode == "success"), result.stderr
    path = tmp_path / "calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()] if expected else []
    assert [call["module"].rsplit(".", 1)[1] for call in calls] == expected
    if "run" in expected:
        assert calls[0]["arguments"] == ["--model", "qwen3-1.7b", "--port", "34567", "--max-num-seqs", "16"]
        assert calls[1]["arguments"] == arguments
        assert calls[1]["endpoint"] == "http://127.0.0.1:34567/v1"
        assert (tmp_path / "stopped").exists()
    if "audit" in expected:
        assert calls[2]["arguments"] == ["--competition", "aime", "--year", "2024", "--model", "qwen3-1.7b",
                                          "--experiment", "unaided", "--compute-multiplier-k", "2", "--seeds", "0", "3"]


def test_termination_stops_children(slurm_environment: dict[str, str], tmp_path: Path) -> None:
    """A Slurm termination signal interrupts collection and cleans up the model server."""
    slurm_environment["TEST_MODE"] = "interrupted"
    process = subprocess.Popen(["bash", str(SCRIPT)], env=slurm_environment, start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        path = tmp_path / "calls.jsonl"
        while not path.exists() or "discovery_execution.cli.run" not in path.read_text():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.terminate()
        assert process.wait(timeout=5) == 143
        assert (tmp_path / "stopped").exists()
        assert "discovery_execution.cli.audit" not in path.read_text()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.mark.parametrize("experiment", ["continue", "continue-oracle"])
def test_branch_audit_arguments(experiment: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the actual embedded CLI selection to verify branch filters reach the audit command."""
    import_context: dict[str, object] = {"__name__": "__main__"}
    script = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", "--model", "qwen3-1.7b", "--competition", "aime", "--year", "2024",
                                     "--experiment", experiment, "--source-arm", "unaided-5x", "--source-seed", "1",
                                     "--continue-at", "2", "--seeds", "0", "3"])
    listener = MagicMock()
    listener.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 34567)
    monkeypatch.setattr(socket, "socket", MagicMock(return_value=listener))
    output = StringIO()
    with redirect_stdout(output):
        exec(compile(script, str(SCRIPT), "exec"), import_context)
    audit = parse_audit(output.getvalue().splitlines()[1].split())
    assert audit.experiment == experiment and audit.source_arm == "unaided-5x"
    assert audit.source_seed == 1 and audit.continue_at == 2 and audit.seeds == [0, 3]
