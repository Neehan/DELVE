"""Isolated serving templates and fake cluster processes for launch tests."""

import os
import sys
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, Template

from discovery_execution.vllm.constants import TEMPLATES_DIRECTORY


@pytest.fixture(params=["qwen3.jinja", "deepseek.jinja"])
def chat_template(request: pytest.FixtureRequest) -> Template:
    """Load the same template files passed to the real vLLM process."""
    return Environment(loader=FileSystemLoader(TEMPLATES_DIRECTORY)).get_template(request.param)


@pytest.fixture
def mock_python(tmp_path: Path) -> Path:
    """Install a fake Python command that records launches and responds to termination."""
    binary = tmp_path / ".venv" / "bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.write_text(f"#!{sys.executable}\n" + '''
import json
import os
import signal
import sys
import time
from pathlib import Path

root = Path(os.environ["SLURM_SUBMIT_DIR"])
mode = os.environ["TEST_MODE"]
arguments = sys.argv[1:]
if arguments[0] == "-":
    sys.stdin.read()
    if mode == "invalid":
        sys.exit(2)
    print("qwen3-1.7b\\n--competition aime --year 2024 --model qwen3-1.7b --experiment unaided --compute-multiplier-k 2 --seeds 0 3\\n16\\n34567")
else:
    module = arguments[1]
    with (root / "calls.jsonl").open("a") as handle:
        handle.write(json.dumps({"module": module, "arguments": arguments[2:],
                                 "endpoint": os.environ["VLLM_BASE_URL"]}) + "\\n")
    if module.endswith(".serve"):
        if mode == "startup_failure":
            sys.exit(1)
        def stop(signum, frame):
            (root / "stopped").touch()
            sys.exit(0)
        signal.signal(signal.SIGTERM, stop)
        (root / "ready").touch()
        while True:
            time.sleep(0.01)
    elif module.endswith(".run") and mode == "generation_failure":
        sys.exit(1)
    elif module.endswith(".run") and mode == "interrupted":
        while True:
            time.sleep(0.01)
    elif module.endswith(".audit") and mode == "audit_failure":
        sys.exit(1)
''')
    python.chmod(0o755)
    return python


@pytest.fixture
def slurm_environment(tmp_path: Path, mock_python: Path) -> dict[str, str]:
    """Replace module setup, HTTP readiness, and Python children without using GPUs or sockets."""
    binary = mock_python.parent
    curl = binary / "curl"
    curl.write_text('#!/bin/bash\n[[ -f "$SLURM_SUBMIT_DIR/ready" ]]\n')
    curl.chmod(0o755)
    bootstrap = tmp_path / "bash_env"
    bootstrap.write_text('module() { :; }\n')
    return {**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_JOB_ID": "123",
            "BASH_ENV": str(bootstrap), "PATH": f"{binary}:{os.environ['PATH']}", "TEST_MODE": "success"}
