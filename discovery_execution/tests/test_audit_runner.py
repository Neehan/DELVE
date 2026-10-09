"""Verify offline selection and grading without run-specific configuration."""

import json
from pathlib import Path

import pytest

from discovery_execution.audit.runner import AuditRunner
from discovery_execution.audit.compiler import AuditCompiler
from discovery_execution.audit.constants import COMPILED_AUDIT_FILENAME
from discovery_execution.audit.models import AuditRecord
from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CONVERSATION_FILENAME, RUN_FILENAME,
)
from discovery_execution.harness.models import DatasetConfig, Problem, RunState
from discovery_execution.harness.storage import Storage, write_json
from discovery_execution.cli.audit import parse_arguments


@pytest.fixture
def saved_runs(tmp_path: Path, conversation: RunState, problem: Problem,
               dataset_config: DatasetConfig, local_partition: Path, model_name: str) -> list[Path]:
    """Create two arbitrary arm directories with sufficient saved data for regrading."""
    rows = [problem.model_copy(update={"problem_id": f"problem-{seed}"}) for seed in (0, 1)]
    local_partition.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    paths = []
    for seed in (0, 1):
        path = tmp_path / "output" / problem.competition / str(problem.year) / model_name / "test" / "unaided-2x" / f"problem-{seed}" / f"seed_{seed}"
        path.mkdir(parents=True)
        write_json(path / RUN_FILENAME, {**dataset_config.model_dump(mode="json"), "problem_id": f"problem-{seed}",
                                         "seed": seed, "arm": "unaided-2x", "compute_multiplier_k": 2, "model": model_name,
                                         "run_id": "test", "experiment": "unaided", "inject_sketch_at": 0, "solution_index": 0})
        state = conversation.model_copy(deep=True)
        text = r"\boxed{42}" if seed == 0 else r"\boxed{0}"
        state.messages = [{"role": "assistant", "content": text}]
        state.checkpoints[0].last_message_chars = len(text)
        with Storage(path) as storage:
            storage.save(state)
        paths.append(path)
    return paths


def test_audit_entire_subtree(saved_runs: list[Path], tmp_path: Path) -> None:
    """Look up each reference in the dataset and grade all discovered saved arms."""
    AuditRunner().run(tmp_path, None, None)
    for seed, path in enumerate(saved_runs):
        audit = json.loads((path / AUDIT_FILENAME).read_text(encoding="utf-8"))
        assert audit["checkpoints"]["1x"]["correctness"]["score"] == int(seed == 0)
        assert not audit["complete"]


def test_audit_filters_by_metadata(saved_runs: list[Path], tmp_path: Path) -> None:
    """Select by recorded problem and seed even when directory names differ."""
    AuditRunner().run(tmp_path, ["problem-1"], [1])
    assert not (saved_runs[0] / AUDIT_FILENAME).exists()
    assert (saved_runs[1] / AUDIT_FILENAME).is_file()


def test_filtered_audits_preserve_compiled_rows(saved_runs: list[Path], tmp_path: Path) -> None:
    """Compiling a filtered audit retains previously graded seeds in the same arm."""
    runner = AuditRunner()
    runner.run(tmp_path, ["problem-0"], None)
    runner.run(tmp_path, ["problem-1"], None)
    path = saved_runs[0].parent.parent / COMPILED_AUDIT_FILENAME
    rows = [AuditRecord.model_validate_json(line) for line in path.read_text().splitlines()]
    assert [(row.problem_id, row.seed, row.correctness.score) for row in rows] == [
        ("problem-0", 0, 1), ("problem-1", 1, 0),
    ]
    assert all(row.compute_multiplier_k == 2 and row.checkpoint_multiplier_k == 1 for row in rows)


def test_failed_compilation_preserves_export(saved_runs: list[Path], tmp_path: Path) -> None:
    """Reject nonbinary seed data before replacing a valid compiled audit."""
    AuditRunner().run(tmp_path, None, None)
    directory = saved_runs[0].parent.parent
    compiled = directory / COMPILED_AUDIT_FILENAME
    before = compiled.read_bytes()
    path = saved_runs[0] / AUDIT_FILENAME
    result = json.loads(path.read_text())
    result["checkpoints"]["1x"]["correctness"]["score"] = 7
    write_json(path, result)
    with pytest.raises(ValueError):
        AuditCompiler(directory).compile()
    assert compiled.read_bytes() == before


def test_audit_single_seed_directory(saved_runs: list[Path]) -> None:
    """A leaf directory is also a complete audit selection."""
    AuditRunner().run(saved_runs[0], None, None)
    assert (saved_runs[0] / AUDIT_FILENAME).is_file()
    assert not (saved_runs[1] / AUDIT_FILENAME).exists()


def test_audit_uses_snapshots_not_live_conversation(saved_runs: list[Path], tmp_path: Path) -> None:
    """Grade the saved boundary snapshots independently of the resumable conversation."""
    (saved_runs[0] / CONVERSATION_FILENAME).write_bytes(b"unreadable live conversation")
    AuditRunner().run(tmp_path, None, None)
    audit = json.loads((saved_runs[0] / AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert audit["checkpoints"]["1x"]["correctness"]["score"] == 1


def test_unmatched_selection_writes_nothing(saved_runs: list[Path], tmp_path: Path) -> None:
    """Reject an unmatched problem filter before publishing results for valid matches."""
    with pytest.raises(ValueError, match="no saved runs for problems"):
        AuditRunner().run(tmp_path, ["problem-0", "missing"], None)
    assert not any((path / AUDIT_FILENAME).exists() for path in saved_runs)


def test_missing_checkpoint_writes_nothing(saved_runs: list[Path], tmp_path: Path) -> None:
    """Discover missing checkpoints before updating other selected seeds."""
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    write_json(incomplete / RUN_FILENAME, {"problem_id": "unfinished", "seed": 0})
    with pytest.raises(FileNotFoundError):
        AuditRunner().run(tmp_path, None, None)
    assert not any((path / AUDIT_FILENAME).exists() for path in saved_runs)


@pytest.mark.parametrize("option,value", [
    ("--batch-size", "16"), ("--experiment", "unaided"), ("--inject-sketch-at", "0"),
])
def test_audit_rejects_generation_options(option: str, value: str, tmp_path: Path) -> None:
    """The audit CLI accepts saved-run selection only."""
    with pytest.raises(SystemExit):
        parse_arguments(["--run-dir", str(tmp_path), option, value])
