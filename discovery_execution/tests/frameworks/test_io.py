"""Validate DELVE audit identities, fitting selection, and DE-compatible prediction output."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from discovery_execution.audit.constants import COMPILED_AUDIT_FILENAME
from discovery_execution.frameworks.models import InputSelection
from discovery_execution.frameworks.runner import FrameworkRunner
from discovery_execution.frameworks.utils.constants import FIT_FILENAME, ORACLE_ARM, PREDICTIONS_FILENAME, SHORT_ARM, TOTAL_BUDGET
from discovery_execution.frameworks.utils.dataloader import DataLoader
from discovery_execution.cli.frameworks import parse_arguments
from discovery_execution.tests.frameworks.constants import DATASET, EXPECTED_ALLOCATIONS, MODEL, ROWS


def test_loader_uses_all_complete_seeds(data_loader: DataLoader, selection: InputSelection) -> None:
    """Fit counts match every input seed; long unaided and delayed oracle data are untouched."""
    assert [asdict(row) for row in data_loader.load(selection)] == [asdict(row) for row in ROWS]


def test_framework_retains_earlier_success(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Convert raw checkpoint regressions to cumulative counts without changing the audit file."""
    path = banks / DATASET / MODEL / selection.run_id / ORACLE_ARM / COMPILED_AUDIT_FILENAME
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["problem_id"] == "p1" and row["seed"] == 0 and row["checkpoint_multiplier_k"] > 1:
            row["correctness"]["score"] = 0
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content)
    assert data_loader.load(selection)[0].oracle_successes == ROWS[0].oracle_successes
    assert path.read_text() == content


@pytest.mark.parametrize("field,value", [
    ("run_id", "other"), ("solver_model", "qwen3-1.7b"), ("dataset", "aime/2025"),
    ("inject_sketch_at", 7), ("compute_multiplier_k", 7), ("correctness", {"score": 7}),
])
def test_loader_rejects_wrong_identity_or_nonbinary_grade(field: str, value: object, banks: Path,
                                                          data_loader: DataLoader, selection: InputSelection) -> None:
    """Misplaced or malformed oracle rows cannot silently change the execution estimate."""
    path = banks / DATASET / MODEL / selection.run_id / ORACLE_ARM / COMPILED_AUDIT_FILENAME
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        data_loader.load(selection)


def test_loader_rejects_duplicates(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Duplicate checkpoint rows must not double-count successful seeds."""
    path = banks / DATASET / MODEL / selection.run_id / SHORT_ARM / COMPILED_AUDIT_FILENAME
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        data_loader.load(selection)


def test_incomplete_seed_is_excluded(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """A partially audited oracle seed is neither a success nor a failure observation."""
    path = banks / DATASET / MODEL / selection.run_id / ORACLE_ARM / COMPILED_AUDIT_FILENAME
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if not (row["problem_id"] == "p1" and row["seed"] == 0
                                       and row["checkpoint_multiplier_k"] == TOTAL_BUDGET)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    observation = data_loader.load(selection)[0]
    assert observation.oracle_attempts == ROWS[0].oracle_attempts - 1
    assert observation.oracle_successes == [count - 1 for count in ROWS[0].oracle_successes]


def test_missing_arm_fails(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Missing fitting data is an explicit error instead of a fallback to another arm."""
    missing = selection.model_copy(update={"run_id": "missing"})
    with pytest.raises(FileNotFoundError):
        data_loader.load(missing)


def test_framework_outputs(runner: FrameworkRunner, selection: InputSelection, banks: Path) -> None:
    """Persist parameters and the original DE allocation grid beneath each partition's run ID."""
    directories = runner.run(selection)
    assert {path.name for path in directories} == {"sg", "de"}
    for directory in directories:
        assert directory.parent == banks / DATASET / MODEL / selection.run_id / "frameworks"
        fitted = json.loads((directory / FIT_FILENAME).read_text())
        assert fitted["passing_score"] == 1
        assert set(fitted["parameters"]) == {row.problem_id for row in ROWS}
        predictions = [json.loads(line) for line in (directory / PREDICTIONS_FILENAME).read_text().splitlines()]
        assert len(predictions) == len(ROWS) * len(EXPECTED_ALLOCATIONS)
        for prediction in predictions:
            assert set(prediction) == {"dataset", "problem_id", "n", "k", "success_probability"}
            assert 0 <= prediction["success_probability"] <= 1


def test_cli_selects_competition_year_cross_product() -> None:
    """Explicit partition lists support joint prior fitting without domain or state-audit filters."""
    selection = parse_arguments(["--competition", "aime", "hmmt", "--year", "2024", "2025",
                                 "--model", MODEL, "--run-id", "test"])
    assert selection.competitions == ["aime", "hmmt"] and selection.years == [2024, 2025]
    assert selection.frameworks == ["sg", "de", "r-sg", "r-de"]
