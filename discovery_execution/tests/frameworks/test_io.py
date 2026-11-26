"""Validate DELVE audit identities, fitting selection, and DE-compatible prediction output."""

import json
from dataclasses import asdict
from collections.abc import Iterator
from typing import Any
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.tests.frameworks.banks import read_records, write_records
from discovery_execution.frameworks.models import InputSelection
from discovery_execution.frameworks.runner import FrameworkRunner
from discovery_execution.frameworks.utils.constants import FIT_FILENAME, ORACLE_ARM, PREDICTIONS_FILENAME, SHORT_ARM, TOTAL_BUDGET
from discovery_execution.frameworks.utils.dataloader import DataLoader
from discovery_execution.cli.frameworks import parse_arguments
from discovery_execution.tests.frameworks.constants import DATASET, EXPECTED_ALLOCATIONS, MODEL, ROWS


def test_loader_uses_all_complete_seeds(data_loader: DataLoader, selection: InputSelection) -> None:
    """Fit counts match every input seed; long unaided and delayed oracle data are untouched."""
    assert [asdict(row) for row in data_loader.load(selection)] == [asdict(row) for row in ROWS]


def test_loader_pools_solution_mixtures(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Different seeds and problems can use different sketches without losing audited outcomes."""
    path = banks / DATASET / MODEL / ORACLE_ARM
    rows = read_records(path)
    for row in rows:
        row["solution_index"] = row["seed"] % 3
    write_records(path, rows)
    assert [asdict(row) for row in data_loader.load(selection)] == [asdict(row) for row in ROWS]


def test_loader_projects_audit_columns(banks: Path, data_loader: DataLoader, selection: InputSelection,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """DE fitting must not load the potentially large conversation column."""
    original = pq.ParquetFile.iter_batches
    projections = []
    def observed(self: pq.ParquetFile, *args: Any, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        """Capture projected columns while delegating actual reads to Arrow."""
        projections.append(kwargs["columns"])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", observed)
    data_loader.load(selection)
    assert projections and all("conversation" not in columns for columns in projections)


def test_framework_retains_earlier_success(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Convert raw checkpoint regressions to cumulative counts without changing the audit file."""
    path = banks / DATASET / MODEL / ORACLE_ARM
    rows = read_records(path)
    for row in rows:
        if row["problem_id"] == "p1" and row["seed"] == 0 and row["checkpoint_multiplier_k"] > 1:
            row["correctness"]["score"] = 0
    write_records(path, rows)
    content = read_records(path)
    assert data_loader.load(selection)[0].oracle_successes == ROWS[0].oracle_successes
    assert read_records(path) == content


@pytest.mark.parametrize("field,value", [
    ("solver_model", "qwen3-1.7b"), ("dataset", "aime/2025"),
    ("inject_sketch_at", 7), ("compute_multiplier_k", 7), ("correctness", {"score": 7}),
])
def test_loader_rejects_wrong_identity_or_nonbinary_grade(field: str, value: object, banks: Path,
                                                          data_loader: DataLoader, selection: InputSelection) -> None:
    """Misplaced or malformed oracle rows cannot silently change the execution estimate."""
    path = banks / DATASET / MODEL / ORACLE_ARM
    rows = read_records(path)
    rows[0][field] = value
    write_records(path, rows)
    with pytest.raises(ValueError):
        data_loader.load(selection)


def test_loader_rejects_duplicates(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Duplicate checkpoint rows must not double-count successful seeds."""
    path = banks / DATASET / MODEL / SHORT_ARM
    bank = path / "seed_0.parquet"
    table = pq.ParquetFile(bank).read()
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), bank)
    with pytest.raises(ValueError, match="duplicate"):
        data_loader.load(selection)


def test_incomplete_seed_is_excluded(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """A partially audited oracle seed is neither a success nor a failure observation."""
    path = banks / DATASET / MODEL / ORACLE_ARM
    rows = read_records(path)
    rows = [row for row in rows if not (row["problem_id"] == "p1" and row["seed"] == 0
                                       and row["checkpoint_multiplier_k"] == TOTAL_BUDGET)]
    write_records(path, rows)
    observation = data_loader.load(selection)[0]
    assert observation.oracle_attempts == ROWS[0].oracle_attempts - 1
    assert observation.oracle_successes == [count - 1 for count in ROWS[0].oracle_successes]


def test_missing_arm_fails(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Missing fitting data is an explicit error instead of a fallback to another arm."""
    missing = selection.model_copy(update={"years": [2025]})
    with pytest.raises(FileNotFoundError):
        data_loader.load(missing)


def test_framework_outputs(runner: FrameworkRunner, selection: InputSelection, banks: Path) -> None:
    """Persist parameters and the original DE allocation grid beneath each partition and model."""
    directories = runner.run(selection)
    assert {path.name for path in directories} == {"sg", "de"}
    for directory in directories:
        assert directory.parent == banks / DATASET / MODEL / "frameworks"
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
                                 "--model", MODEL])
    assert selection.competitions == ["aime", "hmmt"] and selection.years == [2024, 2025]
    assert selection.frameworks == ["sg", "de", "r-sg", "r-de"]
