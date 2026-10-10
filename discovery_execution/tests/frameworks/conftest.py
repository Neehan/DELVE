"""Reusable compiled binary audit banks for framework integration tests."""

from pathlib import Path
from typing import Literal

import pytest

from discovery_execution.audit.constants import COMPILED_AUDIT_FILENAME
from discovery_execution.audit.models import AuditRecord, BinaryGrade
from discovery_execution.frameworks.models import InputSelection
from discovery_execution.frameworks.runner import FrameworkRunner
from discovery_execution.frameworks.utils.constants import ORACLE_ARM, SHORT_ARM, TOTAL_BUDGET
from discovery_execution.frameworks.utils.dataloader import DataLoader
from discovery_execution.cli.frameworks import parse_arguments
from discovery_execution.tests.frameworks.constants import DATASET, MODEL, ROWS


@pytest.fixture
def selection() -> InputSelection:
    """Select one disposable partition and both unregularized estimators."""
    competition, year = DATASET.split("/")
    return parse_arguments(["--competition", competition, "--year", year, "--model", MODEL,
                            "--frameworks", "sg", "de"])


@pytest.fixture
def banks(tmp_path: Path) -> Path:
    """Write complete short/oracle observations and deliberately unrelated long/delayed arms."""
    root = tmp_path / "output" / DATASET / MODEL
    runs: list[tuple[str, Literal["unaided", "oracle-execution"], int, int]] = [
        (SHORT_ARM, "unaided", 1, 0), (ORACLE_ARM, "oracle-execution", TOTAL_BUDGET, 0),
    ]
    for arm, experiment, horizon, injection in runs:
        directory = root / arm
        directory.mkdir(parents=True)
        records = []
        for observation in ROWS:
            trials = observation.short_attempts if experiment == "unaided" else observation.oracle_attempts
            for seed in range(trials):
                for checkpoint in range(1, horizon + 1):
                    count = observation.short_successes if experiment == "unaided" else observation.oracle_successes[checkpoint - 1]
                    records.append(AuditRecord(dataset=DATASET, solver_model=MODEL,
                                               experiment=experiment, compute_multiplier_k=horizon, inject_sketch_at=injection,
                                               solution_index=0, problem_id=observation.problem_id, seed=seed,
                                               checkpoint_multiplier_k=checkpoint, correctness=BinaryGrade(score=1 if seed < count else 0)))
        (directory / COMPILED_AUDIT_FILENAME).write_text("".join(row.model_dump_json() + "\n" for row in records))
    for arm in ("unaided-8x", "oracle-at-7x-execution"):
        (root / arm).mkdir()
        (root / arm / COMPILED_AUDIT_FILENAME).write_text("not a fitting input\n")
    return tmp_path / "output"


@pytest.fixture
def data_loader(banks: Path) -> DataLoader:
    """Read only the temporary compiled audit files."""
    return DataLoader(banks)


@pytest.fixture
def runner(data_loader: DataLoader, banks: Path) -> FrameworkRunner:
    """Use production fitting and output writing with isolated audit observations."""
    return FrameworkRunner(data_loader, banks)
