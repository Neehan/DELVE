"""Assign whole problems once, with exact training size and proportional competition coverage."""

import hashlib
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.harness.models import Problem
from discovery_execution.processing.constants import SPLIT_FILENAME, SPLIT_SEED, SPLITS, TEST_YEAR
from discovery_execution.processing.storage import write_parquet_atomic


class ProblemSplits:
    """Load problems and persist a fixed assignment shared by every solver model."""

    def __init__(self, datasets_directory: Path, processed_directory: Path, training_count: int) -> None:
        """Require unique identities and enough older problems for training and validation."""
        self.path = processed_directory / SPLIT_FILENAME
        self.problems: dict[str, Problem] = {}
        for path in sorted(datasets_directory.glob("*/*.jsonl")):
            for line in path.read_text().splitlines():
                problem = Problem.model_validate_json(line)
                key = f"{problem.competition}/{problem.year}/{problem.problem_id}"
                if key in self.problems:
                    raise ValueError(f"duplicate problem: {key}")
                self.problems[key] = problem
        if not self.problems:
            raise ValueError("no dataset problems")
        self.assignments = self._load_or_assign(training_count)

    def _load_or_assign(self, training_count: int) -> dict[str, str]:
        """Reuse published assignments and reject a changed problem universe or split size."""
        if self.path.exists():
            rows = pq.read_table(self.path).to_pylist()
            assignments = {row["problem_key"]: row["split"] for row in rows}
            if len(assignments) != len(rows) or assignments.keys() != self.problems.keys():
                raise ValueError("saved splits and dataset problem identities differ")
            if sum(split == "train" for split in assignments.values()) != training_count:
                raise ValueError("saved training split size differs")
            for key, split in assignments.items():
                if split not in SPLITS or (split == "test") != (self.problems[key].year >= TEST_YEAR):
                    raise ValueError(f"invalid saved split: {key}")
            return assignments
        return self._assign_problems(training_count)

    def _assign_problems(self, training_count: int) -> dict[str, str]:
        """Allocate exact training counts proportionally, then rank problems with seed 1234."""
        groups: dict[str, list[str]] = defaultdict(list)
        assignments = {}
        for key, problem in self.problems.items():
            if problem.year >= TEST_YEAR:
                assignments[key] = "test"
            else:
                groups[problem.competition].append(key)
        total = sum(map(len, groups.values()))
        if not 0 < training_count < total:
            raise ValueError("training count must leave nonempty training and validation sets")
        quotas = {name: training_count * len(keys) / total for name, keys in groups.items()}
        counts = {name: int(quota) for name, quota in quotas.items()}
        ranked = sorted(groups, key=lambda name: (-(quotas[name] - counts[name]), name))
        for name in ranked[:training_count - sum(counts.values())]:
            counts[name] += 1
        for name, keys in groups.items():
            keys.sort(key=lambda key: hashlib.sha256(f"{SPLIT_SEED}:{key}".encode()).digest())
            assignments.update({key: "train" if index < counts[name] else "validation" for index, key in enumerate(keys)})
        return assignments

    def save_assignments(self) -> None:
        """Publish the assignment only once; later model preparation reuses the same file."""
        if not self.path.exists():
            rows = [{"problem_key": key, "split": split} for key, split in sorted(self.assignments.items())]
            write_parquet_atomic(self.path, pa.Table.from_pylist(rows))
