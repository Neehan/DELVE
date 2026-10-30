"""Assign problems to fixed train, validation, and test splits reused by every solver model."""

import hashlib
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.harness.models import Problem
from discovery_execution.processing.constants import SPLIT_FILENAME, SPLIT_SEED, SPLITS, TEST_YEAR
from discovery_execution.processing.storage import write_parquet_atomic


class ProblemSplits:
    """Keep every trajectory of a problem in the same split across solver models.

    problems holds the loaded problem records; assignments maps their keys to splits. Existing assignments are reused, or new ones are computed with fixed seeded selection and proportional competition coverage. save_assignments writes splits.parquet.
    """

    def __init__(self, datasets_directory: Path, processed_directory: Path, training_count: int) -> None:
        """Load problems and reuse or create their split assignments.

        Args:
            datasets_directory: Root containing competition/year JSONL problem files.
            processed_directory: Shared processing root containing splits.parquet.
            training_count: Exact number of pre-test-year problems assigned to training.

        Returns:
            None. Loads problem records and either reuses saved assignments or computes new assignments in memory.
        """
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
        """Reuse saved splits only if the problem IDs and requested training count still match.

        Args:
            training_count: Required number of training problems; must agree with an existing split file.

        Returns:
            dict[str, str]: Mapping from competition/year/problem_id keys to train, validation, or test.
        """
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
        """Hold out test-year problems and divide older problems between training and validation.

        Args:
            training_count: Exact training quota, leaving at least one older problem for validation.

        Returns:
            dict[str, str]: Deterministic assignments; test-year problems are held out and older training quotas are proportional to competition size.
        """
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
        """Save splits.parquet if it does not already exist.

        Returns:
            None. Writes splits.parquet when absent; an existing assignment file is retained.
        """
        if not self.path.exists():
            rows = [{"problem_key": key, "split": split} for key, split in sorted(self.assignments.items())]
            write_parquet_atomic(self.path, pa.Table.from_pylist(rows))
