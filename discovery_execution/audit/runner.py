"""Discover saved trajectories and regrade their checkpoints without run code."""

import json
from pathlib import Path

from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.audit.compiler import AuditCompiler
from discovery_execution.dataset import Dataset
from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CHECKPOINT_FILENAME, RUN_FILENAME,
)
from discovery_execution.harness.storage import Storage, write_json
from discovery_execution.harness.models import DatasetConfig


class AuditRunner:
    """Run offline grading on a saved output subtree, optionally filtering problem IDs and seeds. No overrides are required."""

    def run(self, directory: Path, problems: list[str] | None, seeds: list[int] | None) -> None:
        """Validate snapshots and local dataset references before writing any scores."""
        paths = self._select(directory, problems, seeds)
        partitions: dict[tuple[str, str, int], dict[str, str]] = {}
        references: dict[Path, str] = {}
        for path in paths:
            for filename in (RUN_FILENAME, CHECKPOINT_FILENAME):
                if not (path / filename).is_file():
                    raise FileNotFoundError(path / filename)
            settings = json.loads((path / RUN_FILENAME).read_text(encoding="utf-8"))
            key = (settings["datasets_directory"], settings["competition"], settings["year"])
            if key not in partitions:
                config = DatasetConfig(datasets_directory=Path(key[0]), competition=key[1], year=key[2],
                                       dataset_repo_url=None)
                partitions[key] = {problem.problem_id: problem.final_answer for problem in Dataset(config)}
            problem_id = settings["problem_id"]
            if problem_id not in partitions[key]:
                raise ValueError(f"problem not found in local dataset: {problem_id}")
            references[path] = partitions[key][problem_id]
        for path in paths:
            self._grade(path, references[path])
        for arm_directory in sorted({path.parent.parent for path in paths}):
            AuditCompiler(arm_directory).compile()

    def _select(self, directory: Path, problems: list[str] | None, seeds: list[int] | None) -> list[Path]:
        """Filter immutable run metadata and reject empty or unmatched selections."""
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        selected = []
        found_problems: set[str] = set()
        found_seeds: set[int] = set()
        for path in sorted(directory.rglob(RUN_FILENAME)):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            problem, seed = metadata["problem_id"], metadata["seed"]
            if problems is not None and problem not in problems:
                continue
            if seeds is not None and seed not in seeds:
                continue
            selected.append(path.parent)
            found_problems.add(problem)
            found_seeds.add(seed)
        if problems is not None and set(problems) - found_problems:
            raise ValueError(f"no saved runs for problems: {sorted(set(problems) - found_problems)}")
        if seeds is not None and set(seeds) - found_seeds:
            raise ValueError(f"no saved runs for seeds: {sorted(set(seeds) - found_seeds)}")
        if not selected:
            raise ValueError(f"no saved runs to audit: {directory}")
        return selected

    def _grade(self, directory: Path, final_answer: str) -> None:
        """Grade saved snapshots under the seed lock using the dataset's reference answer."""
        with Storage(directory) as storage:
            grader = AnswerGrader(final_answer)
            settings = storage.read_run()
            result = grader.grade_checkpoints(storage.read_checkpoints(), settings["compute_multiplier_k"])
            write_json(directory / AUDIT_FILENAME, result)
