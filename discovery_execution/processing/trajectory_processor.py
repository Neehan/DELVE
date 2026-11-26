"""Prepare fixed training splits directly from published per-seed trajectory banks."""

import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from discovery_execution.audit.answer_grader import validate_grading_rule
from discovery_execution.outputs.parquet_reader import ParquetTrajectoryReader
from discovery_execution.outputs.models import Trajectory
from discovery_execution.processing.constants import MANIFEST_FILENAME, SPLITS, TRAJECTORIES_SUBDIRECTORY
from discovery_execution.processing.problem_splits import ProblemSplits
from discovery_execution.processing.storage import file_digest, write_json_atomic
from discovery_execution.processing.trajectory_shard_writer import TrajectoryShardWriter


class TrajectoryProcessor:
    """Reuse one public output reader while retaining existing critic inputs and split semantics."""

    def __init__(self, source: Path, destination: Path, solver_model: str, splits: ProblemSplits) -> None:
        """Inject published banks, the processed destination, and fixed problem assignments."""
        self.source, self.destination = source, destination
        self.solver_model, self.splits = solver_model, splits

    def compile_trajectories(self, source_revision: str | None, seeds: tuple[int, ...], arms: tuple[str, ...] | None) -> None:
        """Read seeds and arms once into fixed splits, recording source_revision; None selects every arm."""
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.destination.parent) as temporary:
            staging = Path(temporary) / self.solver_model
            staging.mkdir()
            writers = {split: TrajectoryShardWriter(staging / split / TRAJECTORIES_SUBDIRECTORY) for split in SPLITS}
            for row in self._rows(seeds, arms):
                writers[row["split"]].append(row)
            checksums = {}
            for split, writer in writers.items():
                shards = writer.finish()
                checksums.update({f"{split}/{TRAJECTORIES_SUBDIRECTORY}/{name}": digest for name, digest in shards.items()})
            if not checksums:
                raise ValueError("no trajectories for selected model and seeds")
            self.splits.save_assignments()
            manifest = {"solver_model": self.solver_model, "source_revision": source_revision,
                        "split_sha256": file_digest(self.splits.path), "trajectories": checksums}
            write_json_atomic(staging / MANIFEST_FILENAME, manifest)
            staging.rename(self.destination)

    def _rows(self, seeds: tuple[int, ...], arms: tuple[str, ...] | None) -> Iterator[dict[str, Any]]:
        """Yield rows for seeds and arms, assigning each problem to its saved split in one pass."""
        for trajectory in ParquetTrajectoryReader(self.source).iter_trajectories(self.solver_model, seeds, arms):
            metadata = json.loads(trajectory.metadata)
            key = f"{metadata['competition']}/{metadata['year']}/{metadata['problem_id']}"
            yield self._trajectory_row(trajectory, self.splits.assignments[key], key)

    def _trajectory_row(self, trajectory: Trajectory, split: str, key: str) -> dict[str, Any]:
        """Retain raw audit policy and conversation boundaries for downstream token reconstruction."""
        validate_grading_rule(json.loads(trajectory.audit), trajectory.trajectory_id)
        return {"trajectory_id": trajectory.trajectory_id, "problem_key": key, "split": split,
                "metadata": trajectory.metadata, "conversation": trajectory.state.model_dump_json(),
                "audit": trajectory.audit, "problem": self.splits.problems[key].model_dump_json()}
