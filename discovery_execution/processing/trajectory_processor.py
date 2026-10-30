"""Combine many small solver output files into reusable Parquet trajectory chunks.

Each row preserves the full conversation, run metadata, audit grades, and matching problem with solutions. Verify saved checkpoints before storing the conversation from which they can be reconstructed. Critic input selection happens later.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import zstandard

from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CHECKPOINT_FILENAME, CONVERSATION_FILENAME, META_FILENAME, OUTPUT_TOKENS_PER_BLOCK,
)
from discovery_execution.harness.models import RunState
from discovery_execution.processing.constants import MANIFEST_FILENAME, SPLITS, TRAJECTORIES_SUBDIRECTORY
from discovery_execution.processing.problem_splits import ProblemSplits
from discovery_execution.processing.storage import checkpoint_snapshot, file_digest, write_json_atomic, write_parquet_shards


class TrajectoryProcessor:
    """Pack one solver's raw runs into larger Parquet files.

    compile_trajectories writes the splits and manifest using ProblemSplits. _trajectory_row checks a run and preserves its metadata, conversation, audit, and problem in one row. It does not construct critic prompts.
    """

    def __init__(self, source: Path, destination: Path, solver_model: str, splits: ProblemSplits) -> None:
        """Store the source, destination, solver selection, and shared problem splits.

        Args:
            source: Root directory of raw solver outputs.
            destination: New processed directory for one solver model.
            solver_model: Solver identifier used to select raw runs.
            splits: Shared problem records and fixed train/validation/test assignments.

        Returns:
            None. Stores the paths and split provider for compile_trajectories.
        """
        self.source, self.destination = source, destination
        self.solver_model, self.splits = solver_model, splits

    def compile_trajectories(self, source_revision: str | None) -> None:
        """Pack complete solver runs into Parquet chunks and write their manifest.

        Args:
            source_revision: Repository@commit provenance for downloaded outputs, or None for local outputs.

        Returns:
            None. Writes trajectory shards and their manifest, saving shared problem assignments if they do not exist.
        """
        if self.destination.exists():
            raise FileExistsError(self.destination)
        partitions: dict[str, list[Path]] = {split: [] for split in SPLITS}
        for path in sorted(self.source.rglob(META_FILENAME)):
            metadata = json.loads(path.read_text())
            if metadata["model"] == self.solver_model:
                key = f"{metadata['competition']}/{metadata['year']}/{metadata['problem_id']}"
                partitions[self.splits.assignments[key]].append(path.parent)
        if not any(partitions.values()):
            raise ValueError("no trajectories for selected model")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.destination.parent) as temporary:
            staging = Path(temporary) / self.solver_model
            staging.mkdir()
            checksums = {}
            for split, paths in partitions.items():
                shards = write_parquet_shards((self._trajectory_row(path, split) for path in paths), staging / split / TRAJECTORIES_SUBDIRECTORY)
                checksums.update({f"{split}/{TRAJECTORIES_SUBDIRECTORY}/{name}": digest for name, digest in shards.items()})
            self.splits.save_assignments()
            manifest = {"solver_model": self.solver_model, "source_revision": source_revision,
                        "split_sha256": file_digest(self.splits.path), "trajectories": checksums}
            write_json_atomic(staging / MANIFEST_FILENAME, manifest)
            staging.rename(self.destination)

    def _trajectory_row(self, directory: Path, split: str) -> dict[str, Any]:
        """Validate one saved run and combine its conversation, metadata, audit, and problem.

        Args:
            directory: Raw directory containing one run's metadata, conversation, audit, and checkpoint snapshots.
            split: Already assigned train/validation/test partition for the run's problem.

        Returns:
            dict[str, Any]: Trajectory/problem IDs, split, and serialized metadata, conversation, audit, and problem record.
        """
        metadata_text = (directory / META_FILENAME).read_text()
        conversation_text = zstandard.decompress((directory / CONVERSATION_FILENAME).read_bytes()).decode()
        audit_text = (directory / AUDIT_FILENAME).read_text()
        metadata, audit = json.loads(metadata_text), json.loads(audit_text)
        state = RunState.model_validate_json(conversation_text)
        budget = metadata["compute_multiplier_k"] * OUTPUT_TOKENS_PER_BLOCK
        if not audit["complete"] or audit["output_tokens"] != budget or state.output_tokens != budget:
            raise ValueError(f"incomplete branch audit or conversation: {directory}")
        for boundary in state.checkpoints:
            block = boundary.output_tokens // OUTPUT_TOKENS_PER_BLOCK
            snapshot = json.loads(zstandard.decompress((directory / CHECKPOINT_FILENAME.format(k=block)).read_bytes()))
            if snapshot != checkpoint_snapshot(state, boundary):
                raise ValueError(f"snapshot disagrees with conversation: {directory}")
        key = f"{metadata['competition']}/{metadata['year']}/{metadata['problem_id']}"
        return {"trajectory_id": str(directory.relative_to(self.source)), "problem_key": key, "split": split,
                "metadata": metadata_text, "conversation": conversation_text, "audit": audit_text,
                "problem": self.splits.problems[key].model_dump_json()}
