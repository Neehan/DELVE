"""Read complete trajectories for critic setup or only checkpoint grades for DE fitting."""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from discovery_execution.audit.models import AuditRecord
from discovery_execution.audit.answer_grader import validate_grading_rule
from discovery_execution.outputs.audit_records import iter_audit_records
from discovery_execution.outputs.constants import AUDIT_COLUMNS, BATCH_SIZE, SCHEMA, SEED_PATTERN
from discovery_execution.outputs.models import Trajectory, seed_file_path, checkpoint_grades, identity_columns


class ParquetTrajectoryReader:
    """Expose iter_trajectories for critic setup and iter_audits for framework fitting.

    Both methods use _read_rows to check the schema and read bounded batches.
    """

    def __init__(self, root: Path) -> None:
        """Use root as the directory containing competition/year/model/arm Parquet files."""
        self.root = root

    def iter_trajectories(self, model: str, seeds: tuple[int, ...] | None, arms: tuple[str, ...] | None) -> Iterator[Trajectory]:
        """Yield trajectories for model, seeds and arms; None includes every seed or arm.

        Reject duplicate IDs and rows whose identity disagrees with their file path.
        """
        paths = sorted(self.root.glob(f"*/*/{model}/**/{SEED_PATTERN}"))
        if not paths:
            raise FileNotFoundError(f"no trajectory Parquet files for {model}: {self.root}")
        for path in paths:
            if arms is not None and path.relative_to(self.root).parts[3] not in arms:
                continue
            if seeds is not None and int(path.stem.removeprefix("seed_")) not in seeds:
                continue
            yield from self._read_trajectories(path)

    def trajectory_fingerprints(self, path: Path) -> dict[str, str]:
        """Validate one Parquet file at path and return IDs with complete trajectory content digests."""
        return {trajectory.trajectory_id: trajectory.fingerprint() for trajectory in self._read_trajectories(path)}

    def _read_trajectories(self, path: Path) -> Iterator[Trajectory]:
        """Yield validated trajectories from path, rejecting duplicate IDs and misplaced rows."""
        seen = set()
        for row in self._read_rows(path, None):
            trajectory = Trajectory.from_row(row)
            if self.root / trajectory.seed_file_path() != path or trajectory.trajectory_id in seen:
                raise ValueError(f"duplicate or misplaced trajectory: {path}")
            seen.add(trajectory.trajectory_id)
            yield trajectory

    def iter_audits(self, directory: Path) -> Iterator[AuditRecord]:
        """Yield AuditRecords from one arm directory without reading conversation text."""
        paths = sorted(directory.glob(SEED_PATTERN))
        if not paths:
            raise FileNotFoundError(f"no trajectory Parquet files: {directory}")
        for path in paths:
            for row in self._read_rows(path, list(AUDIT_COLUMNS)):
                metadata, audit = json.loads(row["metadata"]), json.loads(row["audit"])
                # Direct Parquet fitting must enforce the same grading rule as native audit compilation.
                validate_grading_rule(audit, row["trajectory_id"])
                identity = identity_columns(row["trajectory_id"], metadata)
                if any(row[key] != value for key, value in identity.items()) or row["grades"] != checkpoint_grades(audit):
                    raise ValueError(f"Parquet audit columns disagree: {path}")
                if self.root / seed_file_path(row["trajectory_id"], metadata) != path:
                    raise ValueError(f"audit identity does not match {path}")
                yield from iter_audit_records(metadata, audit)

    def _read_rows(self, path: Path, columns: list[str] | None) -> Iterator[dict[str, Any]]:
        """Yield row dictionaries from path, reading only columns when supplied.

        None reads every column. Reject files with a different schema or version.
        """
        with pq.ParquetFile(path) as parquet:
            if not parquet.schema_arrow.equals(SCHEMA, check_metadata=True):
                raise ValueError(f"unsupported output Parquet schema: {path}")
            for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
                yield from batch.to_pylist()
