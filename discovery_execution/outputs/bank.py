"""Read versioned per-seed Parquet banks without expanding native file trees."""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from discovery_execution.audit.models import AuditRecord
from discovery_execution.outputs.audits import audit_records
from discovery_execution.outputs.constants import AUDIT_COLUMNS, BATCH_SIZE, SCHEMA, SEED_PATTERN
from discovery_execution.outputs.models import Trajectory, bank_path, checkpoint_grades, identity_columns


class OutputBank:
    """Share bounded Parquet reads between trajectory processing and grade-only DE fitting."""

    def __init__(self, root: Path) -> None:
        """Store the root containing contest/year/model/arm seed banks."""
        self.root = root

    def iter_trajectories(self, model: str, seeds: tuple[int, ...] | None) -> Iterator[Trajectory]:
        """Read one solver's banks and reject mislabeled or duplicate trajectory identities."""
        paths = sorted(self.root.glob(f"*/*/{model}/**/{SEED_PATTERN}"))
        if not paths:
            raise FileNotFoundError(f"no Parquet output banks for {model}: {self.root}")
        for path in paths:
            if seeds is not None and int(path.stem.removeprefix("seed_")) not in seeds:
                continue
            seen = set()
            for row in self._rows(path, None):
                trajectory = Trajectory.from_row(row)
                if self.root / trajectory.bank_path() != path or trajectory.trajectory_id in seen:
                    raise ValueError(f"duplicate or misplaced trajectory: {path}")
                seen.add(trajectory.trajectory_id)
                yield trajectory

    def iter_audits(self, directory: Path) -> Iterator[AuditRecord]:
        """Read only metadata and grades, leaving potentially large conversations on disk."""
        paths = sorted(directory.glob(SEED_PATTERN))
        if not paths:
            raise FileNotFoundError(f"no Parquet audit banks: {directory}")
        for path in paths:
            for row in self._rows(path, list(AUDIT_COLUMNS)):
                metadata, audit = json.loads(row["metadata"]), json.loads(row["audit"])
                identity = identity_columns(row["trajectory_id"], metadata)
                if any(row[key] != value for key, value in identity.items()) or row["grades"] != checkpoint_grades(audit):
                    raise ValueError(f"Parquet audit columns disagree: {path}")
                if self.root / bank_path(row["trajectory_id"], metadata) != path:
                    raise ValueError(f"audit identity does not match {path}")
                yield from audit_records(metadata, audit)

    def _rows(self, path: Path, columns: list[str] | None) -> Iterator[dict[str, Any]]:
        """Require the exact schema version and stream small record batches."""
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(SCHEMA, check_metadata=True):
            raise ValueError(f"unsupported output Parquet schema: {path}")
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            yield from batch.to_pylist()
