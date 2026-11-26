"""Publish validated native trajectories as one Parquet bank per arm and seed."""

import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.harness.constants import META_FILENAME
from discovery_execution.outputs.constants import COMPRESSION, ROW_GROUP_SIZE, SCHEMA
from discovery_execution.outputs.models import Trajectory, bank_path
from discovery_execution.outputs.native import NativeReader


class OutputPacker:
    """Pack complete local artifacts and independently round-trip every written trajectory."""

    def __init__(self, source: Path, destination: Path) -> None:
        """Keep raw generation files and the new published representation in separate roots."""
        self.source, self.destination = source, destination
        self.reader = NativeReader(source)

    def pack(self, seeds: tuple[int, ...] | None) -> list[Path]:
        """Group runs by arm/seed and publish a bank only after exact semantic verification."""
        groups: dict[Path, list[Path]] = defaultdict(list)
        if not self.source.is_dir():
            raise FileNotFoundError(self.source)
        for path in sorted(self.source.rglob(META_FILENAME)):
            metadata = json.loads(path.read_text())
            if seeds is not None and metadata["seed"] not in seeds:
                continue
            identity = path.parent.relative_to(self.source).as_posix()
            groups[bank_path(identity, metadata)].append(path.parent)
        if not groups:
            raise ValueError(f"no native output trajectories: {self.source}")
        return [self._write(relative, paths) for relative, paths in sorted(groups.items())]

    def _write(self, relative: Path, sources: list[Path]) -> Path:
        """Bound memory to a row group and compare all restored rows with their native source."""
        destination = self.destination / relative
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        with pq.ParquetWriter(temporary, SCHEMA, compression=COMPRESSION) as writer:
            for start in range(0, len(sources), ROW_GROUP_SIZE):
                rows = [self.reader.read(path).to_row() for path in sources[start:start + ROW_GROUP_SIZE]]
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        restored = pq.ParquetFile(temporary).iter_batches(batch_size=ROW_GROUP_SIZE)
        offset = 0
        for batch in restored:
            for row in batch.to_pylist():
                if Trajectory.from_row(row) != self.reader.read(sources[offset]):
                    raise ValueError(f"Parquet round-trip differs: {sources[offset]}")
                offset += 1
        if offset != len(sources):
            raise ValueError(f"Parquet trajectory count differs: {relative}")
        temporary.replace(destination)
        return destination
