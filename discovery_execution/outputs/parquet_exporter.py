"""Convert harness output files into verified Parquet files grouped by model, arm and seed."""

import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.harness.constants import META_FILENAME
from discovery_execution.outputs.constants import COMPRESSION, ROW_GROUP_SIZE, SCHEMA
from discovery_execution.outputs.models import Trajectory, seed_file_path, require_preserved_trajectories
from discovery_execution.outputs.native_reader import NativeTrajectoryReader
from discovery_execution.outputs.parquet_reader import ParquetTrajectoryReader


class ParquetTrajectoryExporter:
    """Use export_trajectories to convert local runs without changing their source files.

    NativeTrajectoryReader checks each run; _verify_export checks the written rows against those runs before _export_seed_file publishes the finished file.
    """

    def __init__(self, source: Path, destination: Path) -> None:
        """Read native runs under source and write Parquet files under destination."""
        self.source = source
        self.destination = destination
        self.reader = NativeTrajectoryReader(source)

    def export_trajectories(self, seeds: tuple[int, ...] | None) -> list[Path]:
        """Export selected seeds and return their Parquet paths; None selects all seeds.

        Reuse unchanged files and permit additional problems; reject changes or removals of existing trajectories.
        """
        groups: dict[Path, list[Path]] = defaultdict(list)
        if not self.source.is_dir():
            raise FileNotFoundError(self.source)
        for path in sorted(self.source.rglob(META_FILENAME)):
            metadata = json.loads(path.read_text())
            if seeds is not None and metadata["seed"] not in seeds:
                continue
            identity = path.parent.relative_to(self.source).as_posix()
            groups[seed_file_path(identity, metadata)].append(path.parent)
        if not groups:
            raise ValueError(f"no native output trajectories: {self.source}")
        return [self._export_seed_file(relative, paths) for relative, paths in sorted(groups.items())]

    def _export_seed_file(self, relative: Path, sources: list[Path]) -> Path:
        """Write sources to relative under the destination and return the verified file path."""
        destination = self.destination / relative
        if destination.exists() and self._reuse_export(destination, sources):
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        with pq.ParquetWriter(temporary, SCHEMA, compression=COMPRESSION) as writer:
            for start in range(0, len(sources), ROW_GROUP_SIZE):
                rows = [self.reader.read_trajectory(path).to_row() for path in sources[start:start + ROW_GROUP_SIZE]]
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        self._verify_export(temporary, sources)
        temporary.replace(destination)
        return destination

    def _reuse_export(self, path: Path, sources: list[Path]) -> bool:
        """Return whether path matches sources; reject updates that would lose existing trajectory content."""
        previous = ParquetTrajectoryReader(self.destination).trajectory_fingerprints(path)
        current = {}
        for source in sources:
            trajectory = self.reader.read_trajectory(source)
            current[trajectory.trajectory_id] = trajectory.fingerprint()
        require_preserved_trajectories(previous, current)
        return previous == current

    def _verify_export(self, path: Path, sources: list[Path]) -> None:
        """Read path back and require every row to match its corresponding native source."""
        offset = 0
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=ROW_GROUP_SIZE):
                for row in batch.to_pylist():
                    if offset >= len(sources) or Trajectory.from_row(row) != self.reader.read_trajectory(sources[offset]):
                        raise ValueError(f"Parquet round-trip differs: {path}")
                    offset += 1
        if offset != len(sources):
            raise ValueError(f"Parquet trajectory count differs: {path}")
