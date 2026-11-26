"""Buffer a single split's trajectory rows into bounded Parquet shards."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from discovery_execution.processing.constants import PARQUET_SHARD_BYTES
from discovery_execution.processing.storage import file_digest, write_parquet_atomic


class TrajectoryShardWriter:
    """Append rows as they arrive; finish writes the final shard and returns its file hashes."""

    def __init__(self, directory: Path) -> None:
        """Write shards under directory while buffering at most approximately PARQUET_SHARD_BYTES."""
        self.directory = directory
        self.pending: list[dict[str, Any]] = []
        self.pending_bytes = 0
        self.checksums: dict[str, str] = {}

    def append(self, row: dict[str, Any]) -> None:
        """Buffer row and flush when the estimated serialized size reaches the shard limit."""
        self.pending.append(row)
        self.pending_bytes += len(json.dumps(row).encode())
        if self.pending_bytes >= PARQUET_SHARD_BYTES:
            self._flush()

    def finish(self) -> dict[str, str]:
        """Write remaining rows and return shard filenames mapped to SHA-256 digests."""
        self._flush()
        return dict(self.checksums)

    def _flush(self) -> None:
        """Write one buffered shard, record its hash and release its rows."""
        if not self.pending:
            return
        path = self.directory / f"part-{len(self.checksums):05d}.parquet"
        write_parquet_atomic(path, pa.Table.from_pylist(self.pending))
        self.checksums[path.name] = file_digest(path)
        self.pending.clear()
        self.pending_bytes = 0
