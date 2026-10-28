"""Atomic JSON publication, checksums, and bounded Parquet shard writing."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.harness.models import Checkpoint, RunState
from discovery_execution.processing.constants import HASH_CHUNK_BYTES, PARQUET_COMPRESSION, PARQUET_SHARD_BYTES


def file_digest(path: Path) -> str:
    """Hash a file incrementally without loading a large shard into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete metadata record with a same-filesystem rename."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_parquet_atomic(path: Path, table: pa.Table) -> None:
    """Publish a complete Parquet shard without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(table, temporary, compression=PARQUET_COMPRESSION)
    temporary.replace(path)


def write_parquet_shards(rows: Iterable[dict[str, Any]], directory: Path) -> dict[str, str]:
    """Bound buffered serialized rows and return checksums keyed by relative shard path."""
    pending: list[dict[str, Any]] = []
    size, index = 0, 0
    checksums = {}
    for row in rows:
        pending.append(row)
        size += len(json.dumps(row).encode())
        if size >= PARQUET_SHARD_BYTES:
            path = directory / f"part-{index:05d}.parquet"
            write_parquet_atomic(path, pa.Table.from_pylist(pending))
            checksums[path.name] = file_digest(path)
            pending, size, index = [], 0, index + 1
    if pending:
        path = directory / f"part-{index:05d}.parquet"
        write_parquet_atomic(path, pa.Table.from_pylist(pending))
        checksums[path.name] = file_digest(path)
    return checksums


def checkpoint_snapshot(state: RunState, checkpoint: Checkpoint) -> dict[str, Any]:
    """Slice the saved conversation at an exact checkpoint without retaining future text."""
    messages = [dict(item) for item in state.messages[:checkpoint.message_count]]
    messages[-1]["content"] = messages[-1]["content"][:checkpoint.last_message_chars]
    return {"messages": messages, "output_tokens": checkpoint.output_tokens}
