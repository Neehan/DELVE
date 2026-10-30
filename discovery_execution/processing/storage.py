"""Read and write processed data, check file hashes, and reconstruct histories at saved checkpoints."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict
import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.harness.models import Checkpoint, RunState
from discovery_execution.processing.constants import HASH_CHUNK_BYTES, MANIFEST_FILENAME, PARQUET_COMPRESSION, PARQUET_SHARD_BYTES


def file_digest(path: Path) -> str:
    """Compute a file hash without reading the whole file into RAM.

    Args:
        path: File to hash; large files are read in fixed-size chunks.

    Returns:
        str: SHA-256 hexadecimal digest of the file bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write JSON to a temporary file, then replace the destination with the completed file.

    Args:
        path: Target JSON path in an existing directory.
        value: JSON-serializable metadata dictionary to save.

    Returns:
        None. Writes a temporary file and renames it over the target only after serialization completes.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_parquet_atomic(path: Path, table: pa.Table) -> None:
    """Write compressed Parquet to a temporary file, then publish the completed file.

    Args:
        path: Target Parquet path; parent directories are created as needed.
        table: Arrow table containing the complete rows for this file.

    Returns:
        None. Writes compressed Parquet to a temporary file, then renames it to the target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(table, temporary, compression=PARQUET_COMPRESSION)
    temporary.replace(path)


def write_parquet_shards(rows: Iterable[dict[str, Any]], directory: Path) -> dict[str, str]:
    """Group trajectory rows into files near the configured size and record their hashes.

    Args:
        rows: Iterable of complete trajectory dictionaries to serialize.
        directory: Destination for numbered part-*.parquet files.

    Returns:
        dict[str, str]: Written filenames mapped to SHA-256 digests; files are grouped by approximate serialized byte size.
    """
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
    """Rebuild the conversation as it appeared at one saved checkpoint.

    Args:
        state: Complete saved conversation and its checkpoint boundaries.
        checkpoint: Boundary specifying the message count, final-message character limit, and output-token count.

    Returns:
        dict[str, Any]: Messages sliced at that boundary and the checkpoint's output_tokens value.
    """
    messages = [dict(item) for item in state.messages[:checkpoint.message_count]]
    messages[-1]["content"] = messages[-1]["content"][:checkpoint.last_message_chars]
    return {"messages": messages, "output_tokens": checkpoint.output_tokens}


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read settings and file hashes for a processed model directory.

    Args:
        directory: Model root containing manifest.json.

    Returns:
        dict[str, Any]: Recorded trajectory, tokenization, and optional hidden-state metadata.
    """
    return json.loads((directory / MANIFEST_FILENAME).read_text())


def load_parquet_splits(directory: Path, splits: tuple[str, ...], section: dict[str, Any], keep_in_memory: bool) -> DatasetDict:
    """Verify file hashes and open the requested token or hidden-state splits.

    Args:
        directory: Processed model root.
        splits: Requested dataset partitions.
        section: Manifest section containing completion status and chunk checksums.
        keep_in_memory: Keep Arrow data in RAM for small vectors; otherwise use memory mapping.

    Returns:
        DatasetDict: Requested partitions with unique critic example IDs; rejects incomplete or changed inputs.
    """
    if not section["complete"]:
        raise ValueError("incomplete input preparation")
    paths: dict[str, list[str]] = {}
    for name, digest in section["shards"].items():
        split = Path(name).parts[0]
        if split in splits:
            path = directory / name
            if file_digest(path) != digest:
                raise ValueError(f"prepared input file changed: {path}")
            paths.setdefault(split, []).append(str(path))
    datasets = DatasetDict({split: Dataset.from_parquet(cast(Any, files), keep_in_memory=keep_in_memory)
                            for split, files in paths.items()})
    for dataset in datasets.values():
        identifiers = list(dataset["prefix_id"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate critic example IDs")
    return datasets
