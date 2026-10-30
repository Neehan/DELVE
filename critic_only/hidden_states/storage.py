"""Save and reload hidden-state chunks, checking that they match the original token inputs."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

from critic_only.constants import CPU_ATTENTION, DEFAULT_ATTENTION, FEATURE_SUBDIRECTORY
from discovery_execution.processing.constants import MANIFEST_FILENAME
from discovery_execution.processing.storage import file_digest, load_parquet_splits, read_manifest, write_json_atomic, write_parquet_atomic


class HiddenStateStorage:
    """Own the hidden-state files and their saved encoding settings.

    prepare_encoding checks restart settings. save_chunk writes vectors; validate_chunk checks existing files before reuse. complete_cache records completion, and load_splits loads vectors for training.
    """

    def __init__(self, directory: Path) -> None:
        """Choose the processed model directory containing tokens and hidden states.

        Args:
            directory: Model root shared with the token input storage.

        Returns:
            None. Stores the path without loading model weights or data.
        """
        self.directory = directory

    def prepare_encoding(self, cpu: bool) -> None:
        """Save encoding settings or check that they match an earlier interrupted run.

        Args:
            cpu: Whether encoding uses CPU FP32/SDPA instead of GPU BF16/FlashAttention-2.

        Returns:
            None. Stores the token-cache identity, precision, attention, and pooling settings; rejects incompatible restarts.
        """
        path = self.directory / MANIFEST_FILENAME
        manifest = read_manifest(self.directory)
        tokens = manifest["tokenization"]
        if not tokens["complete"]:
            raise ValueError("setup has not completed tokenization")
        token_signature = hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest()
        settings = {"tokenization_sha256": token_signature, "precision": "float32" if cpu else "bfloat16",
                    "attention": CPU_ATTENTION if cpu else DEFAULT_ATTENTION,
                    "pooling": "last_non_padding_token", "hidden_size": tokens["settings"]["hidden_size"]}
        signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        if "features" in manifest and manifest["features"]["signature"] != signature:
            raise ValueError("hidden-state encoding settings differ")
        if "features" not in manifest:
            manifest["features"] = {"signature": signature, "settings": settings, "complete": False, "shards": {}}
            write_json_atomic(path, manifest)

    def chunk_path(self, token_path: str) -> Path:
        """Find the output path with the same split and filename as a token chunk.

        Args:
            token_path: Model-relative path to a tokenized Parquet file.

        Returns:
            Path: Same split and filename under hidden_states instead of tokenized.
        """
        source = Path(token_path)
        return self.directory / source.parent.parent / FEATURE_SUBDIRECTORY / source.name

    def has_chunk(self, token_path: str) -> bool:
        """Check whether the matching hidden-state file exists.

        Args:
            token_path: Model-relative token chunk identity.

        Returns:
            bool: Whether the corresponding output file exists; validate_chunk checks its contents separately.
        """
        return self.chunk_path(token_path).exists()

    def save_chunk(self, token_path: str, records: list[dict[str, Any]], signature: str) -> None:
        """Save vectors and labels in the original token-row order.

        Args:
            token_path: Identity of the input chunk, defining the output filename and partition.
            records: Encoded examples in original input order, with nullable reference vectors.
            signature: Identity of the frozen computation recorded in schema metadata.

        Returns:
            None. Writes float32 fixed-size vectors to compressed Parquet; refuses to overwrite an existing chunk.
        """
        path = self.chunk_path(token_path)
        if path.exists():
            raise FileExistsError(path)
        dimension = len(records[0]["hidden_state"])
        table = pa.Table.from_pylist(records)
        vector = pa.list_(pa.float32(), dimension)
        for name in ("hidden_state", "privileged_hidden_state"):
            table = table.set_column(table.schema.get_field_index(name), name,
                                     pa.array([row[name] for row in records], type=vector))
        table = table.replace_schema_metadata({"signature": signature})
        write_parquet_atomic(path, table)

    def validate_chunk(self, token_path: str, inputs: Dataset, signature: str) -> None:
        """Check that an existing hidden-state file still matches its token inputs.

        Compare the encoding signature, row count, example IDs, token hashes, labels, oracle flags, and remaining budgets. Require the expected ordinary/reference vectors to be present and finite. A mismatch raises ValueError instead of replacing an existing file.

        Args:
            token_path: Token chunk identity whose paired hidden-state file is considered for reuse.
            inputs: Ordered tokenized examples that should correspond to this file.
            signature: Required encoding identity for the existing vectors.

        Returns:
            None. Raises ValueError if the saved identity, labels, view availability, or finite-vector checks fail.
        """
        path = self.chunk_path(token_path)
        table = pq.read_table(path)
        if (table.schema.metadata or {}).get(b"signature") != signature.encode() or table.num_rows != len(inputs):
            raise ValueError(f"feature shard configuration differs: {path}")
        metadata = ["prefix_id", "problem_key", "input_hash", "privileged_input_hash", "labels", "oracle", "remaining_blocks"]
        for row, source in zip(table.to_pylist(), inputs.select_columns(metadata).to_list()):
            for key in metadata:
                if row[key] != source[key]:
                    raise ValueError(f"feature row differs from token cache: {path}: {key}")
            if row["hidden_state"] is None or (row["privileged_hidden_state"] is None) != (source["privileged_input_hash"] is None):
                raise ValueError(f"feature view availability differs: {path}")
            for key in ("hidden_state", "privileged_hidden_state"):
                if row[key] is not None and not np.isfinite(row[key]).all():
                    raise ValueError(f"nonfinite cached feature: {path}")

    def complete_cache(self) -> None:
        """Record completion and file hashes once every token chunk has an output.

        Returns:
            None. Records checksums for one hidden-state file per token file; raises if a file is missing or a completed cache changed.
        """
        path = self.directory / MANIFEST_FILENAME
        manifest = read_manifest(self.directory)
        checksums = {}
        for name in manifest["tokenization"]["shards"]:
            shard = self.chunk_path(name)
            if not shard.is_file():
                raise ValueError(f"missing hidden-state shard: {shard}")
            checksums[str(shard.relative_to(self.directory))] = file_digest(shard)
        previous = manifest["features"]
        if previous["complete"] and checksums != previous["shards"]:
            raise ValueError("completed hidden-state cache changed")
        manifest["features"].update(complete=True, shards=checksums)
        write_json_atomic(path, manifest)

    def load_splits(self, splits: tuple[str, ...], section: dict[str, Any]) -> DatasetDict:
        """Load the requested hidden-state splits into CPU RAM for frozen training.

        Args:
            splits: Requested train, validation, or test partitions.
            section: Completed features section from this collection's manifest.

        Returns:
            DatasetDict: Verified vector datasets with their original labels and metadata.
        """
        return load_parquet_splits(self.directory, splits, section, True)
