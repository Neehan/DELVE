"""Extract final-token representations from exact cached critic inputs, shard by shard."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import PreTrainedTokenizerBase

from critic_only.constants import DEFAULT_LOGGING_STEPS, FEATURE_SHARD_ROWS
from critic_only.data.tokenizer import token_digest
from discovery_execution.processing.storage import write_parquet_atomic


class HiddenStateExtractor:
    """Run an injected frozen transformer and publish resumable shards with aligned labels."""

    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, batch_size: int) -> None:
        """Use right padding, inference-only weights, and bounded forward batches."""
        if batch_size < 1 or tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("extraction requires a positive batch size and right-padding tokenizer")
        self.model = model.eval().requires_grad_(False)
        self.tokenizer, self.batch_size = tokenizer, batch_size
        self.device = next(model.parameters()).device

    def extract_shards(self, datasets: DatasetDict, destination: Path, signature: str, rank: int, world_size: int) -> None:
        """Assign whole shards to ranks; resume only shards with matching inputs and extraction signature."""
        for split, dataset in datasets.items():
            order = np.argsort(np.asarray(dataset["length"]), kind="stable")
            ordered = dataset.select(order, keep_in_memory=True)
            for index, start in enumerate(range(0, len(ordered), FEATURE_SHARD_ROWS)):
                if index % world_size != rank:
                    continue
                rows = ordered.select(range(start, min(start + FEATURE_SHARD_ROWS, len(ordered))))
                path = destination / str(split) / f"part-{index:05d}.parquet"
                if path.exists():
                    self.validate_shard(path, rows, signature)
                else:
                    self._write_feature_shard(path, rows, signature)
                print(f"rank={rank} completed {split}/{path.name} examples={len(rows)}", flush=True)

    def _write_feature_shard(self, path: Path, dataset: Dataset, signature: str) -> None:
        """Keep feature rows self-contained; preserve each view's input hash and its branch label."""
        records = []
        for start in range(0, len(dataset), self.batch_size):
            rows = cast(list[dict[str, Any]], dataset.select(range(start, min(start + self.batch_size, len(dataset)))).to_list())
            ordinary = self._encode_views(rows, "input_ids", "input_hash")
            available = [index for index, row in enumerate(rows) if row["privileged_input_ids"] is not None]
            privileged = self._encode_views([rows[index] for index in available], "privileged_input_ids", "privileged_input_hash")
            references = dict(zip(available, privileged))
            for index, row in enumerate(rows):
                metadata = {key: value for key, value in row.items() if key not in {"input_ids", "privileged_input_ids", "length"}}
                records.append({**metadata, "hidden_state": ordinary[index], "privileged_hidden_state": references.get(index)})
            if (start // self.batch_size + 1) % DEFAULT_LOGGING_STEPS == 0:
                print(f"{path.parent.name}/{path.name} extracted={len(records)}/{len(dataset)}", flush=True)
        dimension = len(records[0]["hidden_state"])
        table = pa.Table.from_pylist(records)
        vector = pa.list_(pa.float32(), dimension)
        for name in ("hidden_state", "privileged_hidden_state"):
            table = table.set_column(table.schema.get_field_index(name), name,
                                     pa.array([row[name] for row in records], type=vector))
        table = table.replace_schema_metadata({"signature": signature})
        write_parquet_atomic(path, table)

    @torch.inference_mode()
    def _encode_views(self, rows: list[dict[str, Any]], token_column: str, hash_column: str) -> list[list[float]]:
        """Read the final non-padding hidden state without generation or retained KV caches."""
        if not rows:
            return []
        for row in rows:
            if token_digest(row[token_column]) != row[hash_column]:
                raise ValueError(f"cached token IDs changed: {row['prefix_id']}")
        encoded = self.tokenizer.pad([{"input_ids": row[token_column]} for row in rows], padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
        output = self.model(**inputs, use_cache=False)
        positions = inputs["attention_mask"].sum(dim=-1) - 1
        hidden = output.last_hidden_state[torch.arange(len(rows), device=self.device), positions].float()
        if not torch.isfinite(hidden).all():
            raise ValueError("nonfinite extracted hidden state")
        return hidden.cpu().tolist()

    @staticmethod
    def validate_shard(path: Path, inputs: Dataset, signature: str) -> None:
        """Verify resumed row identity, labels, hashes, and finite vectors against the frozen input cache."""
        table = pq.read_table(path)
        if (table.schema.metadata or {}).get(b"signature") != signature.encode() or table.num_rows != len(inputs):
            raise ValueError(f"feature shard configuration differs: {path}")
        for row, source in zip(table.to_pylist(), inputs.to_list()):
            for key in ("prefix_id", "problem_key", "input_hash", "privileged_input_hash", "labels", "oracle", "remaining_blocks"):
                if row[key] != source[key]:
                    raise ValueError(f"feature row differs from token cache: {path}: {key}")
            if row["hidden_state"] is None or (row["privileged_hidden_state"] is None) != (source["privileged_input_ids"] is None):
                raise ValueError(f"feature view availability differs: {path}")
            for key in ("hidden_state", "privileged_hidden_state"):
                if row[key] is not None and not np.isfinite(row[key]).all():
                    raise ValueError(f"nonfinite cached feature: {path}")
