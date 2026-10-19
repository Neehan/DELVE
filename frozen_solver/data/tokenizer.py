"""Tokenize critic inputs once and retain branch counts separately from privileged references."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from frozen_solver.constants import CONDITIONS, MANIFEST_FILENAME, PROMPT, REFERENCE_PROMPT, SPLITS
from frozen_solver.data.models import PrefixRecord
from frozen_solver.data.processor import selection_fraction


class PrefixTokenizer:
    """Cache variable-length tokens and masked binomial counts; never truncate a saved prefix."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Require right padding and a pad token for the critic's pooling convention."""
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("critic tokenizer requires right padding and an explicit pad token")
        self.tokenizer = tokenizer

    def pad_batch(self, rows: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Pad inputs only, leaving horizon and condition dimensions fixed across the batch."""
        features = [{key: row[key] for key in ("input_ids", "attention_mask")} for row in rows]
        batch = dict(self.tokenizer.pad(features, padding=True, return_tensors="pt"))
        batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.float32)
        return batch

    def encode(self, row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        """Build one ordinary input; oracle outcomes supervise a separate output, never enter the input."""
        record = PrefixRecord.model_validate(dict(row))
        conversation = "\n\n".join(f"{item['role']}:\n{item['content']}" for item in record.messages)
        text = PROMPT.format(block_tokens=OUTPUT_TOKENS_PER_BLOCK, fork=record.fork,
                             budget=record.horizon, conversation=conversation, phase=record.phase,
                             response_finished=record.response_finished)
        if settings["reference_mode"] == "solution":
            text += REFERENCE_PROMPT.format(solution=record.reference_solution)
        token_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(token_ids) > settings["max_length"]:
            raise ValueError(f"prefix exceeds max-length; truncation is forbidden: {record.prefix_id}")
        counts = []
        oracle_available = selection_fraction(f"oracle:{record.problem_key}", settings["seed"]) < settings["oracle_fraction"]
        for condition in CONDITIONS:
            branches = [item for item in record.branches if item.condition == condition]
            if record.split == "train" and settings["branch_limit"] is not None:
                branches.sort(key=lambda item: selection_fraction(f"branch:{record.prefix_id}:{item.seed}", settings["seed"]))
                branches = branches[:settings["branch_limit"]]
            if condition == "continue-oracle" and record.split == "train" and not oracle_available:
                branches = []
            counts.append([[sum(item.outcomes[h] for item in branches), len(branches)]
                           if h < record.horizon - record.fork else [0, 0]
                           for h in range(record.horizon - 1)])
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids), "labels": counts,
                "length": len(token_ids), "prefix_id": record.prefix_id, "problem_key": record.problem_key,
                "fork": record.fork, "horizon": record.horizon}

    def run(self, source: Path, destination: Path, settings: dict[str, Any]) -> None:
        """Publish a memory-mappable Arrow cache with its exact input and tokenizer provenance."""
        if destination.exists():
            raise FileExistsError(destination)
        datasets = DatasetDict()
        checksums = {}
        for split in SPLITS:
            path = source / f"{split}.parquet"
            if path.exists():
                checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                dataset = Dataset.from_parquet(str(path))
                datasets[split] = dataset.map(self.encode, fn_kwargs={"settings": settings},
                                             remove_columns=dataset.column_names, load_from_cache_file=False)
        if not datasets:
            raise ValueError("no prepared Parquet splits")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "tokenized"
            datasets.save_to_disk(str(staging))
            self.tokenizer.save_pretrained(staging / "tokenizer")
            manifest = {"settings": settings, "input_sha256": checksums,
                        "preparation": json.loads((source / MANIFEST_FILENAME).read_text())}
            (staging / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
            staging.rename(destination)
