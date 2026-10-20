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
from critic_only.constants import MANIFEST_FILENAME, PROMPT, REFERENCE_PROMPT, SPLITS
from critic_only.data.models import PrefixRecord
from critic_only.data.processor import selection_fraction


class PrefixTokenizer:
    """Cache variable-length tokens and masked binomial counts; never truncate a saved prefix."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Require right padding and a pad token for the critic's pooling convention."""
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("critic tokenizer requires right padding and an explicit pad token")
        self.tokenizer = tokenizer

    def pad_batch(self, rows: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Pad ordinary then available privileged views; labels retain one row per prefix.

        privileged_indices maps appended views back to prefixes; input and label batch sizes can differ.
        """
        features = [{"input_ids": row["input_ids"]} for row in rows]
        indices = [index for index, row in enumerate(rows) if row["privileged_input_ids"] is not None]
        for index in indices:
            tokens = rows[index]["privileged_input_ids"]
            features.append({"input_ids": tokens})
        padded = dict(self.tokenizer.pad(features, padding=True, return_tensors="pt"))
        # Trainer infers batch size from the first tensor: count prefixes, not duplicated views.
        return {"labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32),
                **padded, "privileged_indices": torch.tensor(indices, dtype=torch.long),
                "oracle": torch.tensor([row["oracle"] for row in rows] + [rows[index]["oracle"] for index in indices],
                                       dtype=torch.bool)}

    def encode(self, row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        """Cache plain role-labelled text and an optional reference-augmented view, with shared counts.

        Each row contains one actual history. Labels have axes [future horizon, success/trial]; [0, 0] masks horizons beyond the remaining budget.
        """
        record = PrefixRecord.model_validate(dict(row))
        conversation = "\n\n".join(f"{item['role']}:\n{item['content']}" for item in record.messages)
        text = PROMPT.format(block_tokens=OUTPUT_TOKENS_PER_BLOCK, remaining_blocks=record.remaining_blocks,
                             conversation=conversation, phase=record.phase,
                             response_finished=record.response_finished)
        token_ids = self._encode_text(text, settings["max_length"], record.prefix_id)
        privileged_ids = None
        if record.reference_solution is not None:
            if not record.reference_solution.strip():
                raise ValueError(f"empty reference solution: {record.prefix_id}")
            privileged_ids = self._encode_text(text + REFERENCE_PROMPT.format(solution=record.reference_solution),
                                               settings["max_length"], record.prefix_id)
        branches = record.branches
        if record.split == "train" and settings["branch_limit"] is not None:
            branches = sorted(branches, key=lambda item: selection_fraction(f"branch:{record.prefix_id}:{item.seed}", settings["seed"]))
            branches = branches[:settings["branch_limit"]]
        counts = [[sum(item.outcomes[h] for item in branches), len(branches)]
                  if h < record.remaining_blocks else [0, 0]
                  for h in range(record.fork + record.remaining_blocks - 1)]
        return {"input_ids": token_ids, "privileged_input_ids": privileged_ids,
                "labels": counts,
                "length": len(privileged_ids) if privileged_ids is not None else len(token_ids),
                "prefix_id": record.prefix_id, "problem_key": record.problem_key,
                "oracle": record.oracle}

    def _retain_example(self, row: dict[str, Any], settings: dict[str, Any]) -> bool:
        """Limit oracle training problems without retaining empty-label examples or changing evaluation."""
        return (row["split"] != "train" or not row["oracle"]
                or selection_fraction(f"oracle:{row['problem_key']}", settings["seed"]) < settings["oracle_supervision_fraction"])

    def _encode_text(self, text: str, max_length: int, prefix_id: str) -> list[int]:
        """Reject either overlong view instead of silently dropping reference context or prefix tokens."""
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(tokens) > max_length:
            raise ValueError(f"prefix exceeds max-length; truncation is forbidden: {prefix_id}")
        return tokens

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
                dataset = dataset.filter(self._retain_example, fn_kwargs={"settings": settings}, load_from_cache_file=False)
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
