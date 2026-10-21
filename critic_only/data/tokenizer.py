"""Tokenize critic inputs once and retain binary labels separately from privileged references."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from datasets import Dataset, DatasetDict
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from critic_only.constants import CRITIC_PROMPT_PATH, MANIFEST_FILENAME, REFERENCE_PROMPT_PATH, SPLITS
from critic_only.data.models import InputPrefix
from critic_only.data.processor import selection_fraction


class PrefixTokenizer:
    """Cache variable-length tokens and binary terminal labels; never truncate a saved prefix."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Require right padding and a pad token for the critic's pooling convention."""
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("critic tokenizer requires right padding and an explicit pad token")
        self.tokenizer = tokenizer
        self.system_prompt = CRITIC_PROMPT_PATH.read_text().strip()
        self.reference_prompt = REFERENCE_PROMPT_PATH.read_text().strip()

    def pad_batch(self, rows: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Pad unprivileged then available privileged views; labels retain one row per prefix.

        privileged_indices maps appended views back to prefixes; input and label batch sizes can differ.
        """
        features = [{"input_ids": row["input_ids"]} for row in rows]
        indices = [index for index, row in enumerate(rows) if row["privileged_input_ids"] is not None]
        for index in indices:
            tokens = rows[index]["privileged_input_ids"]
            features.append({"input_ids": tokens})
        padded = dict(self.tokenizer.pad(features, padding=True, return_tensors="pt"))
        remaining = [row["remaining_blocks"] for row in rows]
        # Trainer infers batch size from the first tensor: count prefixes, not duplicated views.
        return {"labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32),
                **padded, "privileged_indices": torch.tensor(indices, dtype=torch.long),
                "remaining_blocks": torch.tensor(remaining + [remaining[index] for index in indices], dtype=torch.long),
                "oracle": torch.tensor([row["oracle"] for row in rows] + [rows[index]["oracle"] for index in indices],
                                       dtype=torch.bool)}

    def encode(self, row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        """Apply the model's chat template to critic instructions and the history being evaluated.

        The history is quoted in a user message so templates cannot strip assistant reasoning. The optional reference and its calibration instructions extend only the privileged system message. Each branch supplies one scalar final binary outcome; earlier grades are neither inputs nor training targets.
        """
        record = InputPrefix.model_validate(dict(row))
        conversation = "\n\n".join(f"{item['role']}:\n{item['content']}" for item in record.messages)
        system = self.system_prompt.format(block_tokens=OUTPUT_TOKENS_PER_BLOCK, remaining_blocks=record.remaining_blocks,
                                           phase=record.phase, response_finished=record.response_finished)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": conversation}]
        token_ids = self._encode_messages(messages, settings["max_length"], record.prefix_id)
        privileged_ids = None
        if record.reference_solution is not None:
            if not record.reference_solution.strip():
                raise ValueError(f"empty reference solution: {record.prefix_id}")
            privileged_system = system + "\n\n" + self.reference_prompt.format(solution=record.reference_solution)
            privileged_messages = [{"role": "system", "content": privileged_system}, messages[1]]
            privileged_ids = self._encode_messages(privileged_messages, settings["max_length"], record.prefix_id)
        return {"input_ids": token_ids, "privileged_input_ids": privileged_ids,
                "labels": record.branch.final_correctness, "remaining_blocks": record.remaining_blocks,
                "length": len(privileged_ids) if privileged_ids is not None else len(token_ids),
                "prefix_id": record.prefix_id, "problem_key": record.problem_key,
                "oracle": record.oracle}

    def _retain_example(self, row: dict[str, Any], settings: dict[str, Any]) -> bool:
        """Limit oracle training problems while retaining all unaided and evaluation histories."""
        return (row["split"] != "train" or not row["oracle"]
                or selection_fraction(f"oracle:{row['problem_key']}", settings["seed"]) < settings["oracle_supervision_fraction"])

    def _encode_messages(self, messages: list[dict[str, str]], max_length: int, prefix_id: str) -> list[int]:
        """Tokenize without a generation prefix or truncation; the critic uses a numerical head."""
        tokens = cast(list[int], self.tokenizer.apply_chat_template(messages, tokenize=True,
                      add_generation_prompt=False, truncation=False, return_dict=False))
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
