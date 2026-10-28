"""Tokenize critic inputs once and retain binary labels separately from privileged references."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from critic_only.constants import CRITIC_PROMPT_PATH, REFERENCE_PROMPT_PATH, TOKEN_HASH_DTYPE, TOKEN_SHARD_SIZE
from discovery_execution.processing.constants import SPLITS
from critic_only.data.models import InputPrefix
from critic_only.data.processor import CriticInputProcessor


class PrefixTokenizer:
    """Cache variable-length tokens and binary terminal labels; never truncate a saved prefix."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Require right padding and a pad token for the critic's pooling convention."""
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("critic tokenizer requires right padding and an explicit pad token")
        self.tokenizer = tokenizer
        self.system_prompt = CRITIC_PROMPT_PATH.read_text().strip()
        self.reference_prompt = REFERENCE_PROMPT_PATH.read_text().strip()

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
                "input_hash": token_digest(token_ids),
                "privileged_input_hash": token_digest(privileged_ids) if privileged_ids is not None else None,
                "split": record.split,
                "labels": record.branch.final_correctness, "remaining_blocks": record.remaining_blocks,
                "length": len(privileged_ids) if privileged_ids is not None else len(token_ids),
                "prefix_id": record.prefix_id, "problem_key": record.problem_key,
                "oracle": record.oracle}

    def _encode_messages(self, messages: list[dict[str, str]], max_length: int, prefix_id: str) -> list[int]:
        """Tokenize without a generation prefix or truncation; the critic uses a numerical head."""
        tokens = cast(list[int], self.tokenizer.apply_chat_template(messages, tokenize=True,
                      add_generation_prompt=False, truncation=False, return_dict=False))
        if len(tokens) > max_length:
            raise ValueError(f"prefix exceeds max-length; truncation is forbidden: {prefix_id}")
        return tokens

    def cache_tokens(self, processor: CriticInputProcessor, destination: Path, settings: dict[str, Any]) -> None:
        """Compile exact inputs once and publish sharded Arrow with both reference views."""
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temporary:
            examples = Dataset.from_generator(processor.iter_inputs, cache_dir=temporary)
            if not len(examples):
                raise ValueError("no eligible nonterminal critic inputs")
            encoded = examples.map(self.encode, fn_kwargs={"settings": settings},
                                   remove_columns=examples.column_names, load_from_cache_file=False)
            datasets = DatasetDict({split: encoded.filter(lambda row: row["split"] == split,
                                    load_from_cache_file=False).remove_columns("split") for split in SPLITS})
            staging = Path(temporary) / "tokenized"
            datasets.save_to_disk(str(staging), max_shard_size=TOKEN_SHARD_SIZE)
            self.tokenizer.save_pretrained(staging / "tokenizer")
            staging.rename(destination)


def token_digest(token_ids: list[int]) -> str:
    """Fingerprint exact ordered token IDs using a fixed-width, fixed-endian encoding."""
    return hashlib.sha256(np.asarray(token_ids, dtype=TOKEN_HASH_DTYPE).tobytes()).hexdigest()
