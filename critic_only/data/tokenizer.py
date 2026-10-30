"""Build critic prompts and save their token IDs, with and without a reference solution.

The critic system prompt precedes the saved solver history. The reference version adds the matching solution to the system prompt. Both versions share one final correctness label, kept outside the prompt.
"""

import hashlib
from typing import Any, cast

import numpy as np
from transformers import PreTrainedTokenizerBase

from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from critic_only.constants import TOKEN_HASH_DTYPE
from critic_only.data.models import InputPrefix


class PrefixTokenizer:
    """Turn an InputPrefix into the two token sequences used to train the critic.

    encode applies Qwen's chat template and returns tokens, labels, and IDs. _encode_messages rejects inputs exceeding the length limit rather than truncating the trajectory.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Check the tokenizer's padding settings before formatting critic inputs.

        Args:
            tokenizer: Model tokenizer with right padding and a pad token; its chat template formats both input versions.

        Returns:
            None. Stores the tokenizer; encode uses the prompt text recorded in setup settings.
        """
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("critic tokenizer requires right padding and an explicit pad token")
        self.tokenizer = tokenizer

    def encode(self, row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        """Create token inputs with and without the matching reference solution.

        The history is quoted in a user message so templates cannot strip assistant reasoning. The optional reference and its calibration instructions extend only the privileged system message. Each branch supplies one scalar final binary outcome; earlier grades are neither inputs nor training targets.

        Args:
            row: Serialized InputPrefix containing observed messages, matching reference, metadata, and a separate final label.
            settings: Saved critic_prompt and reference_prompt templates, plus max_length for either input version.

        Returns:
            dict[str, Any]: Ordinary/privileged token IDs and hashes, scalar label, example/problem IDs, split, oracle flag, length, and remaining blocks.
        """
        record = InputPrefix.model_validate(dict(row))
        conversation = "\n\n".join(f"{item['role']}:\n{item['content']}" for item in record.messages)
        system = settings["critic_prompt"].format(block_tokens=OUTPUT_TOKENS_PER_BLOCK, remaining_blocks=record.remaining_blocks,
                                           phase=record.phase, response_finished=record.response_finished)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": conversation}]
        token_ids = self._encode_messages(messages, settings["max_length"], record.prefix_id)
        privileged_ids = None
        if record.reference_solution is not None:
            if not record.reference_solution.strip():
                raise ValueError(f"empty reference solution: {record.prefix_id}")
            privileged_system = system + "\n\n" + settings["reference_prompt"].format(solution=record.reference_solution)
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
        """Apply the chat template and reject inputs longer than the allowed limit.

        Args:
            messages: Formatted critic chat messages, including the system prompt and quoted solver history.
            max_length: Maximum permitted token count; inputs exceeding it are rejected rather than truncated.
            prefix_id: Example identifier included in length errors.

        Returns:
            list[int]: Exact chat-template token IDs, with no added generation prompt.
        """
        tokens = cast(list[int], self.tokenizer.apply_chat_template(messages, tokenize=True,
                      add_generation_prompt=False, truncation=False, return_dict=False))
        if len(tokens) > max_length:
            raise ValueError(f"prefix exceeds max-length; truncation is forbidden: {prefix_id}")
        return tokens


def token_digest(token_ids: list[int]) -> str:
    """Hash token IDs so hidden-state generation can check it is encoding the saved input.

    Args:
        token_ids: Ordered token IDs for one exact formatted critic input.

    Returns:
        str: SHA-256 hexadecimal digest of the IDs encoded as fixed-width little-endian integers.
    """
    return hashlib.sha256(np.asarray(token_ids, dtype=TOKEN_HASH_DTYPE).tobytes()).hexdigest()
