"""Build critic prompts and save their token IDs, with and without a reference solution.

The critic system prompt precedes the saved solver history. The reference version adds the matching solution to the system prompt. Both versions share the trajectory’s future checkpoint grades, kept outside the prompt.
"""

import hashlib
from typing import Any, cast

import numpy as np
from transformers import PreTrainedTokenizerBase

from critic_only.constants import TOKEN_HASH_DTYPE
from critic_only.data.models import InputPrefix
from critic_only.data.solver_tokens import conversation_spans, solver_token_mask


class PrefixTokenizer:
    """Render critic references and histories, then mark each solver token for Monte Carlo supervision.

    encode saves both input views and their solver masks. Inputs end at the final solver token used for endpoint metrics. No terminal labels enter either view.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Store a right-padding fast tokenizer; character offsets are required for exact supervision masks."""
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None or not tokenizer.is_fast:
            raise ValueError("critic tokenization requires a fast right-padding tokenizer with a pad token")
        self.tokenizer = tokenizer

    def encode(self, row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        """Tokenize the observed history and keep its future grades outside both input views.

        Args:
            row: Serialized InputPrefix containing the reconstructed 1× history, matching reference and future checkpoint grades.
            settings: Pinned prompts, total horizon and maximum input length. Only the total budget precedes the history; endpoint metadata would leak backward to earlier tokens.

        Returns:
            dict[str, Any]: Token IDs and solver masks with/without the reference, binary checkpoint labels, and identities for cache validation. Training selects the reference-containing view.
        """
        record = InputPrefix.model_validate(dict(row))
        if len(record.branch.checkpoint_correctness) != settings["horizon"] - 1 or record.remaining_blocks != settings["horizon"] - 1:
            raise ValueError(f"checkpoint labels do not match the 1× input budget: {record.prefix_id}")
        conversation, spans = conversation_spans(record.messages)
        system = settings["critic_prompt"].format(block_tokens=settings["checkpoint_interval_tokens"], horizon=settings["horizon"])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": conversation}]
        token_ids, mask = self._encode_messages(messages, spans, settings["max_length"], record.prefix_id)
        privileged_ids, privileged_mask = None, None
        if record.reference_solution is not None:
            if not record.reference_solution.strip():
                raise ValueError(f"empty reference solution: {record.prefix_id}")
            privileged_system = system + "\n\n" + settings["reference_prompt"].format(solution=record.reference_solution)
            privileged_messages = [{"role": "system", "content": privileged_system}, messages[1]]
            privileged_ids, privileged_mask = self._encode_messages(privileged_messages, spans, settings["max_length"], record.prefix_id)
        return {"input_ids": token_ids, "privileged_input_ids": privileged_ids,
                "solver_mask": mask, "privileged_solver_mask": privileged_mask,
                "input_hash": token_digest(token_ids),
                "privileged_input_hash": token_digest(privileged_ids) if privileged_ids is not None else None,
                "split": record.split, "labels": record.branch.checkpoint_correctness, "branch": record.branch.model_dump(),
                "remaining_blocks": record.remaining_blocks,
                "length": len(privileged_ids) if privileged_ids is not None else len(token_ids),
                "prefix_id": record.prefix_id, "problem_key": record.problem_key, "oracle": record.oracle}

    def _encode_messages(self, messages: list[dict[str, str]], spans: list[tuple[int, int]],
                         max_length: int, prefix_id: str) -> tuple[list[int], list[bool]]:
        """Return exact template token IDs and solver mask through the last solver token.

        messages contains critic system text and the quoted history; spans index solver content within that
        history. max_length is checked without truncating solver text; prefix_id identifies length errors.
        """
        rendered = cast(str, self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        tokens = cast(list[int], encoded["input_ids"])
        mask = solver_token_mask(rendered, messages[1]["content"], spans, encoded["offset_mapping"])
        endpoint = max(index for index, selected in enumerate(mask) if selected) + 1
        if endpoint > max_length:
            raise ValueError(f"prefix exceeds max-length; truncation is forbidden: {prefix_id}")
        # Chat-template closing markers follow the solver state and are not part of its value prediction.
        return tokens[:endpoint], mask[:endpoint]


def token_digest(token_ids: list[int]) -> str:
    """Hash token IDs so storage can verify the saved input sequence.

    Args:
        token_ids: Ordered token IDs for one exact formatted critic input.

    Returns:
        str: SHA-256 hexadecimal digest of the IDs encoded as fixed-width little-endian integers.
    """
    return hashlib.sha256(np.asarray(token_ids, dtype=TOKEN_HASH_DTYPE).tobytes()).hexdigest()
