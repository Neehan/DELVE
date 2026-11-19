"""Batch the reference-containing view of each prefix for critic training and evaluation."""

from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


class CriticBatcher:
    """Batch reference-containing conversations for all-token critic supervision.

    Each row supplies one reference-containing input and a binary label for each future checkpoint. __call__ pads token sequences and solver masks and rejects missing references.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Use a tokenizer to pad critic conversations.

        Args:
            tokenizer: Right-padding tokenizer for critic conversations.

        Returns:
            None. Stores the tokenizer used by __call__.
        """
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("critic batches require right padding and a pad token")
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Select one reference-containing input per row and assemble the model tensors.

        Args:
            rows: Examples with reference tokens, solver masks, labels, oracle flags, and remaining budgets.

        Returns:
            dict[str, Any]: Model inputs for B rows, labels [B,H-1], oracle flags [B], and final budgets [B,L].
        """
        # Both views remain cached for future experiments; this pilot uses references only.
        reference_inputs = [row["privileged_input_ids"] for row in rows]
        if any(value is None for value in reference_inputs):
            raise ValueError("reference-only critic requires a reference input for every row")
        labels = torch.tensor([row["labels"] for row in rows], dtype=torch.float32)
        remaining_blocks = torch.tensor([row["remaining_blocks"] for row in rows], dtype=torch.long)
        if labels.ndim != 2 or labels.shape[1] == 0 or not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("curve labels must be a nonempty matrix of exact binary checkpoint grades")
        if not torch.all(remaining_blocks == labels.shape[1]):
            raise ValueError("curve labels must cover every checkpoint after the 1× input")
        solver_masks = [row["privileged_solver_mask"] for row in rows]
        inputs = self._pad_solver_inputs(reference_inputs, solver_masks)
        # All interior tokens precede 1×: ceil(remaining budget) is one block above the endpoint budget.
        endpoint_positions = inputs["attention_mask"].sum(dim=-1) - 1
        token_positions = torch.arange(inputs["input_ids"].shape[1])
        remaining_blocks = remaining_blocks[:, None] + (token_positions[None, :] < endpoint_positions[:, None]).long()
        return {"labels": labels, "inputs": inputs,
                "oracle": torch.tensor([row["oracle"] for row in rows], dtype=torch.bool),
                "remaining_blocks": remaining_blocks}

    def _pad_solver_inputs(
        self, token_ids: list[list[int]], solver_masks: list[list[bool]]
    ) -> dict[str, Tensor]:
        """Right-pad conversations and their supervision masks to the same batch length.

        Args:
            token_ids: Reference-containing conversations, each ending at its last solver token.
            solver_masks: Aligned flags identifying tokens that receive training loss.

        Returns:
            dict[str, Tensor]: Padded input_ids, attention_mask, and solver_mask [B,L].
        """
        if any(len(mask) != len(tokens) or not any(mask) or not mask[-1]
               for mask, tokens in zip(solver_masks, token_ids)):
            raise ValueError("solver mask must align with tokens and end at a supervised solver token")
        padded = self.tokenizer.pad(
            [{"input_ids": tokens} for tokens in token_ids], padding=True, return_tensors="pt"
        )
        inputs = {name: value for name, value in padded.items() if name in {"input_ids", "attention_mask"}}
        padded_length = inputs["input_ids"].shape[1]
        inputs["solver_mask"] = torch.tensor(
            [mask + [False] * (padded_length - len(mask)) for mask in solver_masks], dtype=torch.bool
        )
        return inputs
