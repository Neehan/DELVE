"""Batch the reference-containing view of each prefix for critic training and evaluation."""

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


class CriticBatcher:
    """Batch tokens for LoRA or cached vectors for frozen training.

    Each row supplies one reference-containing input and one success-fraction label. __call__ pads token sequences or stacks cached vectors and rejects missing references.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase | None) -> None:
        """Use a tokenizer to pad LoRA inputs, or None to stack cached vectors.

        Args:
            tokenizer: Right-padding tokenizer for LoRA, or None for cached vectors.

        Returns:
            None. Stores the tokenizer used by __call__.
        """
        if tokenizer is not None and (tokenizer.padding_side != "right" or tokenizer.pad_token_id is None):
            raise ValueError("critic batches require right padding and a pad token")
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Select one reference-containing input per row and assemble the model tensors.

        Args:
            rows: Examples with reference tokens or vectors, labels, oracle flags, and remaining budgets.

        Returns:
            dict[str, Any]: Model inputs for B rows, labels [B], and oracle/budget tensors [B].
        """
        column = "input_ids" if self.tokenizer is not None else "hidden_state"
        reference_column = f"privileged_{column}"
        # Both views remain cached for future experiments; this pilot uses references only.
        views = [row[reference_column] for row in rows]
        if any(view is None for view in views):
            raise ValueError("reference-only critic requires a reference input for every row")
        if self.tokenizer is None:
            inputs = {"hidden_state": torch.stack([torch.as_tensor(view, dtype=torch.float32) for view in views])}
        else:
            padded = self.tokenizer.pad([{"input_ids": view} for view in views], padding=True, return_tensors="pt")
            inputs = {name: value for name, value in padded.items() if name in {"input_ids", "attention_mask"}}
        return {"labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32), "inputs": inputs,
                "oracle": torch.tensor([row["oracle"] for row in rows], dtype=torch.bool),
                "remaining_blocks": torch.tensor([row["remaining_blocks"] for row in rows], dtype=torch.long)}
