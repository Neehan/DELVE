"""Put each trajectory's inputs with and without a reference solution into one model batch."""

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


class CriticBatcher:
    """Batch tokens for LoRA or cached vectors for frozen training.

    The first B inputs omit reference solutions. The next P include available references. privileged_indices maps those P extra inputs back to their original trajectories; for example, [0, 2] means only trajectories 0 and 2 have references. Both versions use the same final correctness label.
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
        """Put both reference versions into one batch while keeping one label per trajectory.

        Args:
            rows: Examples with tokens or vectors, optional reference inputs, labels, oracle flags, and remaining budgets.

        Returns:
            dict[str, Any]: Model inputs for B+P views, labels [B], privileged_indices [P], and oracle/budget tensors [B+P].
        """
        column = "input_ids" if self.tokenizer is not None else "hidden_state"
        reference_column = f"privileged_{column}"
        indices = [index for index, row in enumerate(rows) if row[reference_column] is not None]
        views = [row[column] for row in rows] + [rows[index][reference_column] for index in indices]
        metadata = rows + [rows[index] for index in indices]
        if self.tokenizer is None:
            inputs = {"hidden_state": torch.stack([torch.as_tensor(view, dtype=torch.float32) for view in views])}
        else:
            padded = self.tokenizer.pad([{"input_ids": view} for view in views], padding=True, return_tensors="pt")
            inputs = {name: value for name, value in padded.items() if name in {"input_ids", "attention_mask"}}
        return {"labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32), "inputs": inputs,
                "privileged_indices": torch.tensor(indices, dtype=torch.long),
                "oracle": torch.tensor([row["oracle"] for row in metadata], dtype=torch.bool),
                "remaining_blocks": torch.tensor([row["remaining_blocks"] for row in metadata], dtype=torch.long)}
