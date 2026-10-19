"""Pad cached token IDs at batch time while keeping count labels in floating point."""

from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


class PrefixCollator:
    """Right-pad variable-length prefixes; auxiliary metadata never reaches the transformer."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Require the pooling convention used by every critic variant."""
        if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
            raise ValueError("critic tokenizer requires right padding and an explicit pad token")
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Pad inputs only, leaving horizon and condition dimensions fixed across the batch."""
        features = [{key: row[key] for key in ("input_ids", "attention_mask")} for row in rows]
        batch = dict(self.tokenizer.pad(features, padding=True, return_tensors="pt"))
        batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.float32)
        return batch
