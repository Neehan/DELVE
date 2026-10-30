"""Expose frozen Qwen's final-token vectors to Hugging Face Trainer.predict."""

import torch
from torch import Tensor, nn


class HiddenStateEncoder(nn.Module):
    """Return one hidden vector per input instead of generating text.

    forward runs frozen Qwen and selects each sequence's last non-padding token. Trainer handles device placement, inference mode, and gathering predictions; storage handles files.
    """

    def __init__(self, model: nn.Module) -> None:
        """Wrap Qwen for inference without adding trainable parameters.

        Args:
            model: Transformer returning last_hidden_state; pretrained weights stay frozen.

        Returns:
            None.
        """
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """Run Qwen and retain only the final real token's vector from each input.

        Args:
            input_ids: Right-padded token IDs [N, L], placed on the model device by Trainer.
            attention_mask: Mask [N, L], with 1 for real tokens and 0 for padding.

        Returns:
            dict[str, Tensor]: Float32 hidden_state [N, hidden_size]. Trainer collects only these vectors, not all token states.
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        positions = attention_mask.sum(dim=-1) - 1
        hidden = output.last_hidden_state[torch.arange(len(positions), device=positions.device), positions]
        return {"hidden_state": hidden.float()}
