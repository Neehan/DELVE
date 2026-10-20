"""Ordinary continuation probabilities with shared heads for the oracle-supervised variant."""

import torch
from torch import Tensor, nn

from critic_only.constants import CONDITION_COUNT
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class Ordinary(BaseCritic):
    """Predict each horizon directly; OracleSupervised reuses the heads with additional supervision."""

    def __init__(self, backbone: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Use identical head capacity for the ordinary and oracle-supervised baselines."""
        super().__init__(backbone, hidden_size, settings)
        self.head = nn.Linear(hidden_size, CONDITION_COUNT * self.horizons, dtype=torch.float32)

    def predict(self, hidden: Tensor) -> dict[str, Tensor]:
        """Produce unrestricted horizon logits so later self-refinement may lower success."""
        return {"logits": self.head(hidden).reshape(-1, CONDITION_COUNT, self.horizons)}
