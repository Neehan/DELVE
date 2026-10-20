"""Direct continuation probabilities with one head shared across history types."""

import torch
from torch import Tensor, nn

from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class Ordinary(BaseCritic):
    """Predict success from the supplied history; OracleSupervised shares this same head."""

    def __init__(self, backbone: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Use identical head capacity for the ordinary and oracle-supervised baselines."""
        super().__init__(backbone, hidden_size, settings)
        self.head = nn.Linear(hidden_size, self.horizons, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor) -> Tensor:
        """Use one horizon head for either history; oracle identifies examples for loss weighting only."""
        return self.head(hidden).float()
