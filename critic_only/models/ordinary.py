"""Direct continuation probabilities with one head shared across history types."""

import torch
from torch import Tensor, nn

from critic_only.constants import VALUE_OUTPUT_SIZE
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class Ordinary(BaseCritic):
    """Predict success from the supplied history; OracleSupervised shares this same head."""

    def __init__(self, model: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Use identical head capacity for the ordinary and oracle-supervised baselines."""
        super().__init__(model, hidden_size, settings)
        self.head = nn.Linear(hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Return one terminal logit; remaining budget is already included in the encoded prompt."""
        return self.head(hidden).float().squeeze(-1)
