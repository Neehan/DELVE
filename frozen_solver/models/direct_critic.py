"""Direct probability heads, optionally trained with the same oracle supervision as DE."""

import torch
from torch import Tensor, nn

from frozen_solver.constants import CONDITION_COUNT
from frozen_solver.models.base_critic import BaseCritic
from frozen_solver.models.models import CriticConfig


class DirectCritic(BaseCritic):
    """Predict each horizon independently; configuration enables the oracle auxiliary loss."""

    def __init__(self, backbone: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Use identical head capacity for the direct and auxiliary baselines."""
        super().__init__(backbone, hidden_size, settings)
        self.head = nn.Linear(hidden_size, CONDITION_COUNT * self.horizons, dtype=torch.float32)

    def predict(self, hidden: Tensor) -> dict[str, Tensor]:
        """Produce unrestricted horizon logits so later self-refinement may lower success."""
        return {"logits": self.head(hidden).reshape(-1, CONDITION_COUNT, self.horizons)}
