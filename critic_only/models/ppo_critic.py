"""Predict final answer correctness directly from the encoded trajectory."""

import torch
from torch import Tensor, nn

from critic_only.constants import VALUE_OUTPUT_SIZE
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class PPOCritic(BaseCritic):
    """Learn final success probability with one linear head after the shared MLP.

    predict implements the direct head; BaseCritic supplies forward, loss, and checkpoints. The training CLI selects unaided trajectories for this variant. PPOOracleCritic reuses the architecture with additional oracle trajectories.
    """

    def __init__(self, settings: CriticConfig) -> None:
        """Add a linear success head after the shared MLP.

        Args:
            settings: Critic settings; hidden_size is the MLP width entering this head.

        Returns:
            None.
        """
        super().__init__(settings)
        self.prediction_head = nn.Linear(settings.hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Predict final correctness directly from each encoded history.

        Args:
            hidden: MLP outputs [N, hidden_size]. N includes both reference versions.
            oracle: Solver-sketch flags [N]; unused here because the sketch is already in the encoded history. Kept for the common critic interface.
            remaining_blocks: Future budgets [N]; unused here because the prompt already contains the budget. DE uses this argument explicitly.

        Returns:
            Tensor: Float32 logits [N]. Apply sigmoid for probabilities, but not before computing the loss.
        """
        return self.prediction_head(hidden).float().squeeze(-1)
