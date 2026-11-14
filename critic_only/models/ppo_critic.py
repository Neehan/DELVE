"""Predict final answer correctness directly from the encoded trajectory."""

import torch
from torch import Tensor, nn

from critic_only.constants import VALUE_OUTPUT_SIZE
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class PPOCritic(BaseCritic):
    """Learn final success probability with one linear head after the shared MLP.

    The same head is applied independently to each final-layer token state in live training. Cached-feature training predicts only the endpoint. The training CLI selects unaided trajectories; PPOOracleCritic also trains on oracle histories.
    """

    token_supervision = True

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
            hidden: MLP outputs [B,hidden_size] or [B,L,hidden_size], with no mixing across token positions.
            oracle: Solver-sketch flags [B]; unused here because the sketch is already in the encoded history. Kept for the common critic interface.
            remaining_blocks: Endpoint budgets [B]; unused here because the prompt already contains the budget. DE uses this argument explicitly.

        Returns:
            Tensor: Float32 logits [B] or [B,L]. Apply sigmoid for probabilities, not before computing BCE.
        """
        return self.prediction_head(hidden).float().squeeze(-1)
