"""Predict exact correctness at each future checkpoint from every encoded token."""

import torch
from torch import Tensor, nn

from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class PPOCritic(BaseCritic):
    """Learn an unrestricted future-checkpoint curve with one linear head after the shared MLP.

    The same head is applied independently to each final-layer token state in live training. The training CLI selects unaided trajectories; PPOOracleCritic also trains on oracle histories.
    """

    def __init__(self, settings: CriticConfig) -> None:
        """Add a linear success head after the shared MLP.

        Args:
            settings: Critic settings; hidden_size is the MLP width entering this head.

        Returns:
            None.
        """
        super().__init__(settings)
        self.prediction_head = nn.Linear(settings.hidden_size, self.num_checkpoints, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> dict[str, Tensor]:
        """Predict all future checkpoint grades directly from each encoded history.

        Args:
            hidden: MLP outputs [B,L,hidden_size], with no mixing across token positions.
            oracle: Solver-sketch flags [B]; unused here because the sketch is already in the encoded history. Kept for the common critic interface.
            remaining_blocks: Final budgets [B,L]; DE uses these to couple future predictions. Unused by this direct head.

        Returns:
            dict: Float32 checkpoint logits [B,L,H-1]. Apply sigmoid only when reporting probabilities.
        """
        return {"logits": self.prediction_head(hidden).float()}
