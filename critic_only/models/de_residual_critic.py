"""Add learned logit corrections to the strict discovery–execution value curve."""

import torch
from torch import Tensor, nn

from critic_only.models.de_critic import DECritic
from critic_only.models.models import CriticConfig


class DEResidualCritic(DECritic):
    """Reuse DE's curve and add a correction for each token and target checkpoint.

    predict returns corrected logits and the residual separately. DEResidualTrainer penalizes the residual; all DE calculations, Qwen encoding, and checkpoint storage are inherited.
    """

    def __init__(self, settings: CriticConfig) -> None:
        """Build the inherited DE heads and a zero-initialized checkpoint correction head.

        Args:
            settings: Shared model configuration, including the residual penalty saved for the trainer.

        Returns:
            None. The initial prediction exactly matches strict DE with the same inherited weights.
        """
        super().__init__(settings)
        self.residual_head = nn.Linear(settings.hidden_size, self.num_checkpoints, dtype=torch.float32)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> dict[str, Tensor]:
        """Correct the inherited DE logits without another Qwen pass.

        Args:
            hidden: Shared MLP outputs [B,L,D]; each token's correction uses only its causal state.
            oracle: Solver-hint flags [B]; the inherited DE term uses execution alone for oracle rows.
            remaining_blocks: Final-checkpoint budgets [B,L], used by the inherited DE calculation.

        Returns:
            dict: Corrected logits and residual, both FP32 [B,L,C]. The trainer uses residual for its penalty.
        """
        outputs = super().predict(hidden, oracle, remaining_blocks)
        residual = self.residual_head(hidden).float()
        # Add in logit space: corrections can increase or decrease success probability without clipping.
        return {"logits": outputs["logits"] + residual, "residual": residual}
