"""Train the residual DE critic with BCE plus a squared logit-correction penalty."""

from typing import cast

from torch import Tensor

from critic_only.models.base_critic import BaseCritic
from critic_only.trainer.base_trainer import BaseTrainer


class DEResidualTrainer(BaseTrainer):
    """Add the residual penalty while inheriting batching, masking, evaluation and checkpoints.

    Only _checkpoint_losses differs: each supervised token/checkpoint contributes BCE(z_DE + r, y) + lambda*r². Both terms use the same oracle weight and equal-per-trajectory averaging.
    """

    def _checkpoint_losses(self, critic: BaseCritic, outputs: dict[str, Tensor], labels: Tensor, oracle: Tensor) -> Tensor:
        """Penalize departures from DE before the base trainer masks and averages losses.

        Args:
            critic: Residual critic with its validated residual_penalty coefficient.
            outputs: Corrected logits and residual, each [B,L,C], from the same model pass.
            labels: Exact future-checkpoint grades [B,C].
            oracle: Solver-hint flags [B], used to weight both loss terms identically.

        Returns:
            Tensor: Weighted BCE plus squared residual [B,L,C]. Endpoint NLL/Brier remain unpenalized.
        """
        losses = super()._checkpoint_losses(critic, outputs, labels, oracle)
        penalty = cast(float, critic.settings.residual_penalty) * outputs["residual"].float().square()
        return losses + critic.weight_losses(penalty, oracle)
