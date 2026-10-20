"""Discovery-execution success curves with shared acquisition and execution heads."""

import torch
from torch import Tensor, nn

from critic_only.constants import DE_STATE_FIELDS, PROBABILITY_EPSILON
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class DECritic(BaseCritic):
    """Learn acquisition and execution heads; latent parameters are not strategy labels."""

    def __init__(self, backbone: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Keep the execution curve unrestricted to allow correct-to-incorrect revisions."""
        super().__init__(backbone, hidden_size, settings)
        self.acquisition = nn.Linear(hidden_size, DE_STATE_FIELDS, dtype=torch.float32)
        self.execution = nn.Linear(hidden_size, self.horizons, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor) -> Tensor:
        """Use execution logits for sketch-injected histories and the DE formula otherwise.

        All heads read the supplied history. Oracle examples supervise execution after the actual sketch; ordinary examples use the shared execution head inside the discovery convolution. Transferring that structure across histories is an inductive bias, not an identified causal decomposition.
        """
        # Keep probabilities in FP32: BF16 rounds the upper clamp bound to exactly one.
        discovered, alpha = self.acquisition(hidden).float().sigmoid().unbind(dim=-1)
        execution_logits = self.execution(hidden).float()
        epsilon = execution_logits.sigmoid()
        steps = torch.arange(self.horizons, device=hidden.device)
        delays = (steps[:, None] - steps[None, :]).clamp_min(0)
        weights = alpha[:, None, None] * (1 - alpha[:, None, None]).pow(delays)
        # Lower-triangular weights convolve discovery delays with the available execution horizon.
        acquired_curve = (weights.tril() * epsilon[:, None, :]).sum(dim=-1)
        prior = discovered[:, None] * epsilon + (1 - discovered[:, None]) * acquired_curve
        prior = prior.clamp(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        return torch.where(oracle[:, None], execution_logits, torch.logit(prior))
