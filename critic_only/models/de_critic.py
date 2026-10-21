"""Discovery-execution success curves with shared acquisition and execution heads."""

import torch
from torch import Tensor, nn

from critic_only.constants import DE_STATE_FIELDS, PROBABILITY_EPSILON
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class DECritic(BaseCritic):
    """Learn acquisition and execution heads; latent parameters are not strategy labels."""

    def __init__(self, model: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Keep the execution curve unrestricted to allow correct-to-incorrect revisions."""
        super().__init__(model, hidden_size, settings)
        self.horizons = settings.horizon - 1
        self.acquisition = nn.Linear(hidden_size, DE_STATE_FIELDS, dtype=torch.float32)
        self.execution = nn.Linear(hidden_size, self.horizons, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Return the DE value at each history's remaining budget, or its execution value for oracle histories.

        All heads read the supplied history. Oracle examples supervise execution after the actual sketch; unaided examples use the shared execution head inside the discovery convolution. Transferring that structure across histories is an inductive bias, not an identified causal decomposition.
        """
        # Keep probabilities in FP32: BF16 rounds the upper clamp bound to exactly one.
        discovered, alpha = self.acquisition(hidden).float().sigmoid().unbind(dim=-1)
        execution_logits = self.execution(hidden).float()
        epsilon = execution_logits.sigmoid()
        steps = torch.arange(self.horizons, device=hidden.device)
        delays = remaining_blocks[:, None] - 1 - steps
        weights = alpha[:, None] * (1 - alpha[:, None]).pow(delays.clamp_min(0))
        # Sum only execution horizons reachable within this history's remaining budget.
        acquired = (weights * epsilon * (delays >= 0)).sum(dim=-1)
        terminal_execution = execution_logits.gather(-1, (remaining_blocks - 1).unsqueeze(-1)).squeeze(-1)
        prior = discovered * terminal_execution.sigmoid() + (1 - discovered) * acquired
        prior = prior.clamp(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        # Oracle histories use only epsilon's logit; unaided histories use the combined discovery-execution logit.
        return torch.where(oracle, terminal_execution, torch.logit(prior))
