"""Discovery-execution success curves with shared acquisition and execution heads."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import DE_STATE_FIELDS
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

        All heads read the supplied history. Oracle examples include the actual sketch and one block of solver execution; they supervise the remaining execution value; unaided examples use the shared execution head inside the discovery convolution. Transferring that structure across histories is an inductive bias, not an identified causal decomposition.

        With discovery state probability d and remaining budget r, success is d*epsilon_r + (1-d)*sum_{j=1}^r alpha*(1-alpha)^(j-1)*epsilon_{r-j+1}. Failure replaces epsilon with 1-epsilon and adds (1-d)*(1-alpha)^r for no discovery. Return log(success) - log(failure) without rounding either probability to zero or one.
        """
        # Compute log probabilities in FP32 even when the model and heads run under BF16 autocast.
        discovered_logits, alpha_logits = self.acquisition(hidden).float().unbind(dim=-1)
        execution_logits = self.execution(hidden).float()
        log_discovered = F.logsigmoid(discovered_logits)
        log_undiscovered = F.logsigmoid(-discovered_logits)
        log_alpha = F.logsigmoid(alpha_logits)
        log_no_discovery = F.logsigmoid(-alpha_logits)
        steps = torch.arange(self.horizons, device=hidden.device)
        delays = remaining_blocks[:, None] - 1 - steps

        # Discovery after a delay has weight alpha * (1 - alpha)^delay; exclude unreachable execution horizons.
        log_weights = log_alpha[:, None] + delays.clamp_min(0) * log_no_discovery[:, None]
        log_weights = log_weights.masked_fill(delays < 0, -torch.inf)
        log_acquired_success = torch.logsumexp(log_weights + F.logsigmoid(execution_logits), dim=-1)
        log_acquired_failure = torch.logsumexp(log_weights + F.logsigmoid(-execution_logits), dim=-1)

        # Failure also includes no discovery before the budget ends: (1 - alpha)^remaining_blocks.
        log_search_failure = torch.logaddexp(remaining_blocks * log_no_discovery, log_acquired_failure)
        terminal_execution = execution_logits.gather(-1, (remaining_blocks - 1).unsqueeze(-1)).squeeze(-1)

        # Mix already-discovered and still-searching states separately for success and failure, avoiding 1 - p.
        log_success = torch.logaddexp(log_discovered + F.logsigmoid(terminal_execution),
                                     log_undiscovered + log_acquired_success)
        log_failure = torch.logaddexp(log_discovered + F.logsigmoid(-terminal_execution),
                                     log_undiscovered + log_search_failure)
        # Oracle histories use only epsilon's logit; unaided histories use the combined discovery-execution logit.
        return torch.where(oracle, terminal_execution, log_success - log_failure)
