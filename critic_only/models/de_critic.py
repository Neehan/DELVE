"""Geometric discovery and execution curves for terminal success prediction."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import VALUE_OUTPUT_SIZE
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class DECritic(BaseCritic):
    """Learn whether a strategy is discovered, future discovery, and execution; latent parameters are not strategy labels."""

    def __init__(self, model: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Keep the execution curve unrestricted to allow correct-to-incorrect revisions."""
        super().__init__(model, hidden_size, settings)
        self.horizons = settings.horizon - 1
        self.strategy_discovered = nn.Linear(hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)
        self.discovery = nn.Linear(hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)
        self.execution = nn.Linear(hidden_size, self.horizons, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Return the DE value at each history's remaining budget, or its execution value for oracle histories.

        All heads read the supplied history. Oracle examples include the actual sketch and one block of solver execution; they supervise the remaining execution value; unaided examples use the shared execution head inside the discovery convolution. Transferring that structure across histories is an inductive bias, not an identified causal decomposition.

        strategy_discovered is the probability that the observed history already contains a viable strategy. Alpha describes discovery per block conditional on still needing one. For remaining budget r and strategy-discovered probability s, success is s*epsilon_r + (1-s)*sum_{j=1}^r alpha*(1-alpha)^(j-1)*epsilon_{r-j+1}. Failure includes the possibility of never discovering a strategy. Return log(success) - log(failure) without rounding either probability to zero or one.
        """
        # Compute log probabilities in FP32 even when the model and heads run under BF16 autocast.
        strategy_discovered_logits = self.strategy_discovered(hidden).float().squeeze(-1)
        log_strategy_discovered = F.logsigmoid(strategy_discovered_logits)
        log_strategy_not_discovered = F.logsigmoid(-strategy_discovered_logits)
        alpha_logits = self.discovery(hidden).float().squeeze(-1)
        execution_logits = self.execution(hidden).float()
        log_alpha = F.logsigmoid(alpha_logits)
        log_no_discovery = F.logsigmoid(-alpha_logits)
        steps = torch.arange(self.horizons, device=hidden.device)
        delays = remaining_blocks[:, None] - 1 - steps

        # Discovery after a delay has weight alpha * (1 - alpha)^delay; exclude unreachable execution horizons.
        log_weights = log_alpha[:, None] + delays.clamp_min(0) * log_no_discovery[:, None]
        log_weights = log_weights.masked_fill(delays < 0, -torch.inf)
        log_search_success = torch.logsumexp(log_weights + F.logsigmoid(execution_logits), dim=-1)
        log_acquired_failure = torch.logsumexp(log_weights + F.logsigmoid(-execution_logits), dim=-1)

        # Failure also includes no discovery before the budget ends: (1 - alpha)^remaining_blocks.
        log_search_failure = torch.logaddexp(remaining_blocks * log_no_discovery, log_acquired_failure)
        terminal_execution = execution_logits.gather(-1, (remaining_blocks - 1).unsqueeze(-1)).squeeze(-1)

        # A discovered strategy goes directly to execution; otherwise mix over future discovery times.
        log_success = torch.logaddexp(log_strategy_discovered + F.logsigmoid(terminal_execution),
                                     log_strategy_not_discovered + log_search_success)
        log_failure = torch.logaddexp(log_strategy_discovered + F.logsigmoid(-terminal_execution),
                                     log_strategy_not_discovered + log_search_failure)
        # Oracle histories use only epsilon's logit; unaided histories use the combined discovery-execution logit.
        return torch.where(oracle, terminal_execution, log_success - log_failure)
