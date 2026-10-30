"""Predict final answer correctness by combining strategy discovery and execution probabilities."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import VALUE_OUTPUT_SIZE
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class DECritic(BaseCritic):
    """Estimate success by separating strategy discovery from strategy execution.

    The heads predict s, the probability a strategy is already discovered; alpha, the probability of discovering one per future block; and epsilon_r, success given a strategy and r remaining blocks. Unaided success is s*epsilon_r + (1-s)*sum_{j=1}^r alpha*(1-alpha)^(j-1)*epsilon_(r-j+1). Oracle histories use epsilon_r alone. These components are learned from final correctness labels, not explicit strategy labels. Epsilon may decrease because later revisions can spoil an answer.

    predict implements this mixture in log space; BaseCritic supplies the MLP, forward pass, loss, and checkpoints.
    """

    def __init__(self, settings: CriticConfig) -> None:
        """Add heads for strategy discovery and execution to the shared MLP.

        Args:
            settings: hidden_size sets head input width; horizon - 1 sets the number of future execution budgets.

        Returns:
            None.
        """
        super().__init__(settings)
        self.horizons = settings.horizon - 1
        self.strategy_discovered = nn.Linear(settings.hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)
        self.discovery = nn.Linear(settings.hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)
        self.execution = nn.Linear(settings.hidden_size, self.horizons, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Combine discovery and execution probabilities into one final-success logit.

        Unaided rows use the mixture defined in the class docstring. Oracle rows use epsilon_r alone. Log-space calculation avoids underflow in small probabilities.

        Args:
            hidden: MLP outputs [N, hidden_size], including both reference versions.
            oracle: Solver-sketch flags [N]. True selects execution alone; a critic-only reference does not set this flag.
            remaining_blocks: Future budget [N], from 1 to self.horizons; selects epsilon_r and limits discovery delays.

        Returns:
            Tensor: Float32 log(success) - log(failure) [N]. Sigmoid gives final-success probability.
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
