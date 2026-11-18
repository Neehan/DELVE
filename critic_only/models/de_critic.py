"""Predict exact checkpoint correctness through shared discovery and execution components."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import VALUE_OUTPUT_SIZE
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class DECritic(BaseCritic):
    """Couple future-checkpoint values through a block-based discovery–execution model.

    predict estimates strategy-discovered probability s, discovery rate alpha, and execution probabilities
    epsilon_1 through epsilon_H from each token. At remaining budget r, unaided success is
    s*epsilon_r + (1-s)*sum_{j=1}^r alpha*(1-alpha)^(j-1)*epsilon_(r-j+1).
    Oracle histories use epsilon_r. Execution can decrease with budget because revisions may spoil answers.
    BaseCritic handles Qwen, the shared MLP, curve loss and checkpoints for every variant.
    """

    def __init__(self, settings: CriticConfig) -> None:
        """Create discovery heads and one execution logit per whole-block remaining budget.

        Args:
            settings: MLP width and total solver horizon. Interior first-block tokens need budgets up to H.

        Returns:
            None.
        """
        super().__init__(settings)
        self.strategy_discovered = nn.Linear(settings.hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)
        self.discovery = nn.Linear(settings.hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32)
        self.execution = nn.Linear(settings.hidden_size, settings.horizon, dtype=torch.float32)

    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Compute all absolute checkpoint logits using the same DE parameters at each token.

        Args:
            hidden: Shared MLP outputs [B,D] or [B,L,D].
            oracle: Solver-hint flags [B]; a critic-only reference does not activate execution-only prediction.
            remaining_blocks: Final budget [B] or [B,L]. Interior tokens round up to whole blocks;
                the saved 1× endpoint uses the exact remaining budget. Earlier deadlines subtract whole blocks.

        Returns:
            Tensor: FP32 logits [B,H-1] or [B,L,H-1], for exact correctness at absolute checkpoints 2×–H×.
        """
        strategy_discovered_logits = self.strategy_discovered(hidden).float()
        alpha_logits = self.discovery(hidden).float()
        execution_logits = self.execution(hidden).float()
        deadlines_before_final = torch.arange(self.settings.horizon - 2, -1, -1, device=hidden.device)
        remaining_at_deadlines = remaining_blocks[..., None] - deadlines_before_final
        execution_indices = torch.arange(self.settings.horizon, device=hidden.device)
        delays = remaining_at_deadlines[..., None] - 1 - execution_indices

        # Broadcast over both deadline and execution-budget axes; impossible discovery delays have zero weight.
        log_weights = F.logsigmoid(alpha_logits)[..., None] + delays.clamp_min(0) * F.logsigmoid(-alpha_logits)[..., None]
        log_weights = log_weights.masked_fill(delays < 0, -torch.inf)
        log_search_success = torch.logsumexp(log_weights + F.logsigmoid(execution_logits)[..., None, :], dim=-1)
        log_acquired_failure = torch.logsumexp(log_weights + F.logsigmoid(-execution_logits)[..., None, :], dim=-1)
        log_search_failure = torch.logaddexp(remaining_at_deadlines * F.logsigmoid(-alpha_logits), log_acquired_failure)
        deadline_execution_logits = execution_logits.gather(-1, remaining_at_deadlines - 1)

        # A strategy already discovered needs only execution; otherwise marginalize over future discovery times.
        log_success = torch.logaddexp(F.logsigmoid(strategy_discovered_logits) + F.logsigmoid(deadline_execution_logits),
                                     F.logsigmoid(-strategy_discovered_logits) + log_search_success)
        log_failure = torch.logaddexp(F.logsigmoid(strategy_discovered_logits) + F.logsigmoid(-deadline_execution_logits),
                                     F.logsigmoid(-strategy_discovered_logits) + log_search_failure)
        oracle_mask = oracle.reshape((-1,) + (1,) * (deadline_execution_logits.ndim - 1))
        return torch.where(oracle_mask, deadline_execution_logits, log_success - log_failure)
