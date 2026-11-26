"""Predict exact checkpoint correctness through shared discovery and execution components."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import INPUT_PREFIX_BLOCKS, VALUE_OUTPUT_SIZE
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class DECritic(BaseCritic):
    """Predict checkpoint correctness from strategy discovery and execution.

    At each token, predict estimates alpha (discovery probability per block) and epsilon_r (correctness after r execution blocks). With r blocks left, unaided value is sum_{d=0}^{r-1} alpha*(1-alpha)^d*epsilon_(r-d). Oracle histories use epsilon_r directly because the sketch supplies a strategy.

    Execution probabilities need not increase with budget: later revisions can spoil an answer. BaseCritic supplies Qwen, the shared MLP, loss and checkpoint storage; this class implements predict and the discovery-time sum.
    """

    execution_blocks: Tensor
    blocks_before_discovery: Tensor
    target_checkpoints: Tensor

    def __init__(self, settings: CriticConfig) -> None:
        """Create one discovery head and one execution logit per whole-block remaining budget.

        Args:
            settings: MLP width and total solver horizon. Interior first-block tokens need budgets up to H.

        Returns:
            None.
        """
        super().__init__(settings)
        self.discovery = nn.Linear(
            settings.hidden_size, VALUE_OUTPUT_SIZE, dtype=torch.float32
        )
        self.execution = nn.Linear(
            settings.hidden_size, settings.horizon, dtype=torch.float32
        )

        # Buffers are fixed tensors that PyTorch moves to the GPU with the model; they are not learned.
        execution_blocks = torch.arange(1, settings.horizon + 1)
        self.register_buffer("execution_blocks", execution_blocks, persistent=False)
        # With 3 blocks left: 3 execution blocks means delay 0; 2 means delay 1; 1 means delay 2.
        self.register_buffer(
            "blocks_before_discovery",
            execution_blocks[:, None] - execution_blocks,
            persistent=False,
        )
        # The four prediction targets are correctness at total compute 2×, 3×, 4× and 5×.
        self.register_buffer(
            "target_checkpoints",
            torch.arange(INPUT_PREFIX_BLOCKS + 1, settings.horizon + 1),
            persistent=False,
        )

    def predict(
        self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor
    ) -> dict[str, Tensor]:
        """Build the DE value curve, then select the requested checkpoint predictions.

        Args:
            hidden: Shared MLP outputs [B,L,D] for all tokens.
            oracle: Solver-hint flags [B]; critic-only references do not count as hints.
            remaining_blocks: Blocks until the final checkpoint, [B,L].

        Returns:
            dict: FP32 logits [B,L,C], with C=num_checkpoints (four for 2×–5×).
        """
        discovery_logits = self.discovery(hidden).float()
        execution_logits = self.execution(hidden).float()
        unaided_curve = self._compute_unaided_curve(discovery_logits, execution_logits)
        # One oracle flag per history applies to every token and every remaining budget.
        oracle_mask = oracle.reshape((-1,) + (1,) * (execution_logits.ndim - 1))
        # Oracle supplies the strategy: use epsilon directly.
        value_curve = torch.where(oracle_mask, execution_logits, unaided_curve)
        return {"logits": self._select_checkpoints(value_curve, remaining_blocks)}

    def _compute_unaided_curve(
        self,
        discovery_logits: Tensor,
        execution_logits: Tensor,
    ) -> Tensor:
        """Compute unaided success logits from geometric discovery followed by execution.

        Args:
            discovery_logits: Logits of discovering a strategy per future block, [...,1].
            execution_logits: Logits of correct execution with 1 through H blocks, [...,H].

        Returns:
            Tensor: Unaided value logits [...,H] for 1 through H remaining blocks, before checkpoint selection.
        """
        log_discovery = F.logsigmoid(
            discovery_logits
        )  # log(alpha): discover in this block.
        log_no_discovery = F.logsigmoid(
            -discovery_logits
        )  # log(1-alpha): no discovery in this block.
        # With r blocks left and k execution blocks, d=r-k blocks pass before discovery.
        # The added axis broadcasts each token's discovery rate across all (r, k) combinations.
        log_discovery_weights = (
            log_discovery[..., None]  # log(alpha): discovery after the wait.
            + self.blocks_before_discovery.clamp_min(0)
            * log_no_discovery[..., None]  # d * log(1-alpha): the wait.
        )
        # Negative delays mean k > r: clamp above only for arithmetic, then assign probability zero.
        log_discovery_weights = log_discovery_weights.masked_fill(
            self.blocks_before_discovery < 0, -torch.inf
        )
        # log(sum_k alpha*(1-alpha)^(r-k)*epsilon_k): add log execution success, then sum over k.
        log_success_after_discovery = torch.logsumexp(
            log_discovery_weights + F.logsigmoid(execution_logits)[..., None, :], dim=-1
        )
        # Same discovery weights, but multiply by execution failure (1-epsilon_k).
        log_failure_after_discovery = torch.logsumexp(
            log_discovery_weights + F.logsigmoid(-execution_logits)[..., None, :],
            dim=-1,
        )
        log_never_discovered = (
            self.execution_blocks * log_no_discovery
        )  # log((1-alpha)^r): no discovery by r.
        # logaddexp adds probabilities: never discover OR discover and execute incorrectly.
        log_discovery_execution_failure = torch.logaddexp(
            log_never_discovered, log_failure_after_discovery
        )
        # log(p) - log(1-p) gives logits without rounding small probabilities to zero.
        return log_success_after_discovery - log_discovery_execution_failure

    def _select_checkpoints(
        self, value_curve: Tensor, remaining_blocks: Tensor
    ) -> Tensor:
        """Map remaining-budget values to absolute target checkpoints 2× through H×.

        Args:
            value_curve: Value logits [...,H], indexed by 1 through H remaining blocks.
            remaining_blocks: Blocks left until H×, aligned with the leading dimensions of value_curve.

        Returns:
            Tensor: Checkpoint logits [...,C]. At the 1× endpoint select budgets 1–4; earlier tokens use the agreed rounded-up budgets 2–5 for H=5.
        """
        checkpoint_offsets = self.settings.horizon - self.target_checkpoints
        remaining_blocks_per_checkpoint = (
            remaining_blocks[..., None] - checkpoint_offsets
        )
        return value_curve.gather(-1, remaining_blocks_per_checkpoint - 1)
