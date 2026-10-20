"""A soft discovery-execution prior with a penalized correction at every horizon."""

import torch
from torch import Tensor, nn

from critic_only.constants import DE_STATE_FIELDS, PROBABILITY_EPSILON
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig


class DECritic(BaseCritic):
    """Learn acquisition, execution, and residual heads; latent parameters are not strategy labels."""

    def __init__(self, backbone: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Keep the execution curve unrestricted to allow correct-to-incorrect revisions."""
        super().__init__(backbone, hidden_size, settings)
        self.acquisition = nn.Linear(hidden_size, DE_STATE_FIELDS, dtype=torch.float32)
        self.execution = nn.Linear(hidden_size, self.horizons, dtype=torch.float32)
        self.correction = nn.Linear(hidden_size, self.horizons, dtype=torch.float32)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def predict(self, hidden: Tensor) -> dict[str, Tensor]:
        """Mix already-acquired execution with geometric acquisition, then correct its logits."""
        # Keep probabilities in FP32: BF16 rounds the upper clamp bound to exactly one.
        discovered, alpha = self.acquisition(hidden).float().sigmoid().unbind(dim=-1)
        execution_logits = self.execution(hidden).float()
        epsilon = execution_logits.sigmoid()
        curve = []
        for horizon in range(self.horizons):
            value = discovered * epsilon[:, horizon]
            for delay in range(horizon + 1):
                value = value + (1 - discovered) * alpha * (1 - alpha).pow(delay) * epsilon[:, horizon - delay]
            curve.append(value)
        prior = torch.stack(curve, dim=-1).clamp(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        correction = self.correction(hidden).float()
        ordinary_logits = torch.logit(prior) + correction
        return {"logits": torch.stack((ordinary_logits, execution_logits), dim=1),
                "alpha": alpha, "discovered": discovered, "epsilon": epsilon, "correction": correction}

    def regularization(self, prediction: dict[str, Tensor], trials: Tensor) -> Tensor:
        """Penalize corrections only at observed horizons, making DE a soft structural prior."""
        mask = trials > 0
        squared = prediction["correction"].square() * mask
        return self.settings.correction_weight * (squared.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)).mean()
