"""Shared LoRA backbone, prefix pooling, and count-weighted critic objective."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from frozen_solver.constants import ORACLE, ORDINARY, SUCCESS, TRIALS
from frozen_solver.models.models import CriticConfig


class BaseCritic(nn.Module, ABC):
    """Forward pools the last visible token; subclasses implement predict and optional regularization."""

    def __init__(self, backbone: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Share backbone execution and configuration across all critic heads."""
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.settings = settings
        self.horizons = settings.horizon - 1

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """Pool right-padded inputs without materializing vocabulary logits."""
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        positions = attention_mask.sum(dim=-1) - 1
        hidden = output.last_hidden_state[torch.arange(input_ids.shape[0], device=input_ids.device), positions]
        return self.predict(hidden.float())

    @abstractmethod
    def predict(self, hidden: Tensor) -> dict[str, Tensor]:
        """Return ordinary/oracle logits shaped [batch, condition, remaining horizon]."""
        raise NotImplementedError

    def loss(self, prediction: dict[str, Tensor], counts: Tensor) -> Tensor:
        """Average per-prefix binomial NLLs; absent oracle labels contribute exactly zero."""
        logits = prediction["logits"].float()
        successes, trials = counts[..., SUCCESS], counts[..., TRIALS]
        nll = -successes * F.logsigmoid(logits) - (trials - successes) * F.logsigmoid(-logits)
        ordinary = nll[:, ORDINARY].sum(dim=-1) / trials[:, ORDINARY].sum(dim=-1).clamp_min(1)
        oracle = nll[:, ORACLE].sum(dim=-1) / trials[:, ORACLE].sum(dim=-1).clamp_min(1)
        weight = self.settings.oracle_weight if self.settings.variant != "direct" else 0.0
        return (ordinary + weight * oracle).mean() + self.regularization(prediction, trials[:, ORDINARY])

    def regularization(self, prediction: dict[str, Tensor], trials: Tensor) -> Tensor:
        """Default heads have no structural penalty; keep the result on the prediction device."""
        return prediction["logits"].sum() * 0
