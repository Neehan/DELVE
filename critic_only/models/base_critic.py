"""Shared critic construction, checkpoints, prefix pooling, and count-weighted objective."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel

from critic_only.constants import CONFIG_FILENAME, MODEL_FILENAME, ORACLE, ORDINARY, SUCCESS, TRIALS
from critic_only.models.models import CriticConfig


class BaseCritic(nn.Module, ABC):
    """Share construction and save/load; subclasses implement predict and optional regularization."""

    def __init__(self, backbone: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Share backbone execution and configuration across all critic heads."""
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.settings = settings
        self.horizons = settings.horizon - 1

    @classmethod
    def create_from_pretrained(cls, settings: CriticConfig, dtype: torch.dtype) -> Self:
        """Build this critic with a pinned backbone and adapters on attention and MLP layers."""
        backbone = AutoModel.from_pretrained(settings.backbone, revision=settings.revision,
                                             dtype=dtype, attn_implementation=settings.attention)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapter = LoraConfig(r=settings.lora_rank, lora_alpha=settings.lora_alpha, target_modules="all-linear")
        adapted = get_peft_model(backbone, adapter)
        return cls(adapted, backbone.config.hidden_size, settings)

    def load_from_weights(self, directory: Path) -> None:
        """Restore saved LoRA adapters and critic heads into this model; Trainer restores optimizer state."""
        settings = CriticConfig.model_validate_json((directory / CONFIG_FILENAME).read_text())
        if settings != self.settings:
            raise ValueError("critic checkpoint configuration differs")
        weights = load_file(str(directory / MODEL_FILENAME))
        parameters = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if parameters.keys() != weights.keys():
            raise ValueError("critic checkpoint trainable parameter names differ")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(weights[name])

    def save(self, directory: Path) -> None:
        """Save adapters, prediction heads, and configuration; Trainer owns optimizer/RNG state."""
        directory.mkdir(parents=True, exist_ok=True)
        weights = {name: parameter.detach().cpu().contiguous() for name, parameter in self.named_parameters()
                   if parameter.requires_grad}
        save_file(weights, str(directory / MODEL_FILENAME))
        (directory / CONFIG_FILENAME).write_text(self.settings.model_dump_json(indent=2) + "\n")

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
        weight = self.settings.oracle_weight if self.settings.variant != "ordinary" else 0.0
        return (ordinary + weight * oracle).mean() + self.regularization(prediction, trials[:, ORDINARY])

    def regularization(self, prediction: dict[str, Tensor], trials: Tensor) -> Tensor:
        """Default heads have no structural penalty; keep the result on the prediction device."""
        return prediction["logits"].sum() * 0
