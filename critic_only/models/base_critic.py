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

from critic_only.constants import CONFIG_FILENAME, MODEL_FILENAME, SUCCESS_COUNT_INDEX, TRIAL_COUNT_INDEX
from critic_only.models.models import CriticConfig


class BaseCritic(nn.Module, ABC):
    """Construct, restore, and score critics; subclasses implement predict."""

    def __init__(self, model: nn.Module, hidden_size: int, settings: CriticConfig) -> None:
        """Share model execution and configuration across all critic heads."""
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.settings = settings

    @classmethod
    def create_from_pretrained(cls, settings: CriticConfig, dtype: torch.dtype) -> Self:
        """Build this critic with a pinned model and adapters on attention and MLP layers."""
        model = AutoModel.from_pretrained(settings.model, revision=settings.revision,
                                          dtype=dtype, attn_implementation=settings.attention)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapter = LoraConfig(r=settings.lora_rank, lora_alpha=settings.lora_alpha, target_modules="all-linear")
        adapted = get_peft_model(model, adapter)
        return cls(adapted, model.config.hidden_size, settings)

    def load_from_weights(self, directory: Path) -> None:
        """Restore saved LoRA adapters and critic heads into this model; Trainer restores optimizer state."""
        settings = CriticConfig.model_validate_json((directory / CONFIG_FILENAME).read_text())
        if settings != self.settings:
            raise ValueError("critic checkpoint configuration differs")
        weights = load_file(str(directory / MODEL_FILENAME))
        parameters = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if parameters.keys() != weights.keys():
            raise ValueError("critic checkpoint trainable parameter names differ")
        for name, parameter in parameters.items():
            if weights[name].shape != parameter.shape:
                raise ValueError(f"critic checkpoint shape differs for {name}: expected {parameter.shape}, got {weights[name].shape}")
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

    def forward(self, input_ids: Tensor, attention_mask: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Read the whole history once and map its last non-padding hidden state to a terminal-value logit."""
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        positions = attention_mask.sum(dim=-1) - 1
        hidden = output.last_hidden_state[torch.arange(input_ids.shape[0], device=input_ids.device), positions]
        return self.predict(hidden.float(), oracle, remaining_blocks)

    @abstractmethod
    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Return one terminal-value logit per history [views]; DE uses remaining_blocks to select its horizon."""
        raise NotImplementedError

    def loss(self, logits: Tensor, counts: Tensor, oracle: Tensor) -> Tensor:
        """Compute one count-weighted prediction loss per actual solver history.

        Args:
            logits: Predicted terminal success logits [examples] for each actual history.
            counts: Labels [examples, 2], with [successful branches, total branches]. Success grades the last answer when the remaining budget is exhausted, not any earlier answer.
            oracle: Boolean mask [examples] identifying histories with an injected sketch, used only for loss weighting.

        For successes s, trials n, and probability p = sigmoid(logit), the negative log-likelihood is -s*log(p) - (n-s)*log(1-p). Thus [3, 8] represents three successes and five failures. The parameter-independent binomial coefficient is omitted.

        Divide by the number of observed branches for this history, giving each history equal weight regardless of branch count. Only terminal outcomes contribute; earlier checkpoints are not auxiliary targets.

        Oracle examples receive oracle_weight; ordinary examples receive weight one. Ordinary-only training excludes oracle examples before batching.

        Returns:
            Losses [examples], without batch averaging. BaseTrainer averages the paired reference/no-reference losses for each example, then averages examples. References and generated oracle sketches are distinct inputs.
        """
        logits = logits.float()
        successes, trials = counts[..., SUCCESS_COUNT_INDEX], counts[..., TRIAL_COUNT_INDEX]
        nll = -successes * F.logsigmoid(logits) - (trials - successes) * F.logsigmoid(-logits)
        normalized = nll / trials
        weight = torch.where(oracle, self.settings.oracle_weight, 1.0)
        return weight * normalized
