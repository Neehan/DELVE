"""Shared feature projection, checkpoints, and binary terminal-outcome loss."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import CONFIG_FILENAME, MODEL_FILENAME
from critic_only.models.models import CriticConfig


class BaseCritic(nn.Module, ABC):
    """Construct, restore, and score critics; subclasses implement predict."""

    def __init__(self, settings: CriticConfig) -> None:
        """Project frozen Qwen vectors through the same MLP for every critic variant."""
        super().__init__()
        self.settings = settings
        self.model = nn.Sequential(nn.Linear(settings.input_size, settings.hidden_size), nn.GELU(),
                                   nn.Linear(settings.hidden_size, settings.hidden_size), nn.GELU())

    def load_from_weights(self, directory: Path) -> None:
        """Restore the MLP and prediction heads; Trainer restores optimizer and RNG state."""
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
        """Save the complete MLP critic and configuration; Trainer owns optimizer and RNG state."""
        directory.mkdir(parents=True, exist_ok=True)
        weights = {name: parameter.detach().cpu().contiguous() for name, parameter in self.named_parameters()
                   if parameter.requires_grad}
        save_file(weights, str(directory / MODEL_FILENAME))
        (directory / CONFIG_FILENAME).write_text(self.settings.model_dump_json(indent=2) + "\n")

    def forward(self, hidden_state: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Map cached final-token vectors [views, input_size] to one terminal logit per view."""
        return self.predict(self.model(hidden_state.float()), oracle, remaining_blocks)

    @abstractmethod
    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Return one terminal-value logit per history [views]; DE uses remaining_blocks to select its horizon."""
        raise NotImplementedError

    def loss(self, logits: Tensor, labels: Tensor, oracle: Tensor) -> Tensor:
        """Compute binary cross-entropy for each observed solver history.

        Args:
            logits: Predicted terminal success logits [examples] for each actual history.
            labels: Binary final correctness [examples]. Success grades the last answer when the remaining budget is exhausted, not any earlier answer.
            oracle: Boolean mask [examples] identifying histories with an injected sketch, used only for loss weighting.

        For label y and probability p = sigmoid(logit), the loss is -y*log(p) - (1-y)*log(1-p). Compute directly from logits for numerical stability. Earlier checkpoint grades are not auxiliary targets.

        Oracle examples receive oracle_weight; unaided examples receive weight one. PPOCritic training excludes oracle examples before batching.

        Returns:
            Losses [examples], without batch averaging. BaseTrainer averages the paired reference/no-reference losses for each example, then averages examples. Critic-only references and solver-visible sketches are distinct inputs.
        """
        losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
        weight = torch.where(oracle, self.settings.oracle_weight, 1.0)
        return weight * losses
