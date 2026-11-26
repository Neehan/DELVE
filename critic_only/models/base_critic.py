"""Share the MLP, forward pass, loss weighting, and checkpoint code used by the critics."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import CONFIG_FILENAME, INPUT_PREFIX_BLOCKS, MODEL_FILENAME
from critic_only.models.models import CriticConfig


class BaseCritic(nn.Module, ABC):
    """Turn Qwen token states into critic predictions.

    Subclasses implement predict with num_checkpoints outputs for every supplied token. This base owns the output count, MLP, checkpoint-curve BCE and checkpoint storage.
    """

    def __init__(self, settings: CriticConfig) -> None:
        """Build the shared two-layer GELU MLP.

        The loader attaches Qwen before forward is called. Every mode predicts from token states.

        Args:
            settings: Input vector size, MLP width, loss weight, and checkpoint settings.

        Returns:
            None.
        """
        super().__init__()
        self.settings = settings
        self.num_checkpoints = settings.horizon - INPUT_PREFIX_BLOCKS
        self.language_model: nn.Module
        self.hidden_state_projection = nn.Sequential(nn.Linear(settings.input_size, settings.hidden_size), nn.GELU(),
                                   nn.Linear(settings.hidden_size, settings.hidden_size), nn.GELU())

    def load_from_weights(self, directory: Path) -> None:
        """Restore saved trainable weights into this already constructed critic.

        The configuration, parameter names, and shapes must match before any weights are copied.

        Args:
            directory: Checkpoint directory containing critic configuration and safetensors weights.

        Returns:
            None. Updates the MLP, prediction heads, and trainable Qwen parameters; Trainer restores optimizer state separately.
        """
        settings = CriticConfig.model_validate_json((directory / CONFIG_FILENAME).read_text())
        if settings != self.settings:
            raise ValueError("critic checkpoint configuration differs")
        weights = load_file(str(directory / MODEL_FILENAME))
        parameters = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        self._copy_weights(weights, parameters)

    def _copy_weights(self, weights: dict[str, Tensor], parameters: dict[str, nn.Parameter]) -> None:
        """Check all weight names and shapes before copying any saved weights into the model.

        Args:
            weights: Saved tensors to restore.
            parameters: Exact destination parameter set; names and shapes must match.

        Returns:
            None. Copies weights only after all validation succeeds.
        """
        if parameters.keys() != weights.keys():
            raise ValueError("critic checkpoint trainable parameter names differ")
        for name, parameter in parameters.items():
            if weights[name].shape != parameter.shape:
                raise ValueError(f"critic checkpoint shape differs for {name}: expected {parameter.shape}, got {weights[name].shape}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(weights[name])

    def save(self, directory: Path) -> None:
        """Save the critic settings and trainable weights.

        Includes the MLP, prediction heads, and trainable Qwen parameters. Frozen Qwen weights are identified by model revision rather than copied into each checkpoint.

        Args:
            directory: Destination checkpoint directory, created if needed.

        Returns:
            None. Writes configuration and safetensors files.
        """
        directory.mkdir(parents=True, exist_ok=True)
        weights = {name: parameter.detach().cpu().contiguous() for name, parameter in self.named_parameters()
                   if parameter.requires_grad}
        save_file(weights, str(directory / MODEL_FILENAME))
        (directory / CONFIG_FILENAME).write_text(self.settings.model_dump_json(indent=2) + "\n")

    def forward(self, inputs: dict[str, Tensor], oracle: Tensor, remaining_blocks: Tensor) -> dict[str, Tensor]:
        """Predict a checkpoint curve at every token of the live causal transformer.

        Args:
            inputs: input_ids and attention_mask [B,L]. Saved inputs end at the last solver token; solver_mask marks training positions.
            oracle: Solver-hint flags [B], separate from critic reference availability.
            remaining_blocks: Budget to the final checkpoint [B,L], aligned with token positions.

        Returns:
            dict: logits [B,L,H-1], plus any variant-specific tensors needed by its trainer.
        """
        output = self.language_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        hidden_states = output.last_hidden_state
        projected = self.hidden_state_projection(hidden_states.float())
        return self.predict(projected, oracle, remaining_blocks)

    @abstractmethod
    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> dict[str, Tensor]:
        """Map MLP outputs to exact future-checkpoint logits; implemented by each variant.

        Args:
            hidden: MLP outputs [B,L,D] for token prediction.
            oracle: Solver-sketch flags [B]; DE uses these to select execution-only predictions.
            remaining_blocks: Whole-block final budget [B,L], aligned with hidden states.

        Returns:
            dict: Float32 logits [B,L,H-1], in absolute checkpoint order 2× through H×.
            Subclasses may also return tensors needed for their additional loss terms.
        """
        raise NotImplementedError

    def loss(self, logits: Tensor, labels: Tensor, oracle: Tensor) -> Tensor:
        """Compute BCE for each checkpoint without reducing tokens, horizons, or trajectories.

        Args:
            logits: Causal token logits [B,L,H-1].
            labels: Exact checkpoint grades [B,H-1]; each token predicts those same future outcomes.
            oracle: Solver-hint flags [B]; oracle_weight scales those trajectories' losses.

        Returns:
            Tensor: FP32 BCE losses with the same shape as logits. Trainer masks non-solver positions,
            averages horizons and valid tokens within each trajectory, then averages trajectories.
        """
        labels = labels[:, None, :].expand_as(logits)
        losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
        return self.weight_losses(losses, oracle)

    def weight_losses(self, losses: Tensor, oracle: Tensor) -> Tensor:
        """Apply the same trajectory weights to BCE and any auxiliary loss.

        Args:
            losses: Unreduced token/checkpoint losses [B,L,C].
            oracle: Solver-hint flags [B]; oracle_weight scales all losses for these rows.

        Returns:
            Tensor: Weighted losses [B,L,C], before the trainer masks and averages them.
        """
        weight = torch.where(oracle, self.settings.oracle_weight, 1.0)
        return weight[:, None, None] * losses
