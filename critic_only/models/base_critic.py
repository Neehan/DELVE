"""Share the MLP, forward pass, loss, and checkpoint code used by all three critics."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from critic_only.constants import CONFIG_FILENAME, MODEL_FILENAME
from critic_only.models.models import CriticConfig


class BaseCritic(nn.Module, ABC):
    """Turn cached vectors or Qwen outputs into critic predictions.

    forward uses cached vectors when language_model is None; otherwise it runs Qwen with LoRA. Both paths use the same MLP. Subclasses implement predict; this base supplies the per-prediction loss and trainable-weight save/load methods.
    """

    def __init__(self, settings: CriticConfig) -> None:
        """Build the shared two-layer GELU MLP.

        language_model starts as None for cached-vector inputs. The loader attaches Qwen with adapters for LoRA mode.

        Args:
            settings: Input vector size, MLP width, loss weight, and checkpoint settings.

        Returns:
            None.
        """
        super().__init__()
        self.settings = settings
        self.language_model: nn.Module | None = None
        self.hidden_state_projection = nn.Sequential(nn.Linear(settings.input_size, settings.hidden_size), nn.GELU(),
                                   nn.Linear(settings.hidden_size, settings.hidden_size), nn.GELU())

    def load_from_weights(self, directory: Path) -> None:
        """Restore saved trainable weights into this already constructed critic.

        The configuration, parameter names, and shapes must match before any weights are copied.

        Args:
            directory: Checkpoint directory containing critic configuration and safetensors weights.

        Returns:
            None. Updates the MLP, prediction heads, and any adapters; Trainer restores optimizer state separately.
        """
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
        """Save the critic settings and trainable weights.

        Includes the MLP, prediction heads, and any LoRA adapters. Frozen Qwen weights are identified by model revision rather than copied into each checkpoint.

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

    def forward(self, inputs: dict[str, Tensor], oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Convert each input into a vector, apply the MLP, and predict final answer correctness.

        N counts both versions of the inputs. Frozen mode uses saved vectors; LoRA computes Qwen's last non-padding token vector with gradients. Neither path generates text.

        Args:
            inputs: Cached hidden_state [N, input_size], or LoRA input_ids and attention_mask [N, L]. Tokens are right-padded; mask entries are 1 for real tokens and 0 for padding.
            oracle: Solver-sketch flags [N], not flags for critic-only references.
            remaining_blocks: Budget left after each history [N]. Paired reference inputs share the same budget.

        Returns:
            Tensor: Float32 success logits [N]; sigmoid gives probabilities.
        """
        if self.language_model is None:
            hidden_states = inputs["hidden_state"]
        else:
            output = self.language_model(**inputs, use_cache=False)
            positions = inputs["attention_mask"].sum(dim=-1) - 1
            hidden_states = output.last_hidden_state[torch.arange(len(positions), device=positions.device), positions]
        projected = self.hidden_state_projection(hidden_states.float())
        return self.predict(projected, oracle, remaining_blocks)

    @abstractmethod
    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Map MLP outputs to final-success logits; each critic variant implements this.

        Args:
            hidden: MLP outputs [N, settings.hidden_size], including both reference versions.
            oracle: Solver-sketch flags [N]; DE uses these to select execution-only predictions.
            remaining_blocks: Future budget [N], from 1 through settings.horizon - 1.

        Returns:
            Tensor: Float32 logits [N], before sigmoid. These go directly into the binary cross-entropy loss.
        """
        raise NotImplementedError

    def loss(self, logits: Tensor, labels: Tensor, oracle: Tensor) -> Tensor:
        """Measure prediction error against the trajectory's final correct/incorrect answer.

        This computes binary cross-entropy for each supplied prediction. It does not average examples or pair reference inputs; BaseTrainer.compute_loss does that.

        Args:
            logits: Predicted success logits [N]; sigmoid(logit) is the probability of success.
            labels: Final answer correctness [N]: 1 for correct, 0 for incorrect.
            oracle: Solver-sketch flags [N]. True applies settings.oracle_weight to that loss. A critic-only reference solution does not change this flag.

        Returns:
            Tensor: Float32 losses [N], one per prediction. Computed directly from logits for numerical stability.
        """
        losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
        weight = torch.where(oracle, self.settings.oracle_weight, 1.0)
        return weight * losses
