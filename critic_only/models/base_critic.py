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

    Subclasses implement predict. PPO variants enable token_supervision for live Qwen; DE keeps its endpoint-only block model. Cached vectors always supply one endpoint. This base shares the MLP, Monte Carlo BCE and checkpoint storage.
    """

    token_supervision = False

    def __init__(self, settings: CriticConfig) -> None:
        """Build the shared two-layer GELU MLP.

        language_model starts as None for cached-vector inputs. The loader attaches Qwen for token-based training.

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
            None. Updates the MLP, prediction heads, and trainable Qwen parameters; Trainer restores optimizer state separately.
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

    def forward(self, inputs: dict[str, Tensor], oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Predict values from cached endpoints or a live causal transformer.

        Args:
            inputs: Cached hidden_state [B,H], or input_ids/attention_mask [B,L]. Saved token inputs end at
                the last solver token. solver_mask marks eligible training positions and is used by Trainer.
            oracle: Solver-hint flags [B], separate from critic reference availability.
            remaining_blocks: Endpoint budget [B], used only by the block-based DE prediction.

        Returns:
            Tensor: PPO/OraclePPO return logits [B,L] for live tokens, one value per causal prefix. Cached
            features and DE return endpoint logits [B]. No hidden states are averaged across positions.
        """
        if self.language_model is None:
            hidden_states = inputs["hidden_state"]
        else:
            output = self.language_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
            hidden_states = output.last_hidden_state
            if not self.token_supervision:
                positions = inputs["attention_mask"].sum(dim=-1) - 1
                hidden_states = hidden_states[torch.arange(len(positions), device=positions.device), positions]
        projected = self.hidden_state_projection(hidden_states.float())
        return self.predict(projected, oracle, remaining_blocks)

    @abstractmethod
    def predict(self, hidden: Tensor, oracle: Tensor, remaining_blocks: Tensor) -> Tensor:
        """Map MLP outputs to final-success logits; each critic variant implements this.

        Args:
            hidden: MLP outputs [B,H] for endpoints or [B,L,H] for token-wise PPO prediction.
            oracle: Solver-sketch flags [B]; DE uses these to select execution-only predictions.
            remaining_blocks: Endpoint budget [B], from 1 through settings.horizon - 1.

        Returns:
            Tensor: Float32 logits [B] or [B,L], matching the leading dimensions of hidden. These go directly into binary cross-entropy.
        """
        raise NotImplementedError

    def loss(self, logits: Tensor, labels: Tensor, oracle: Tensor) -> Tensor:
        """Compute Monte Carlo binary cross-entropy without reducing token or trajectory dimensions.

        Args:
            logits: Endpoint logits [B] or causal token logits [B,L].
            labels: Each trajectory's verified final outcome [B]. For token predictions, the same sampled
                return supervises every position; intermediate correctness labels are not required.
            oracle: Solver-hint flags [B]; oracle_weight scales those trajectories' losses.

        Returns:
            Tensor: FP32 BCE losses with the same shape as logits. Trainer masks non-solver positions,
            averages within each trajectory, then averages trajectories across the optimizer update.
        """
        if logits.ndim == 2:
            labels = labels[:, None].expand_as(logits)
            oracle = oracle[:, None]
        losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
        weight = torch.where(oracle, self.settings.oracle_weight, 1.0)
        return weight * losses
