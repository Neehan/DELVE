"""Hugging Face Trainer integration; Accelerate owns distributed execution and optimizer state."""

from pathlib import Path
from typing import Any, cast

from torch import nn
from transformers import Trainer
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME

from frozen_solver.models.base_critic import BaseCritic
from frozen_solver.models.checkpoint import load_critic_weights, save_critic


class CriticTrainer(Trainer):
    """Override only the count loss and compact model serialization; retain HF training/resume machinery."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Keep HF initialization and fail before training if compact saves cannot support the backend."""
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            raise ValueError("compact critic checkpoints support Trainer single-device/DDP only")
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model: nn.Module, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped model for DDP synchronization, then score its binomial predictions."""
        counts = inputs.pop("labels")
        prediction = model(**inputs)
        critic = cast(BaseCritic, self.accelerator.unwrap_model(model))
        loss = critic.loss(prediction, counts)
        return_outputs = args[0] if args else kwargs.get("return_outputs", False)
        if return_outputs:
            return loss, {"logits": prediction["logits"]}
        return loss

    def _save(self, *args: Any, **kwargs: Any) -> None:
        """Store adapters and heads only; distributed sharded saves are deliberately unsupported here."""
        output_dir = args[0] if args else kwargs.get("output_dir", self.args.output_dir)
        directory = Path(output_dir)
        save_critic(cast(BaseCritic, self.accelerator.unwrap_model(self.model)), directory)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, *args: Any, **kwargs: Any) -> None:
        """Restore trainable weights; Trainer restores optimizer, scheduler, random state, and data position."""
        directory = Path(resume_from_checkpoint)
        rng = f"rng_state_{self.args.process_index}.pth" if self.args.world_size > 1 else "rng_state.pth"
        for filename in (OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, rng):
            if not (directory / filename).is_file():
                raise FileNotFoundError(directory / filename)
        load_critic_weights(cast(BaseCritic, self.model), directory)

    def _load_best_model(self) -> None:
        """Load the model selected by held-out ordinary terminal NLL."""
        if self.state.best_model_checkpoint is None:
            raise ValueError("no best critic checkpoint was selected")
        load_critic_weights(cast(BaseCritic, self.model), Path(self.state.best_model_checkpoint))
