"""Hugging Face Trainer integration; Accelerate owns distributed execution and optimizer state."""

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from critic_only.constants import PAIRED_INPUT_WEIGHT, TRAINING_ARGS_FILENAME
from critic_only.models.base_critic import BaseCritic


class BaseTrainer(Trainer):
    """Override only the count loss and compact model serialization; retain HF training/resume machinery."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Keep HF initialization and fail before training if compact saves cannot support the backend."""
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            raise ValueError("compact critic checkpoints support Trainer single-device/DDP only")
        self.model_accepts_loss_kwargs = False
        self.args.disable_tqdm = True
        self.remove_callback(PrinterCallback)
        self.remove_callback(ProgressCallback)

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Flush one metric record and atomically persist HF history on the primary process."""
        super().log(logs, *args, **kwargs)
        if self.is_world_process_zero():
            print(json.dumps(self.state.log_history[-1]), flush=True)
            directory = Path(cast(str, self.args.output_dir))
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / f"{TRAINER_STATE_NAME}.tmp"
            self.state.save_to_json(str(temporary))
            temporary.replace(directory / TRAINER_STATE_NAME)

    def compute_loss(self, model: nn.Module, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Combine reference-conditioned and ordinary input losses into one training objective.

        The batch contains B input views without reference solutions followed by P privileged views that include a reference solution. labels contains counts for the B prefixes; privileged_indices maps the P extra views to those same labels. A single model forward processes all B+P views.

        BaseCritic.loss returns one loss per prefix and input view. A prefix with a reference contributes (without_reference_loss + with_reference_loss)/2; a prefix without one contributes its full without-reference loss. The final scalar is the mean over B prefixes, so adding a reference does not double a prefix's weight. Predictions themselves are never averaged.

        Each example has one history and one set of continuation labels. Oracle histories already include the sketch message; a privileged view appends the reference solution to that same history.

        Returns:
            The scalar loss, or (loss, outputs) when Trainer requests predictions. Outputs retain separate terminal logits for both views, each shaped [B], plus a [B] privileged_available mask. Zero-filled privileged logits for missing references must be ignored using that mask.
        """
        counts = inputs.pop("labels")
        indices = inputs.pop("privileged_indices")
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], oracle=inputs["oracle"],
                       remaining_blocks=inputs["remaining_blocks"])
        unprivileged_logits = logits[:len(counts)]
        oracle = inputs["oracle"][:len(counts)]
        critic = cast(BaseCritic, self.accelerator.unwrap_model(model))
        losses = critic.loss(unprivileged_logits, counts, oracle)
        privileged_logits = torch.zeros_like(unprivileged_logits)
        available = torch.zeros(len(counts), dtype=torch.bool, device=counts.device)
        if indices.numel():
            privileged = logits[len(counts):]
            privileged_losses = critic.loss(privileged, counts[indices], oracle[indices])
            losses = losses.index_copy(0, indices, PAIRED_INPUT_WEIGHT * (losses[indices] + privileged_losses))
            privileged_logits = privileged_logits.index_copy(0, indices, privileged)
            available[indices] = True
        loss = losses.mean()
        return_outputs = args[0] if args else kwargs.get("return_outputs", False)
        if return_outputs:
            return loss, {"logits": unprivileged_logits, "privileged_logits": privileged_logits,
                          "privileged_available": available}
        return loss

    def _save(self, *args: Any, **kwargs: Any) -> None:
        """Store adapters and heads only; distributed sharded saves are deliberately unsupported here."""
        output_dir = args[0] if args else kwargs.get("output_dir", self.args.output_dir)
        directory = Path(output_dir)
        cast(BaseCritic, self.accelerator.unwrap_model(self.model)).save(directory)
        (directory / TRAINING_ARGS_FILENAME).write_text(self.args.to_json_string() + "\n")

    def _load_from_checkpoint(self, resume_from_checkpoint: str, *args: Any, **kwargs: Any) -> None:
        """Restore trainable weights; Trainer restores optimizer, scheduler, random state, and data position."""
        directory = Path(resume_from_checkpoint)
        rng = f"rng_state_{self.args.process_index}.pth" if self.args.world_size > 1 else "rng_state.pth"
        for filename in (OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, rng):
            if not (directory / filename).is_file():
                raise FileNotFoundError(directory / filename)
        cast(BaseCritic, self.model).load_from_weights(directory)

    def _load_best_model(self) -> None:
        """Load the model selected by held-out problem-macro terminal NLL without references."""
        if self.state.best_model_checkpoint is None:
            raise ValueError("no best critic checkpoint was selected")
        cast(BaseCritic, self.model).load_from_weights(Path(self.state.best_model_checkpoint))
