"""Train critics on the same trajectory with and without a reference solution, averaging the two losses."""

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from critic_only.constants import TRAINING_ARGS_FILENAME
from critic_only.models.base_critic import BaseCritic


class BaseTrainer(Trainer):
    """Add paired-reference loss, metric logging, and critic checkpoints to Hugging Face Trainer.

    compute_loss defines our objective. Hugging Face handles batching, optimization, and resume; subclasses need no additional overrides.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize Hugging Face training with our loss, logging, and checkpoint methods.

        Only single-device and DDP training are supported by our checkpoint format. Progress bars are replaced with periodic JSON logs.

        Args:
            *args: Positional arguments forwarded to Trainer.
            **kwargs: Trainer options such as model, datasets, collator, and training arguments.

        Returns:
            None.
        """
        super().__init__(*args, **kwargs)
        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            raise ValueError(
                "compact critic checkpoints support Trainer single-device/DDP only"
            )
        # Request the total label count so uneven microbatches retain equal per-trajectory weight.
        self.model_accepts_loss_kwargs = True
        self.args.disable_tqdm = True
        self.remove_callback(PrinterCallback)
        self.remove_callback(ProgressCallback)

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Print and save the latest metrics so progress survives a job cancellation.

        Args:
            logs: Metrics supplied by Trainer, such as loss and learning rate.
            *args: Additional positional arguments forwarded to Trainer.log.
            **kwargs: Additional keyword arguments forwarded to Trainer.log.

        Returns:
            None. The primary process updates trainer_state.json after every log.
        """
        super().log(logs, *args, **kwargs)
        if self.is_world_process_zero():
            print(json.dumps(self.state.log_history[-1]), flush=True)
            directory = Path(cast(str, self.args.output_dir))
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / f"{TRAINER_STATE_NAME}.tmp"
            self.state.save_to_json(str(temporary))
            temporary.replace(directory / TRAINER_STATE_NAME)

    def compute_loss(
        self, model: nn.Module, inputs: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Average each trajectory's losses with and without its reference solution.

        Both predictions use the same final correctness label. With a reference, trajectory_loss = (loss_without_reference + loss_with_reference) / 2. Without one, trajectory_loss = loss_without_reference. Normalize over trajectories in the entire optimizer update, including accumulated microbatches.

        Args:
            model: Critic, possibly wrapped for distributed training; run it once for both input versions.
            inputs: B labels and the batched model inputs. The first B inputs omit references; the remaining P include them. privileged_indices [P] identifies which of the B trajectories those extra inputs belong to. oracle identifies solver-visible sketches, not critic-only references.
            *args: Hugging Face may pass return_outputs as the first positional argument.
            **kwargs: return_outputs requests evaluation predictions. num_items_in_batch counts trajectories across the accumulated batches, and across ranks when configured.

        Returns:
            Tensor: Mean training loss, or (loss, predictions) for evaluation. Predictions keep the two input versions separate.
        """
        labels = inputs["labels"]
        reference_indices = inputs["privileged_indices"]
        logits = model(
            inputs=inputs["inputs"],
            oracle=inputs["oracle"],
            remaining_blocks=inputs["remaining_blocks"],
        )
        # The batcher places all inputs without references first, then those with references.
        logits_without_reference = logits[: len(labels)]
        logits_with_reference = logits[len(labels) :]
        oracle = inputs["oracle"][: len(labels)]
        critic = cast(BaseCritic, self.accelerator.unwrap_model(model))
        loss_without_reference = critic.loss(logits_without_reference, labels, oracle)
        trajectory_losses = loss_without_reference
        if reference_indices.numel():
            loss_with_reference = critic.loss(
                logits_with_reference,
                labels[reference_indices],
                oracle[reference_indices],
            )
            paired_losses = 0.5 * (
                loss_without_reference[reference_indices] + loss_with_reference
            )
            # Replace only trajectories that have a reference; unpaired ones keep their original loss.
            trajectory_losses = trajectory_losses.index_copy(
                0, reference_indices, paired_losses
            )
        trajectory_count = kwargs.get("num_items_in_batch")
        loss = (
            trajectory_losses.mean() if trajectory_count is None
            else trajectory_losses.sum() / trajectory_count
        )
        if trajectory_count is not None and self.args.average_tokens_across_devices:
            # DDP averages rank gradients; undo that average when dividing by the global trajectory count.
            loss = loss * self.accelerator.num_processes
        return_outputs = args[0] if args else kwargs.get("return_outputs", False)
        if return_outputs:
            return loss, self._evaluation_predictions(
                logits_without_reference, logits_with_reference, reference_indices
            )
        return loss

    def _evaluation_predictions(
        self,
        logits_without_reference: torch.Tensor,
        logits_with_reference: torch.Tensor,
        reference_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Line up the two predictions per trajectory so evaluation can score them separately.

        Args:
            logits_without_reference: Predictions [B], one for every trajectory.
            logits_with_reference: Predictions [P], only for trajectories with references.
            reference_indices: Positions [P] of those trajectories in the full batch.

        Returns:
            dict[str, Tensor]: Two logit arrays [B] and a reference-availability mask. Missing reference predictions are zero placeholders ignored by metrics.
        """
        aligned_reference_logits = torch.zeros_like(
            logits_without_reference
        ).index_copy(0, reference_indices, logits_with_reference)
        reference_available = torch.zeros_like(
            logits_without_reference, dtype=torch.bool
        )
        reference_available[reference_indices] = True
        return {
            "logits": logits_without_reference,
            "privileged_logits": aligned_reference_logits,
            "privileged_available": reference_available,
        }

    def _save(self, *args: Any, **kwargs: Any) -> None:
        """Save trainable critic weights and settings; Trainer saves optimizer state separately.

        Args:
            *args: Trainer may pass the output directory first.
            **kwargs: Trainer may pass output_dir; otherwise use the configured training directory. Other save arguments are unused.

        Returns:
            None. Writes critic weights, critic configuration, and training_args.json.
        """
        output_dir = args[0] if args else kwargs.get("output_dir", self.args.output_dir)
        directory = Path(output_dir)
        cast(BaseCritic, self.accelerator.unwrap_model(self.model)).save(directory)
        (directory / TRAINING_ARGS_FILENAME).write_text(
            self.args.to_json_string() + "\n"
        )

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, *args: Any, **kwargs: Any
    ) -> None:
        """Restore critic weights after checking that training can resume from this checkpoint.

        Args:
            resume_from_checkpoint: Directory containing critic weights, optimizer, scheduler, and random state.
            *args: Additional Trainer checkpoint arguments; unused.
            **kwargs: Additional Trainer checkpoint arguments; unused.

        Returns:
            None. Trainer restores optimizer, scheduler, and random state after this method.
        """
        directory = Path(resume_from_checkpoint)
        rng = (
            f"rng_state_{self.args.process_index}.pth"
            if self.args.world_size > 1
            else "rng_state.pth"
        )
        for filename in (OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, rng):
            if not (directory / filename).is_file():
                raise FileNotFoundError(directory / filename)
        cast(BaseCritic, self.model).load_from_weights(directory)

    def _load_best_model(self) -> None:
        """Restore the checkpoint with the best unaided validation NLL without reference solutions.

        Returns:
            None. Replaces the current trainable weights with the selected checkpoint.
        """
        if self.state.best_model_checkpoint is None:
            raise ValueError("no best critic checkpoint was selected")
        cast(BaseCritic, self.model).load_from_weights(
            Path(self.state.best_model_checkpoint)
        )
