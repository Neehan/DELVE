"""Train future checkpoint curves at every solver token and evaluate at the 1× endpoint."""

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import Trainer
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.trainer_callback import PrinterCallback

from critic_only.constants import TRAINING_ARGS_FILENAME
from critic_only.models.base_critic import BaseCritic


class BaseTrainer(Trainer):
    """Add per-trajectory Monte Carlo loss, endpoint evaluation, logging and checkpoints to Hugging Face Trainer.

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
        # Request the total label count so uneven microbatches retain equal per-prefix weight.
        self.model_accepts_loss_kwargs = True
        self.args.disable_tqdm = True
        # Our log() prints metrics as JSON; disable HF's printer to avoid duplicate console logs.
        self.remove_callback(PrinterCallback)

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
        """Fit the exact future-checkpoint curve at every solver token, weighting trajectories equally.

        Args:
            model: Critic, possibly wrapped in DDP.
            inputs: Tokens, solver masks, checkpoint labels [B,H-1], oracle flags and aligned budgets.
            *args: Hugging Face may pass return_outputs as the first positional argument.
            **kwargs: num_items_in_batch counts all checkpoint labels across the accumulated batch;
                return_outputs requests endpoint curves for evaluation.

        Returns:
            Tensor: BCE averaged over horizons and solver tokens, then trajectories. With return_outputs,
            also returns endpoint logits [B,H-1]. Best-checkpoint selection still uses final-checkpoint NLL.
        """
        logits = model(inputs=inputs["inputs"], oracle=inputs["oracle"], remaining_blocks=inputs["remaining_blocks"])
        critic = cast(BaseCritic, self.accelerator.unwrap_model(model))
        checkpoint_losses = critic.loss(logits, inputs["labels"], inputs["oracle"])
        trajectory_losses = self._mean_trajectory_losses(checkpoint_losses, inputs["inputs"])
        loss = self._normalize_batch_loss(
            trajectory_losses, inputs["labels"].shape[-1], kwargs.get("num_items_in_batch")
        )
        return_outputs = args[0] if args else kwargs.get("return_outputs", False)
        if not return_outputs:
            return loss
        # Endpoint metrics ask: given the complete 1× prefix, how accurately do we predict 2×–5× outcomes?
        # Earlier solver tokens provide additional loss supervision; best-checkpoint selection uses endpoint final-outcome NLL.
        endpoint_positions = inputs["inputs"]["attention_mask"].sum(dim=-1) - 1
        batch_indices = torch.arange(len(endpoint_positions), device=logits.device)
        logits = logits[batch_indices, endpoint_positions]
        return loss, {"logits": logits}

    def _mean_trajectory_losses(
        self, checkpoint_losses: Tensor, model_inputs: dict[str, Tensor]
    ) -> Tensor:
        """Average checkpoints and solver tokens so each trajectory has one loss.

        Args:
            checkpoint_losses: Unreduced BCE [B,L,C] at every token and checkpoint.
            model_inputs: Inputs include solver_mask [B,L], excluding instructions and padding.

        Returns:
            Tensor: Losses [B], with equal weight per trajectory regardless of its token count.
        """
        # First average checkpoint targets: [B,L,C] becomes [B,L].
        checkpoint_mean = checkpoint_losses.mean(dim=-1)
        solver_mask = model_inputs["solver_mask"]
        return (checkpoint_mean * solver_mask).sum(dim=-1) / solver_mask.sum(dim=-1)

    def _normalize_batch_loss(
        self, trajectory_losses: Tensor, num_checkpoints: int, checkpoint_label_count: Tensor | int | None
    ) -> Tensor:
        """Scale this microbatch's loss for gradient accumulation and distributed training.

        Args:
            trajectory_losses: One mean loss per trajectory in this microbatch.
            num_checkpoints: Number of labels per trajectory.
            checkpoint_label_count: HF's total label count across accumulated microbatches, or None for evaluation.

        Returns:
            Tensor: Scalar loss whose accumulated gradients weight every trajectory equally.
        """
        if checkpoint_label_count is None:
            return trajectory_losses.mean()
        # HF counts every entry of [B,C]; divide by C to count trajectories, including uneven microbatches.
        trajectory_count = checkpoint_label_count / num_checkpoints
        loss = trajectory_losses.sum() / trajectory_count
        if self.args.average_tokens_across_devices:
            # DDP averages rank gradients; undo that average when dividing by the global trajectory count.
            loss = loss * self.accelerator.num_processes
        return loss

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
        """Restore the checkpoint with the best reference-conditioned NLL on unaided validation prefixes.

        Returns:
            None. Replaces the current trainable weights with the selected checkpoint.
        """
        if self.state.best_model_checkpoint is None:
            raise ValueError("no best critic checkpoint was selected")
        cast(BaseCritic, self.model).load_from_weights(
            Path(self.state.best_model_checkpoint)
        )
