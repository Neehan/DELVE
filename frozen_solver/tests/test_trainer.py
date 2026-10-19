"""Exercise HF training, best-model selection, and exact interrupted optimizer/RNG recovery."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, TrainerCallback, TrainingArguments, set_seed

from frozen_solver.data.collator import PrefixCollator
from frozen_solver.eval.metrics import compute_metrics
from frozen_solver.models.checkpoint import build_critic
from frozen_solver.models.models import CriticConfig
from frozen_solver.trainer.critic_trainer import CriticTrainer


class InterruptTraining(TrainerCallback):
    """Stop after a saved step while preserving the original learning-rate schedule."""

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Request a checkpoint before simulating an interruption."""
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True
        return control


def make_trainer(directory: Path, settings: CriticConfig, tokenizer: PreTrainedTokenizerFast,
                 rows: list[dict[str, Any]], callbacks: list[TrainerCallback]) -> CriticTrainer:
    """Use real Trainer accumulation, evaluation, and checkpoint selection on a tiny CPU model."""
    set_seed(17)
    model = build_critic(settings, torch.float32)
    data = Dataset.from_list(rows)
    args = TrainingArguments(output_dir=str(directory), max_steps=4, per_device_train_batch_size=1,
                             per_device_eval_batch_size=1, gradient_accumulation_steps=2, learning_rate=1e-3,
                             use_cpu=True, optim="adamw_torch", eval_strategy="steps", eval_steps=2,
                             save_strategy="steps", save_steps=2, load_best_model_at_end=True,
                             metric_for_best_model="terminal_nll", greater_is_better=False, seed=17,
                             remove_unused_columns=False, label_names=["labels"], report_to="none")
    return CriticTrainer(model=model, args=args, data_collator=PrefixCollator(tokenizer), train_dataset=data,
                          eval_dataset=data, compute_metrics=compute_metrics, callbacks=callbacks)


def test_training_resume(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
                         training_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Resuming an interrupted run must match uninterrupted training under the same schedule."""
    settings = critic_settings("de")
    full = make_trainer(tmp_path / "full", settings, tokenizer, training_rows, [])
    full.train()
    interrupted = make_trainer(tmp_path / "resumed", settings, tokenizer, training_rows, [InterruptTraining()])
    interrupted.train()
    assert interrupted.state.global_step == 2
    resumed = make_trainer(tmp_path / "resumed", settings, tokenizer, training_rows, [])
    resumed.train(resume_from_checkpoint=str(tmp_path / "resumed" / "checkpoint-2"))
    assert resumed.state.global_step == 4
    assert full.model is not None and resumed.model is not None
    full_weights = dict(full.model.named_parameters())
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(parameter, full_weights[name], rtol=0, atol=0)
    resumed.save_model(str(tmp_path / "best"))
    assert (tmp_path / "best" / "critic.safetensors").is_file()
    assert "eval_terminal_nll" in resumed.evaluate()
