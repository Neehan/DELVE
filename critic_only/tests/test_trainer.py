"""Exercise HF training, best-model selection, and exact interrupted optimizer/RNG recovery."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, TrainerCallback, TrainingArguments, set_seed

from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.eval.metrics import compute_metrics
from critic_only.models.de_critic import DECritic
from critic_only.models.models import CriticConfig
from critic_only.trainer.base_trainer import BaseTrainer


class InterruptTraining(TrainerCallback):
    """Stop after a saved step while preserving the original learning-rate schedule."""

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Request a checkpoint before simulating an interruption."""
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True
        return control


def make_trainer(directory: Path, settings: CriticConfig, tokenizer: PreTrainedTokenizerFast,
                 rows: list[dict[str, Any]], callbacks: list[TrainerCallback]) -> BaseTrainer:
    """Use real Trainer accumulation, evaluation, and checkpoint selection on a tiny CPU model."""
    set_seed(17)
    model = DECritic.create_from_pretrained(settings, torch.float32)
    data = Dataset.from_list(rows)
    args = TrainingArguments(output_dir=str(directory), max_steps=4, per_device_train_batch_size=1,
                             per_device_eval_batch_size=1, gradient_accumulation_steps=2, learning_rate=1e-3,
                             use_cpu=True, optim="adamw_torch", eval_strategy="steps", eval_steps=2,
                             save_strategy="steps", save_steps=2, load_best_model_at_end=True,
                             logging_steps=1, disable_tqdm=True,
                             metric_for_best_model="terminal_nll", greater_is_better=False, seed=17,
                             remove_unused_columns=False, label_names=["labels"], report_to="none")
    return BaseTrainer(model=model, args=args, data_collator=PrefixTokenizer(tokenizer).pad_batch, train_dataset=data,
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
    history = json.loads((tmp_path / "resumed" / "trainer_state.json").read_text())["log_history"]
    assert [row["step"] for row in history if "loss" in row] == [1, 2, 3, 4]
    assert full.model is not None and resumed.model is not None
    full_weights = dict(full.model.named_parameters())
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(parameter, full_weights[name], rtol=0, atol=0)
    resumed.save_model(str(tmp_path / "best"))
    assert (tmp_path / "best" / "critic.safetensors").is_file()
    assert "eval_terminal_nll" in resumed.evaluate()


def test_metrics_persist_before_checkpoint(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
                                          training_rows: list[dict[str, Any]], tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Persist each log before any checkpoint exists, with no duplicate writes from other ranks."""
    trainer = make_trainer(tmp_path / "run", critic_settings("de"), tokenizer, training_rows, [])
    trainer.log({"loss": 0.5, "learning_rate": 0.001})
    path = tmp_path / "run" / "trainer_state.json"
    saved = path.read_bytes()
    assert json.loads(saved)["log_history"][-1]["loss"] == 0.5
    assert not list(path.parent.glob("checkpoint-*"))
    assert '"loss": 0.5' in capsys.readouterr().out
    monkeypatch.setattr(trainer, "is_world_process_zero", lambda: False)
    trainer.log({"loss": 0.4})
    assert path.read_bytes() == saved
    assert capsys.readouterr().out == ""
