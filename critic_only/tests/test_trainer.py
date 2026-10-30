"""Exercise HF training, best-model selection, and exact interrupted optimizer/RNG recovery."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_pt_utils import find_batch_size

from critic_only.data.critic_batcher import CriticBatcher
from critic_only.eval.metrics import compute_metrics
from critic_only.models.base_critic import BaseCritic
from critic_only.models.models import CriticConfig
from critic_only.models.loader import create_critic
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
    model = create_critic(settings, True)
    data = Dataset.from_list(rows)
    args = TrainingArguments(output_dir=str(directory), max_steps=4, per_device_train_batch_size=1,
                             per_device_eval_batch_size=1, gradient_accumulation_steps=2, learning_rate=1e-3,
                             use_cpu=True, optim="adamw_torch", eval_strategy="steps", eval_steps=2,
                             save_strategy="steps", save_steps=2, load_best_model_at_end=True,
                             logging_steps=1, disable_tqdm=True,
                             metric_for_best_model="terminal_nll", greater_is_better=False, seed=17,
                             remove_unused_columns=False, label_names=["labels"], report_to="none")
    return BaseTrainer(model=model, args=args, data_collator=CriticBatcher(tokenizer if settings.training_mode == "lora" else None), train_dataset=data,
                       eval_dataset=data, compute_metrics=partial(compute_metrics, problem_keys=list(data["problem_key"])),
                       callbacks=callbacks)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_training_resume(mode: str, critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
                         training_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Resuming an interrupted run must match uninterrupted training under the same schedule."""
    settings = critic_settings("de-critic").model_copy(update={"training_mode": mode,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    training_rows = token_rows if mode == "lora" else training_rows
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
    trainer = make_trainer(tmp_path / "run", critic_settings("de-critic"), tokenizer, training_rows, [])
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


@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic"])
@pytest.mark.parametrize("references", [(False, False), (True, True), (False, True)])
@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_paired_loss_and_gradients(variant: str, references: tuple[bool, bool], mode: str,
                                 critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
                                 training_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]],
                                 tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Match independent per-example losses and gradients, including rows with missing references."""
    settings = critic_settings(variant).model_copy(update={"training_mode": mode,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    column = "input_ids" if mode == "lora" else "hidden_state"
    reference = [2, 3, 2, 3] if mode == "lora" else [0.25] * 16
    source = token_rows if mode == "lora" else training_rows
    rows = [{**row, "oracle": variant != "ppo-critic" and index == 1,
             f"privileged_{column}": reference if present else None}
            for index, (row, present) in enumerate(zip(source, references))]
    trainer = make_trainer(tmp_path / "run", settings, tokenizer, rows, [])
    model = cast(BaseCritic, trainer.model)
    model.eval()
    forward = Mock(wraps=model.forward)
    monkeypatch.setattr(model, "forward", forward)
    batcher = CriticBatcher(tokenizer if mode == "lora" else None)
    batch = batcher(rows)
    assert find_batch_size(batch) == len(rows)
    actual, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    assert forward.call_count == 1
    expected_losses = []
    for row in rows:
        labels = torch.tensor([row["labels"]], dtype=torch.float32)
        views = [row[column]]
        if row[f"privileged_{column}"] is not None:
            views.append(row[f"privileged_{column}"])
        losses = []
        for tokens in views:
            inputs = batcher([{**row, column: tokens, f"privileged_{column}": None}])["inputs"]
            prediction = model(inputs=inputs, oracle=torch.tensor([row["oracle"]]),
                               remaining_blocks=torch.tensor([row["remaining_blocks"]]))
            losses.append(model.loss(prediction, labels, torch.tensor([row["oracle"]])).squeeze(0))
        expected_losses.append(torch.stack(losses).mean())
    expected = torch.stack(expected_losses).mean()
    torch.testing.assert_close(actual, expected)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for actual_gradient, expected_gradient in zip(torch.autograd.grad(actual, parameters), torch.autograd.grad(expected, parameters)):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-4)
    assert outputs["privileged_available"].tolist() == list(references)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_accumulation_matches_full_batch(mode: str, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, training_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Uneven microbatches must give each trajectory equal weight in an optimizer update."""
    settings = critic_settings("de-critic").model_copy(update={"training_mode": mode,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    rows = token_rows if mode == "lora" else training_rows
    dataset = Dataset.from_list([rows[0], rows[1], {**rows[0], "labels": 1, "oracle": True}])
    updates = []
    for batch_size, accumulation in ((3, 1), (2, 2), (1, 3)):
        set_seed(17)
        model = create_critic(settings, True)
        args = TrainingArguments(output_dir=str(tmp_path / f"batch-{batch_size}"), max_steps=1,
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
            learning_rate=0.1, optim="sgd", max_grad_norm=0, weight_decay=0,
            use_cpu=True, save_strategy="no", logging_strategy="no", disable_tqdm=True,
            remove_unused_columns=False, label_names=["labels"], report_to="none", seed=17)
        trainer = BaseTrainer(model=model, args=args, train_dataset=dataset,
            data_collator=CriticBatcher(tokenizer if mode == "lora" else None))
        trainer.train()
        updates.append({name: parameter.detach().clone() for name, parameter in model.named_parameters()
                        if parameter.requires_grad})
    for accumulated in updates[1:]:
        for name, expected in updates[0].items():
            torch.testing.assert_close(accumulated[name], expected, atol=1e-6, rtol=1e-5, msg=name)
