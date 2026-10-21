"""Run the actual training/evaluation entry points against a tiny cached dataset."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerFast

from critic_only.cli import evaluate, setup, train
from critic_only.constants import MANIFEST_FILENAME, TRAINING_ARGS_FILENAME
from critic_only.data.models import PreparationConfig
from critic_only.models.models import CriticConfig


@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_train_and_evaluate_cli(variant: str, fraction: float, critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
                                training_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: Any,
                                capsys: pytest.CaptureFixture[str]) -> None:
    """Use the public CPU smoke-test path, select best on validation, and write terminal test metrics."""
    settings = critic_settings(variant)
    trainer_factory = Mock(wraps=train.BaseTrainer)
    monkeypatch.setattr(train, "BaseTrainer", trainer_factory)
    cache, output, report = tmp_path / "cache", tmp_path / "training", tmp_path / "evaluation"
    splits = {}
    for split in ("train", "validation", "test"):
        rows = training_rows + [{**training_rows[0], "oracle": True}]
        splits[split] = Dataset.from_list([{**row, "prefix_id": f"{split}-{index}", "problem_key": f"{split}-{index}"}
                                          for index, row in enumerate(rows)])
    splits["train"] = Dataset.from_list([
        {**row, "oracle": oracle, "problem_key": f"train-{index}", "prefix_id": f"train-{index}-{oracle}"}
        for index, row in enumerate(training_rows) for oracle in (False, True)
    ])
    DatasetDict(splits).save_to_disk(str(cache))
    tokenizer.save_pretrained(cache / "tokenizer")
    manifest = {"settings": {"model": settings.model, "revision": settings.revision,
                             "oracle_supervision_fraction": 1, "max_length": 256}, "preparation": {"settings": {"horizon": 3}}}
    (cache / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["train", "--data", str(cache), "--output", str(output), "--variant", variant,
                                     "--rank", "2", "--lora-alpha", "4", "--epochs", "1", "--accumulation-steps", "1", "--cpu",
                                     *(["--training-problem-fraction", str(fraction)] if fraction != 1 else [])])
    train.main()
    assert list(trainer_factory.call_args.kwargs["train_dataset"]["oracle"]) == (
        [False] * int(2 * fraction) if variant == "ppo-critic" else [False, True] * int(2 * fraction))
    assert list(trainer_factory.call_args.kwargs["eval_dataset"]["oracle"]) == [False, False]
    assert json.loads((output / "training.json").read_text())["options"]["training_problem_fraction"] == fraction
    state = json.loads((output / "trainer_state.json").read_text())
    loss_record = next(row for row in state["log_history"] if "loss" in row)
    assert {"step", "epoch", "learning_rate", "grad_norm"} <= loss_record.keys()
    assert any("eval_terminal_nll" in row and "eval_terminal_brier" in row for row in state["log_history"])
    assert any("train_runtime" in row and "train_samples_per_second" in row for row in state["log_history"])
    captured = capsys.readouterr()
    assert '"learning_rate"' in captured.out and "it/s" not in captured.err
    checkpoint = output / "best"
    assert (checkpoint / "critic.safetensors").is_file()
    assert json.loads((checkpoint / TRAINING_ARGS_FILENAME).read_text())["disable_tqdm"]
    monkeypatch.setattr("sys.argv", ["evaluate", "--data", str(cache), "--checkpoint", str(checkpoint),
                                     "--destination", str(report), "--split", "test", "--cpu"])
    evaluate.main()
    metrics = json.loads((report / "metrics.json").read_text())
    assert metrics["without_reference"]["problems"] == 2 and metrics["without_reference"]["prefixes"] == 2
    assert metrics["with_reference"]["prefixes"] == 1
    assert 0 <= metrics["without_reference"]["terminal_brier"] <= 1
    assert (report / "predictions.parquet").is_file()
    predictions = Dataset.from_parquet(str(report / "predictions.parquet"))
    assert predictions["final_correctness"] == [row["labels"] for row in training_rows]
    assert "counts" not in predictions.column_names
    assert predictions[0]["privileged_probability"] is not None
    assert predictions[1]["privileged_probability"] is None


@pytest.mark.parametrize("fraction", ["0", "-0.5", "1.1", "nan", "inf"])
def test_invalid_training_fraction_fails_before_loading(fraction: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject invalid fractions before reading caches or initializing models."""
    load = Mock()
    monkeypatch.setattr(train, "load_from_disk", load)
    monkeypatch.setattr("sys.argv", ["train", "--training-problem-fraction", fraction])
    with pytest.raises(SystemExit):
        train.main()
    load.assert_not_called()


def test_hf_download_selects_only_needed_snapshots(prepared_inputs: PreparationConfig, monkeypatch: Any) -> None:
    """Pin downloads and request conversations for input phases but no later checkpoint files."""
    api = Mock()
    api.dataset_info.return_value.sha = "pinned-commit"
    download = Mock(return_value=str(prepared_inputs.outputs_directory))
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(setup, "snapshot_download", download)
    root, revision = setup.download_outputs("owner/outputs", "main", "tiny", 3)
    assert root == prepared_inputs.outputs_directory and revision == "pinned-commit"
    assert download.call_count == 2
    selected = download.call_args_list[-1].kwargs["allow_patterns"]
    assert all("*" not in path for path in selected)
    assert any("continue-oracle" in path for path in selected if path.endswith("conversation.json.zst"))
    assert any("continue-at" in path for path in selected if path.endswith("conversation.json.zst"))
    assert not any(path.endswith("checkpoint_3x.json.zst") for path in selected)
    assert all(call.kwargs["revision"] == "pinned-commit" for call in download.call_args_list)


@pytest.mark.parametrize("max_length", [1, 256])
def test_setup_cli(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast,
                       max_length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Produce both caches in one command; failed tokenization must publish neither."""
    api = Mock()
    api.model_info.return_value.sha = "model-commit"
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    tokenizer_factory = Mock(from_pretrained=Mock(return_value=tokenizer))
    monkeypatch.setattr(setup, "AutoTokenizer", tokenizer_factory)
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": "owner/tiny"})
    destination = prepared_inputs.destination.parent / "processed"
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-directory", str(prepared_inputs.outputs_directory),
                                     "--datasets-directory", str(prepared_inputs.datasets_directory),
                                     "--destination", str(destination), "--solver-model", "tiny", "--horizon", "3",
                                     "--max-length", str(max_length)])
    if max_length == 1:
        with pytest.raises(ValueError, match="truncation"):
            setup.main()
        assert not destination.exists()
    else:
        setup.main()
        assert (destination / "prepared" / "test.parquet").is_file()
        cache = load_from_disk(str(destination / "tokenized"))
        assert cache["test"][0]["labels"] == 0
        assert len(cache["test"]) == 5
        manifest = json.loads((destination / "tokenized" / MANIFEST_FILENAME).read_text())
        assert manifest["settings"]["model"] == "owner/tiny"
        assert manifest["settings"]["revision"] == "model-commit"
        assert manifest["preparation"]["settings"]["source_revision"] is None
        api.model_info.assert_called_once_with("owner/tiny")
        tokenizer_factory.from_pretrained.assert_called_once_with("owner/tiny", revision="model-commit", padding_side="right")
        with pytest.raises(FileExistsError):
            setup.main()


@pytest.mark.parametrize("options", [["--horizon", "1"], ["--validation-fraction", "1"], ["--seed", "-1"],
                                    ["--revision", "local"]])
def test_setup_invalid_options_fail_before_loading(options: list[str], prepared_inputs: PreparationConfig,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Local collection options fail before any model metadata request or tokenizer initialization."""
    hub, tokenizer = Mock(), Mock()
    monkeypatch.setattr(setup, "HfApi", hub)
    monkeypatch.setattr(setup, "AutoTokenizer", tokenizer)
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": "owner/tiny"})
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-directory", str(prepared_inputs.outputs_directory),
                                     "--datasets-directory", str(prepared_inputs.datasets_directory),
                                     "--solver-model", "tiny", *options])
    with pytest.raises(SystemExit):
        setup.main()
    hub.assert_not_called()
    tokenizer.from_pretrained.assert_not_called()


def test_setup_requires_remote_revision(prepared_inputs: PreparationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote collections still require an explicit revision before any download."""
    download = Mock()
    monkeypatch.setattr(setup, "download_outputs", download)
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": "owner/tiny"})
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-repo", "owner/outputs", "--solver-model", "tiny",
                                     "--datasets-directory", str(prepared_inputs.datasets_directory)])
    with pytest.raises(SystemExit):
        setup.main()
    download.assert_not_called()


def test_setup_bad_audit_fails_before_loading(prepared_inputs: PreparationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate local outcomes before resolving the model or loading tokenizer files."""
    path = next(prepared_inputs.outputs_directory.rglob("audit.json"))
    audit = json.loads(path.read_text())
    audit["complete"] = False
    path.write_text(json.dumps(audit))
    hub, tokenizer = Mock(), Mock()
    monkeypatch.setattr(setup, "HfApi", hub)
    monkeypatch.setattr(setup, "AutoTokenizer", tokenizer)
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": "owner/tiny"})
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-directory", str(prepared_inputs.outputs_directory),
                                     "--datasets-directory", str(prepared_inputs.datasets_directory), "--solver-model", "tiny",
                                     "--horizon", "3", "--destination", str(prepared_inputs.destination)])
    with pytest.raises(ValueError, match="incomplete branch audit"):
        setup.main()
    assert not prepared_inputs.destination.exists()
    hub.assert_not_called()
    tokenizer.from_pretrained.assert_not_called()


@pytest.mark.parametrize("failure", ["output", "context", "resume"])
def test_training_checks_before_loading(failure: str, critic_settings: Callable[[str], CriticConfig],
                                        training_rows: list[dict[str, Any]], tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject occupied outputs, changed resume settings, and overlong contexts without loading model weights."""
    settings = critic_settings("ppo-critic")
    cache, output = tmp_path / "cache", tmp_path / "output"
    cache.mkdir()
    manifest = {"settings": {"model": settings.model, "revision": settings.revision, "max_length": 257},
                "preparation": {"settings": {"horizon": settings.horizon}}}
    (cache / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    dataset = DatasetDict({split: Dataset.from_list(training_rows) for split in ("train", "validation")})
    monkeypatch.setattr(train, "load_from_disk", Mock(return_value=dataset))
    model, tokenizer = Mock(), Mock()
    monkeypatch.setattr(train, "create_critic_from_pretrained", model)
    monkeypatch.setattr(train, "AutoTokenizer", tokenizer)
    arguments = ["train", "--data", str(cache), "--output", str(output), "--variant", "ppo-critic", "--cpu"]
    if failure in {"output", "resume"}:
        output.mkdir()
        (output / "training.json").write_text("{}")
    if failure == "resume":
        checkpoint = output / "checkpoint-1"
        checkpoint.mkdir()
        arguments += ["--resume", str(checkpoint)]
    monkeypatch.setattr("sys.argv", arguments)
    error = FileExistsError if failure == "output" else ValueError
    message = {"output": "output is not empty", "context": "context limit", "resume": "settings or cached data differ"}[failure]
    with pytest.raises(error, match=message):
        train.main()
    model.assert_not_called()
    tokenizer.from_pretrained.assert_not_called()
