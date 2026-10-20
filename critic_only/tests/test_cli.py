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


def test_train_and_evaluate_cli(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
                                training_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: Any,
                                capsys: pytest.CaptureFixture[str]) -> None:
    """Use the public CPU smoke-test path, select best on validation, and write terminal test metrics."""
    settings = critic_settings("oracle-supervised")
    cache, output, report = tmp_path / "cache", tmp_path / "training", tmp_path / "evaluation"
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = Dataset.from_list([{**row, "prefix_id": f"{split}-{index}", "problem_key": f"{split}-{index}"}
                                          for index, row in enumerate(training_rows)])
    DatasetDict(splits).save_to_disk(str(cache))
    tokenizer.save_pretrained(cache / "tokenizer")
    manifest = {"settings": {"backbone": settings.backbone, "revision": settings.revision, "reference_mode": "none",
                             "oracle_supervision_fraction": 1, "max_length": 256}, "preparation": {"settings": {"horizon": 3}}}
    (cache / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    monkeypatch.setattr("sys.argv", ["train", "--data", str(cache), "--output", str(output), "--variant", "oracle-supervised",
                                     "--rank", "2", "--lora-alpha", "4", "--epochs", "1", "--accumulation-steps", "1", "--cpu"])
    train.main()
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
    assert metrics["problems"] == 2 and metrics["prefixes"] == 2
    assert 0 <= metrics["terminal_brier"] <= 1
    assert (report / "predictions.parquet").is_file()


def test_hf_download_selects_only_needed_snapshots(prepared_inputs: PreparationConfig, monkeypatch: Any) -> None:
    """Pin both download stages to one commit and avoid fetching every full branch conversation."""
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
    assert all("unaided" in path for path in selected if path.endswith("conversation.json.zst"))
    assert all(call.kwargs["revision"] == "pinned-commit" for call in download.call_args_list)


@pytest.mark.parametrize("max_length", [1, 256])
def test_setup_cli(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast,
                       max_length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Produce both caches in one command; failed tokenization must publish neither."""
    api = Mock()
    api.model_info.return_value.sha = "model-commit"
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(setup, "AutoTokenizer", Mock(from_pretrained=Mock(return_value=tokenizer)))
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", ("tiny",))
    destination = prepared_inputs.destination.parent / "processed"
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-directory", str(prepared_inputs.outputs_directory),
                                     "--datasets-directory", str(prepared_inputs.datasets_directory),
                                     "--destination", str(destination), "--solver-model", "tiny", "--horizon", "3",
                                     "--revision", "local", "--backbone", "tiny", "--backbone-revision", "main",
                                     "--max-length", str(max_length)])
    if max_length == 1:
        with pytest.raises(ValueError, match="truncation"):
            setup.main()
        assert not destination.exists()
    else:
        setup.main()
        assert (destination / "prepared" / "test.parquet").is_file()
        cache = load_from_disk(str(destination / "tokenized"))
        assert cache["test"][0]["labels"] == [[[2, 2], [1, 2]], [[2, 2], [1, 2]]]
        manifest = json.loads((destination / "tokenized" / MANIFEST_FILENAME).read_text())
        assert manifest["settings"]["revision"] == "model-commit"
        with pytest.raises(FileExistsError):
            setup.main()
