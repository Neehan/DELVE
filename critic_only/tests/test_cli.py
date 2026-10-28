"""Exercise setup, extraction, MLP training, and evaluation without network or GPU dependencies."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pyarrow as pa
import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from critic_only.cli import evaluate, extract, setup, train
from critic_only.models.models import CriticConfig
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.storage import file_digest, write_json_atomic, write_parquet_atomic


def save_features(root: Path, settings: CriticConfig, rows: list[dict[str, Any]]) -> None:
    """Publish a small self-contained feature collection with matching checksums and extraction identity."""
    checksums = {}
    for split in ("train", "validation", "test"):
        values = [{**row, "problem_key": f"{split}-{index}", "prefix_id": f"{split}-{index}-{oracle}", "oracle": oracle}
                  for index, row in enumerate(rows) for oracle in ((False, True) if split == "train" else (False,))]
        path = root / "critic_only" / "hidden_states" / split / "part-00000.parquet"
        write_parquet_atomic(path, pa.Table.from_pylist(values))
        checksums[f"{split}/{path.name}"] = file_digest(path)
    write_json_atomic(root / "manifest.json", {
        "tokenization": {"settings": {"model": settings.model, "revision": settings.revision, "horizon": settings.horizon}},
        "features": {"complete": True, "signature": settings.feature_signature, "shards": checksums,
                     "settings": {"hidden_size": settings.input_size}}})


@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_train_and_evaluate_cli(variant: str, fraction: float, critic_settings: Callable[[str], CriticConfig],
    training_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """All variants train from features only, preserve paired losses, and export final-outcome predictions."""
    settings = critic_settings(variant)
    root, output, report = tmp_path / "data", tmp_path / "run", tmp_path / "evaluation"
    save_features(root, settings, training_rows)
    trainer_factory = Mock(wraps=train.BaseTrainer)
    monkeypatch.setattr(train, "BaseTrainer", trainer_factory)
    feature_factory = Mock(wraps=train.FeatureDataset)
    monkeypatch.setattr(train, "FeatureDataset", feature_factory)
    monkeypatch.setattr("sys.argv", ["train", "--data", str(root), "--output", str(output), "--variant", variant,
        "--hidden-size", "16", "--epochs", "1", "--batch-size", "2", "--training-problem-fraction", str(fraction), "--cpu"])
    train.main()
    training = trainer_factory.call_args.kwargs["train_dataset"]
    assert len(set(feature_factory.call_args_list[0].args[0]["problem_key"])) == int(2 * fraction)
    assert any(training.oracle) == (variant != "ppo-critic")
    state = json.loads((output / "trainer_state.json").read_text())
    assert any("eval_terminal_nll" in row for row in state["log_history"])
    assert '"learning_rate"' in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["evaluate", "--data", str(root), "--checkpoint", str(output / "best"),
        "--destination", str(report), "--split", "test", "--cpu"])
    evaluate.main()
    predictions = Dataset.from_parquet(str(report / "predictions.parquet"))
    assert list(predictions["final_correctness"]) == [row["labels"] for row in training_rows]
    assert predictions[0]["privileged_probability"] is not None
    assert predictions[1]["privileged_probability"] is None


@pytest.mark.parametrize("fraction", ["0", "-1", "1.1", "nan"])
def test_bad_training_fraction_fails_early(fraction: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject invalid experiments before loading feature shards."""
    loader = Mock()
    monkeypatch.setattr(train, "load_feature_splits", loader)
    monkeypatch.setattr("sys.argv", ["train", "--data", "/missing", "--training-problem-fraction", fraction])
    with pytest.raises(SystemExit):
        train.main()
    loader.assert_not_called()


def test_setup_and_extraction_cli(prepared_inputs: CollectionFixture, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared setup does not load Qwen; extraction pins one model and reuses its exact token cache."""
    settings = critic_settings("ppo-critic")
    tokenizer.save_pretrained(settings.model)
    root = prepared_inputs.destination.parent / "processed"
    monkeypatch.setattr(setup, "TRAIN_PROBLEMS", 1)
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": settings.model})
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-directory", str(prepared_inputs.outputs_directory),
        "--datasets-directory", str(prepared_inputs.datasets_directory), "--destination", str(root), "--solver-model", "tiny"])
    setup.main()
    assert (root / "splits.parquet").is_file()
    assert not (root / "tiny" / "critic_only").exists()
    api = Mock(); api.model_info.return_value.sha = "local"
    monkeypatch.setattr(extract, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(extract, "SUPPORTED_MODELS", {"tiny": settings.model})
    monkeypatch.setattr("sys.argv", ["extract", "--data", str(root / "tiny"), "--horizon", "3", "--max-length", "256", "--cpu"])
    extract.main()
    manifest = json.loads((root / "tiny" / "manifest.json").read_text())
    assert manifest["features"]["complete"] and manifest["tokenization"]["complete"]
    extract.main()
    api.model_info.assert_called_once()


def test_pinned_hf_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remote preparation downloads only collection artifacts for the selected model at one revision."""
    api = Mock(); api.dataset_info.return_value.sha = "commit"
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(setup, "snapshot_download", download)
    root, revision = setup.download_outputs("owner/outputs", "main", "tiny")
    assert root == tmp_path and revision == "owner/outputs@commit"
    assert download.call_args.kwargs["revision"] == "commit"
    assert all("/tiny/" in pattern for pattern in download.call_args.kwargs["allow_patterns"])
