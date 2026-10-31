"""Exercise setup, hidden-state caching, frozen/LoRA training, and evaluation without network or GPU dependencies."""

import json
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import Mock

import pyarrow as pa
import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from critic_only.cli import cache_hidden_states, evaluate, setup, train
from critic_only.models.models import CriticConfig
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.storage import file_digest, write_json_atomic, write_parquet_atomic


@pytest.fixture
def temporary_dataset_lifetimes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require temporary Arrow datasets to close before their cache directory is removed."""
    references: list[weakref.ReferenceType[Dataset]] = []
    generate, map_dataset = Dataset.from_generator, Dataset.map

    def track_generated(*args: Any, **kwargs: Any) -> Dataset:
        """Observe the source dataset without keeping its memory maps alive."""
        dataset = generate(*args, **kwargs)
        references.append(weakref.ref(dataset))
        return dataset

    def track_mapped(dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        """Observe the tokenized dataset without retaining it."""
        encoded = map_dataset(dataset, *args, **kwargs)
        references.append(weakref.ref(encoded))
        return encoded

    @contextmanager
    def checked_directory(*args: Any, **kwargs: Any) -> Iterator[str]:
        """Check resource lifetimes before cleanup, including on filesystems that allow open-file deletion."""
        with TemporaryDirectory(*args, **kwargs) as directory:
            yield directory
            assert references and all(reference() is None for reference in references)

    monkeypatch.setattr(Dataset, "from_generator", track_generated)
    monkeypatch.setattr(Dataset, "map", track_mapped)
    monkeypatch.setattr(setup, "TemporaryDirectory", checked_directory)


def save_features(root: Path, settings: CriticConfig, rows: list[dict[str, Any]]) -> None:
    """Publish a small self-contained feature collection with matching checksums and extraction identity."""
    checksums = {}
    for split in ("train", "validation", "test"):
        values = [{**row, "problem_key": f"{split}-{index}", "prefix_id": f"{split}-{index}-{oracle}", "oracle": oracle}
                  for index, row in enumerate(rows) for oracle in ((False, True) if split == "train" else (False,))]
        folder = "hidden_states" if settings.training_mode == "frozen" else "tokenized"
        path = root / split / "critic_only" / folder / "part-00000.parquet"
        write_parquet_atomic(path, pa.Table.from_pylist(values))
        checksums[str(path.relative_to(root))] = file_digest(path)
    tokenization = {"settings": {"model": settings.model, "revision": settings.revision,
                                "horizon": settings.horizon, "hidden_size": settings.input_size},
                    "complete": True, "shards": checksums if settings.training_mode == "lora" else {}}
    write_json_atomic(root / "manifest.json", {"tokenization": tokenization,
        "features": {"complete": True, "signature": settings.data_signature, "shards": checksums,
                     "settings": {"hidden_size": settings.input_size}}})


@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_train_and_evaluate_cli(variant: str, fraction: float, mode: str, critic_settings: Callable[[str], CriticConfig],
    training_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerFast, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """All variants train from features only, preserve paired losses, and export final-outcome predictions."""
    settings = critic_settings(variant).model_copy(update={"training_mode": mode,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    root, output, report = tmp_path / "data", tmp_path / "run", tmp_path / "evaluation"
    save_features(root, settings, training_rows if mode == "frozen" else token_rows)
    tokenizer.save_pretrained(root / "critic_only" / "tokenizer")
    trainer_factory = Mock(wraps=train.BaseTrainer)
    monkeypatch.setattr(train, "BaseTrainer", trainer_factory)
    feature_factory = Mock(wraps=train.HiddenStateDataset)
    monkeypatch.setattr(train, "HiddenStateDataset", feature_factory)
    monkeypatch.setattr("sys.argv", ["train", "--data", str(root), "--output", str(output), "--variant", variant,
        "--training-mode", mode, "--accumulation-steps", "1", "--hidden-size", "16", "--epochs", "1", "--batch-size", "2", "--training-problem-fraction", str(fraction), "--cpu"])
    train.main()
    training = trainer_factory.call_args.kwargs["train_dataset"]
    selected = feature_factory.call_args_list[0].args[0] if mode == "frozen" else training
    assert len(set(selected["problem_key"])) == int(2 * fraction)
    assert any(selected["oracle"]) == (variant != "ppo-critic")
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
    monkeypatch.setattr(train, "load_critic_splits", loader)
    monkeypatch.setattr("sys.argv", ["train", "--data", "/missing", "--training-problem-fraction", fraction])
    with pytest.raises(SystemExit):
        train.main()
    loader.assert_not_called()


def test_setup_and_extraction_cli(prepared_inputs: CollectionFixture, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch, temporary_dataset_lifetimes: None) -> None:
    """Shared setup does not load Qwen; extraction pins one model and reuses its exact token cache."""
    settings = critic_settings("ppo-critic")
    tokenizer.save_pretrained(settings.model)
    root = prepared_inputs.destination.parent / "processed"
    api = Mock(); api.model_info.return_value.sha = "local"
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(setup, "TRAIN_PROBLEMS", 1)
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": settings.model})
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-directory", str(prepared_inputs.outputs_directory),
        "--datasets-directory", str(prepared_inputs.datasets_directory), "--destination", str(root), "--solver-model", "tiny", "--horizon", "3", "--max-length", "256"])
    setup.main()
    assert (root / "splits.parquet").is_file()
    assert (root / "tiny" / "train" / "critic_only" / "tokenized").is_dir()
    assert not (root / "tiny" / "train" / "critic_only" / "hidden_states").exists()
    monkeypatch.setattr("sys.argv", ["cache_hidden_states", "--data", str(root / "tiny"), "--cpu"])
    cache_hidden_states.main()
    manifest = json.loads((root / "tiny" / "manifest.json").read_text())
    assert manifest["features"]["complete"] and manifest["tokenization"]["complete"]
    cache_hidden_states.main()
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
