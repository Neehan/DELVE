"""Exercise setup, head/LoRA/full training, and evaluation without network or GPU dependencies."""

import json
import shutil
import weakref
from fnmatch import fnmatch
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import Mock

import pyarrow as pa
import pytest
import torch
import zstandard
from safetensors.torch import load_file
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from critic_only.data.critic_batcher import CriticBatcher
from critic_only.models.loader import create_critic
from discovery_execution.audit.constants import GRADING_RULE
from critic_only.constants import CRITIC_TARGET
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.processing.constants import CHECKPOINT_INTERVAL_TOKENS
from critic_only.cli import evaluate, setup, train
from critic_only.models.models import CriticConfig
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.storage import file_digest, write_json_atomic, write_parquet_atomic
from discovery_execution.outputs.parquet_exporter import ParquetTrajectoryExporter


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
            assert all(reference() is None for reference in references)

    monkeypatch.setattr(Dataset, "from_generator", track_generated)
    monkeypatch.setattr(Dataset, "map", track_mapped)
    monkeypatch.setattr(setup, "TemporaryDirectory", checked_directory)


def save_token_inputs(root: Path, settings: CriticConfig, rows: list[dict[str, Any]]) -> None:
    """Publish token splits with matching checksums and model settings."""
    checksums = {}
    for split in ("train", "validation", "test"):
        values = [{**row, "problem_key": f"{split}-{index}", "prefix_id": f"{split}-{index}-{oracle}", "oracle": oracle}
                  for index, row in enumerate(rows) for oracle in ((False, True) if split == "train" else (False,))]
        path = root / split / "critic_only" / "tokenized" / "part-00000.parquet"
        write_parquet_atomic(path, pa.Table.from_pylist(values))
        checksums[str(path.relative_to(root))] = file_digest(path)
    tokenization = {"settings": {"target": CRITIC_TARGET, "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS, "model": settings.model, "revision": settings.revision,
                                "horizon": settings.horizon, "hidden_size": settings.input_size},
                    "complete": True, "shards": checksums}
    write_json_atomic(root / "manifest.json", {"tokenization": tokenization})


@pytest.mark.parametrize("variant", ["ppo-critic", "ppo-oracle-critic", "de-critic", "de-residual-critic"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
@pytest.mark.parametrize("mode", ["head", "lora", "full"])
def test_train_and_evaluate_cli(variant: str, fraction: float, mode: str, critic_settings: Callable[[str], CriticConfig],
    reference_token_rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerFast, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """All variants train on solver tokens and export reference-conditioned checkpoint curves."""
    settings = critic_settings(variant).model_copy(update={"training_mode": mode, "trainable_layers": None if mode == "head" else 64,
        "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
    root, output, report = tmp_path / "data", tmp_path / "run", tmp_path / "evaluation"
    save_token_inputs(root, settings, reference_token_rows)
    tokenizer.save_pretrained(root / "critic_only" / "tokenizer")
    trainer_factory = Mock(wraps=train.get_critic_trainer(variant))
    monkeypatch.setattr(train, "get_critic_trainer", Mock(return_value=trainer_factory))
    monkeypatch.setattr("sys.argv", ["train", "--data", str(root), "--output", str(output), "--variant", variant,
        "--training-mode", mode, "--accumulation-steps", "1", "--hidden-size", "16", "--batch-size", "2", "--training-problem-fraction", str(fraction), "--cpu"])
    train.main()
    selected = trainer_factory.call_args.kwargs["train_dataset"]
    assert len(set(selected["problem_key"])) == int(2 * fraction)
    assert any(selected["oracle"]) == (variant != "ppo-critic")
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["epoch"] == 1
    assert json.loads((output / "training.json").read_text())["options"]["epochs"] == 1
    assert any("eval_terminal_nll" in row for row in state["log_history"])
    assert '"learning_rate"' in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["evaluate", "--data", str(root), "--checkpoint", str(output / "best"),
        "--destination", str(report), "--split", "test", "--cpu"])
    evaluate.main()
    predictions = Dataset.from_parquet(str(report / "predictions.parquet"))
    assert list(predictions["checkpoint_correctness"]) == [row["labels"] for row in reference_token_rows]
    assert "checkpoint_probabilities" in predictions.column_names
    assert "privileged_probability" not in predictions.column_names
    metrics = json.loads((report / "metrics.json").read_text())
    assert metrics["reference_solution"] is True
    assert "terminal_nll" in metrics and "without_reference" not in metrics


@pytest.mark.parametrize("fraction", ["0", "-1", "1.1", "nan"])
def test_bad_training_fraction_fails_early(fraction: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject invalid experiments before loading token shards."""
    loader = Mock()
    monkeypatch.setattr(train, "load_critic_splits", loader)
    monkeypatch.setattr("sys.argv", ["train", "--data", "/missing", "--training-problem-fraction", fraction])
    with pytest.raises(SystemExit):
        train.main()
    loader.assert_not_called()


@pytest.mark.parametrize('source_kind', ['local', 'hf'])
def test_setup_cli(prepared_inputs: CollectionFixture, critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch, temporary_dataset_lifetimes: None, source_kind: str) -> None:
    """Setup retains processed/token shards, cleans only owned downloads, and reprocesses without downloading."""
    settings = critic_settings("ppo-critic")
    tokenizer.save_pretrained(settings.model)
    root = prepared_inputs.destination.parent / "processed"
    published = prepared_inputs.destination.parent / "published"
    ParquetTrajectoryExporter(prepared_inputs.outputs_directory, published).export_trajectories(None)
    original_exports = {path: file_digest(path) for path in published.rglob('*.parquet')}
    api = Mock(); api.model_info.return_value.sha = "local"; api.dataset_info.return_value.sha = 'output-commit'
    def download(repository: str, **kwargs: Any) -> str:
        """Populate the requested temporary directory with real exported fixture files."""
        assert repository == 'owner/outputs'
        shutil.copytree(published, kwargs['local_dir'], dirs_exist_ok=True)
        return str(kwargs['local_dir'])
    snapshot = Mock(side_effect=download)
    monkeypatch.setattr(setup, 'snapshot_download', snapshot)
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(setup, "TRAIN_PROBLEMS", 1)
    monkeypatch.setattr(setup, "CHECKPOINT_INTERVAL_TOKENS", OUTPUT_TOKENS_PER_BLOCK)
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": settings.model})
    source_args = ['--outputs-directory', str(published)] if source_kind == 'local' else ['--outputs-repo', 'owner/outputs', '--revision', 'main']
    monkeypatch.setattr("sys.argv", ["setup", *source_args,
        "--datasets-directory", str(prepared_inputs.datasets_directory), "--destination", str(root), "--solver-model", "tiny", "--horizon", "3", "--max-length", "256"])
    setup.main()
    assert (root / "splits.parquet").is_file()
    assert (root / "tiny" / "train" / "critic_only" / "tokenized").is_dir()
    assert not (root / "tiny" / "train" / "critic_only" / "hidden_states").exists()
    manifest = json.loads((root / "tiny" / "manifest.json").read_text())
    assert manifest["tokenization"]["complete"]
    assert manifest['source_revision'] == ('owner/outputs@output-commit' if source_kind == 'hf' else None)
    assert snapshot.call_count == int(source_kind == 'hf')
    if source_kind == 'hf':
        assert not snapshot.call_args.kwargs['local_dir'].exists()
    assert all(file_digest(path) == digest for path, digest in original_exports.items())
    api.model_info.assert_called_once()
    raw_hashes = manifest["trajectories"]
    split_hash = file_digest(root / "splits.parquet")
    monkeypatch.setattr("sys.argv", ["setup", "--reprocess", "--destination", str(root),
        "--solver-model", "tiny", "--horizon", "3", "--max-length", "256"])
    setup.main()
    rebuilt = json.loads((root / "tiny" / "manifest.json").read_text())
    assert rebuilt == manifest
    assert all(file_digest(root / "tiny" / name) == digest for name, digest in raw_hashes.items())
    assert file_digest(root / "splits.parquet") == split_hash
    assert len(list((root / "tiny").glob("checkpoints-*.parquet"))) == 1
    api.model_info.assert_called_once()
    assert snapshot.call_count == int(source_kind == 'hf')


def test_failed_hf_download_cleans_only_temporary_files(prepared_inputs: CollectionFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted download leaves no processed dataset and does not touch existing local data."""
    root = prepared_inputs.destination.parent / 'processed'
    root.mkdir()
    existing = root / 'keep.txt'
    existing.write_text('existing data')
    api = Mock(); api.dataset_info.return_value.sha = 'output-commit'
    def interrupted(repository: str, **kwargs: Any) -> str:
        """Write a partial setup-owned download, then simulate a network failure."""
        assert repository == 'owner/outputs'
        (kwargs['local_dir'] / 'partial.parquet').write_bytes(b'partial')
        raise RuntimeError('download interrupted')
    monkeypatch.setattr(setup, 'HfApi', Mock(return_value=api))
    monkeypatch.setattr(setup, 'snapshot_download', interrupted)
    monkeypatch.setattr(setup, 'tokenization_settings', Mock(return_value={}))
    monkeypatch.setattr(setup, 'TRAIN_PROBLEMS', 1)
    monkeypatch.setattr(setup, 'SUPPORTED_MODELS', {'tiny': 'unused'})
    monkeypatch.setattr('sys.argv', ['setup', '--outputs-repo', 'owner/outputs', '--revision', 'main',
        '--datasets-directory', str(prepared_inputs.datasets_directory), '--destination', str(root), '--solver-model', 'tiny'])
    with pytest.raises(RuntimeError, match='download interrupted'):
        setup.main()
    assert list(root.iterdir()) == [existing]
    assert existing.read_text() == 'existing data'


@pytest.mark.parametrize("mode,count", [("lora", "0"), ("lora", "-1"), ("head", "1"), ("full", "0"), ("full", "-1")])
def test_invalid_trainable_layers(mode: str, count: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject nonpositive layer counts and layer selection for frozen training before loading data."""
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--training-mode", mode, "--trainable-layers", count])
    with pytest.raises(ValueError, match="trainable-layers"):
        train.parse_arguments()


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_trainable_layers_argument(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the requested final-layer count through the training CLI."""
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--training-mode", mode, "--trainable-layers", "1"])
    assert train.parse_arguments().trainable_layers == 1


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_default_trainable_layers(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default token-based training to eight final layers."""
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--training-mode", mode])
    assert train.parse_arguments().trainable_layers == 8
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused"])
    assert train.parse_arguments().trainable_layers == 8


@pytest.mark.parametrize("mode", ["head", "full"])
@pytest.mark.parametrize("option", ["--lora-rank", "--lora-alpha"])
def test_adapter_options_require_lora(mode: str, option: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject adapter settings for modes that train without adapters."""
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--training-mode", mode, option, "32"])
    with pytest.raises(ValueError, match="LoRA options"):
        train.parse_arguments()


def test_pinned_hf_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remote preparation downloads only collection artifacts for the selected model at one revision."""
    api = Mock(); api.dataset_info.return_value.sha = "commit"
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(setup, "snapshot_download", download)
    root, revision = setup.download_outputs("owner/outputs", "main", "tiny", tmp_path)
    assert root == tmp_path and revision == "owner/outputs@commit"
    assert download.call_args.kwargs["revision"] == "commit"
    assert download.call_args.kwargs['local_dir'] == tmp_path
    assert all("/tiny/" in pattern for pattern in download.call_args.kwargs["allow_patterns"])
    patterns = download.call_args.kwargs['allow_patterns']
    for arm in setup.CRITIC_TRAJECTORY_ARMS:
        assert any(fnmatch(f'aime/2024/tiny/{arm}/seed_7.parquet', pattern) for pattern in patterns)
        assert not any(fnmatch(f'aime/2024/tiny/{arm}/seed_8.parquet', pattern) for pattern in patterns)
    assert not any(fnmatch('aime/2024/tiny/continue-at-1x/source_seed_0/seed_0.parquet', pattern) for pattern in patterns)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_random_head_finetune_and_resume(mode: str, critic_settings: Callable[[str], CriticConfig],
    reference_token_rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh heads train without a source checkpoint; resume exactly restores optimizer and model state."""
    settings = critic_settings("ppo-oracle-critic").model_copy(update={"training_mode": mode})
    root, output = tmp_path / "data", tmp_path / "run"
    save_token_inputs(root, settings, reference_token_rows)
    tokenizer.save_pretrained(root / "critic_only" / "tokenizer")
    command = ["train", "--data", str(root), "--output", str(output), "--variant", "ppo-oracle-critic",
        "--training-mode", mode, "--trainable-layers", "1", "--hidden-size", "16",
        "--batch-size", "2", "--accumulation-steps", "1", "--checkpoint-steps", "1", "--cpu"]
    monkeypatch.setattr("sys.argv", command)
    train.main()
    expected = load_file(str(output / "best" / "critic.safetensors"))
    record = json.loads((output / "training.json").read_text())
    assert "head_checkpoint" not in record["options"]
    assert record["options"]["learning_rate"] == 1e-5
    monkeypatch.setattr("sys.argv", command + ["--resume", str(output / "checkpoint-1")])
    train.main()
    actual = load_file(str(output / "best" / "critic.safetensors"))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("option", ["--trainable-layers", "--lora-rank", "--lora-alpha", "--head-checkpoint"])
def test_head_mode_rejects_tuning_options(option: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail before loading data if head-only training requests Qwen updates or a transfer source."""
    monkeypatch.setattr("sys.argv", ["train", "--data", "missing", "--training-mode", "head", option, "1"])
    with pytest.raises((SystemExit, ValueError)):
        train.parse_arguments()


@pytest.mark.parametrize("mode,expected", [("head", 3e-4), ("full", 1e-5), ("lora", 1e-5)])
def test_learning_rate_defaults_by_stage(mode: str, expected: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the larger head-training LR only when Qwen is frozen; explicit overrides remain available."""
    arguments = ["train", "--data", "unused", "--training-mode", mode]
    monkeypatch.setattr("sys.argv", arguments)
    assert train.parse_arguments().learning_rate == expected
    monkeypatch.setattr("sys.argv", arguments + ["--learning-rate", "1e-5"])
    assert train.parse_arguments().learning_rate == 1e-5


@pytest.mark.parametrize("steps", ["0", "-1"])
def test_checkpoint_interval_fails_early(steps: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject invalid checkpoint intervals before loading any model or data."""
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--checkpoint-steps", steps])
    with pytest.raises(SystemExit):
        train.parse_arguments()


@pytest.mark.parametrize("option", ["--learning-rate", "--oracle-weight"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_training_values_fail_early(option: str, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject non-finite optimization values before allocating Qwen or opening data shards."""
    loader = Mock()
    monkeypatch.setattr(train, "load_critic_splits", loader)
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", f"{option}={value}"])
    with pytest.raises(SystemExit):
        train.main()
    loader.assert_not_called()


def test_checkpoint_interval_longer_than_run(critic_settings: Callable[[str], CriticConfig],
    reference_token_rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A short head run still evaluates its final update and saves a resumable best checkpoint."""
    settings = critic_settings("ppo-oracle-critic").model_copy(update={"training_mode": "head"})
    root, output = tmp_path / "data", tmp_path / "run"
    save_token_inputs(root, settings, reference_token_rows)
    tokenizer.save_pretrained(root / "critic_only" / "tokenizer")
    monkeypatch.setattr("sys.argv", ["train", "--data", str(root), "--output", str(output),
        "--variant", "ppo-oracle-critic", "--training-mode", "head", "--hidden-size", "16",
        "--batch-size", "2", "--accumulation-steps", "1", "--checkpoint-steps", "125", "--cpu"])
    train.main()
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    assert [row["step"] for row in state["log_history"] if "eval_terminal_nll" in row] == [2]
    assert Path(state["best_model_checkpoint"]).name == "checkpoint-2"
    assert (output / "best" / "critic.safetensors").is_file()
    assert (output / "checkpoint-2" / "optimizer.pt").is_file()


def test_default_recipe_starts_full_tuning_without_head_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default recipe uses random heads, eight full blocks, one epoch and periodic checkpoints."""
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused"])
    args = train.parse_arguments()
    assert (args.training_mode, args.trainable_layers, args.learning_rate) == ("full", 8, 1e-5)
    assert (args.batch_size, args.accumulation_steps, args.checkpoint_steps) == (1, 16, 125)
    assert "head_checkpoint" not in vars(args)
    args.cpu = True
    training = train.training_arguments(args)
    assert training.num_train_epochs == 1
    assert training.get_warmup_steps(500) == 25
    assert training.data_seed == training.seed == 1234


def test_cached_mode_rejected_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """The removed endpoint-only mode must fail before reading data or allocating Qwen."""
    loader = Mock()
    monkeypatch.setattr(train, "load_critic_splits", loader)
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--training-mode", "frozen"])
    with pytest.raises(SystemExit):
        train.main()
    loader.assert_not_called()


@pytest.mark.parametrize("mode", ["head", "lora", "full"])
def test_epoch_override_rejected_before_loading(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An obsolete wrapper cannot silently extend any training mode beyond the one-pass budget."""
    loader = Mock()
    monkeypatch.setattr(train, "load_critic_splits", loader)
    monkeypatch.setattr("sys.argv", ["train", "--data", "unused", "--training-mode", mode, "--epochs", "5"])
    with pytest.raises(SystemExit):
        train.main()
    loader.assert_not_called()


def test_finer_setup_to_native_model(prepared_inputs: CollectionFixture, tokenizer: PreTrainedTokenizerFast,
    critic_settings: Callable[[str], CriticConfig], monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke the finer-cut setup through native head/full/LoRA forward and backward, using tiny token units."""
    from_path = prepared_inputs.outputs_directory
    root = prepared_inputs.destination.parent / "finer"
    sources = root / "raw"
    for module in ("discovery_execution.outputs.models", "discovery_execution.outputs.native_reader", "critic_only.data.critic_input_processor",
                   "discovery_execution.processing.checkpoint_reconstructor"):
        monkeypatch.setattr(f"{module}.OUTPUT_TOKENS_PER_BLOCK", 8)
    for path in from_path.rglob("meta.json"):
        metadata = json.loads(path.read_text())
        if metadata["experiment"] not in ("unaided", "oracle-execution"):
            continue
        directory = sources / path.parent.relative_to(from_path)
        write_tiny_source(directory, metadata)
    settings = critic_settings("ppo-critic")
    tokenizer.save_pretrained(settings.model)
    api = Mock(); api.model_info.return_value.sha = "local"
    monkeypatch.setattr(setup, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(setup, "SUPPORTED_MODELS", {"tiny": settings.model})
    monkeypatch.setattr(setup, "TRAIN_PROBLEMS", 1)
    monkeypatch.setattr(setup, "CHECKPOINT_INTERVAL_TOKENS", 4)
    published = root / "published"
    ParquetTrajectoryExporter(sources, published).export_trajectories(None)
    monkeypatch.setattr("sys.argv", ["setup", "--outputs-directory", str(published), "--destination", str(root / "processed"),
        "--datasets-directory", str(prepared_inputs.datasets_directory), "--solver-model", "tiny", "--horizon", "3", "--max-length", "256"])
    setup.main()
    directory = root / "processed" / "tiny"
    manifest = json.loads((directory / "manifest.json").read_text())
    boundaries = Dataset.from_parquet(str(directory / manifest["analysis_checkpoints"]["path"]))
    assert len(boundaries) == 32 and all(len(row["checkpoints"]) == 6 for row in boundaries.to_list())
    token_shard = next(iter(manifest["tokenization"]["shards"]))
    rows = Dataset.from_parquet(str(directory / token_shard)).select(range(2)).to_list()
    assert all(row["labels"] == [1, 1] for row in rows)
    assert all(row["prefix_id"].endswith("checkpoint_4tokens") for row in rows)
    for mode in ("head", "lora", "full"):
        model_settings = settings.model_copy(update={"training_mode": mode, "trainable_layers": None if mode == "head" else 1,
            "lora_rank": 2 if mode == "lora" else None, "lora_alpha": 4 if mode == "lora" else None})
        critic = create_critic(model_settings, True)
        batch = CriticBatcher(tokenizer)(rows)
        logits = critic(batch["inputs"], batch["oracle"], batch["remaining_blocks"])["logits"]
        loss = critic.loss(logits, batch["labels"], batch["oracle"])
        assert torch.isfinite(loss).all()
        (loss.mean(-1) * batch["inputs"]["solver_mask"]).sum().backward()
        assert any(parameter.grad is not None for parameter in critic.parameters() if parameter.requires_grad)


def write_tiny_source(directory: Path, metadata: dict[str, Any]) -> None:
    """Write three eight-token generation requests; their four-token midpoint cuts are reproducible."""
    directory.mkdir(parents=True)
    messages = [{"role": "user", "content": "Find the number."}]
    if metadata["experiment"] == "oracle-execution":
        messages.append({"role": "user", "content": "Use symmetry."})
    messages.append({"role": "assistant", "content": ""})
    checkpoints, requests = [], []
    for block in range(1, 4):
        messages[-1]["content"] += f"\\boxed{{{block % 2}}} word word word "
        checkpoints.append({"output_tokens": block * 8, "message_count": len(messages), "last_message_chars": len(messages[-1]["content"])})
        requests.append({"output_tokens": block * 8, "usage": {"completion_tokens": 8}, "finish_reason": "length",
                         "injected": metadata["experiment"] == "oracle-execution"})
        snapshot = {"output_tokens": block * 8, "messages": messages}
        (directory / f"checkpoint_{block}x.json.zst").write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
    state = {"output_tokens": 24, "messages": messages, "requests": requests, "checkpoints": checkpoints,
             "phase": "solve", "injected": metadata["experiment"] == "oracle-execution", "response_finished": False}
    audit = {"grading_rule": GRADING_RULE, "complete": True, "output_tokens": 24,
             "checkpoints": {f"{block}x": {"correctness": {"score": block % 2}} for block in range(1, 4)}}
    (directory / "meta.json").write_text(json.dumps(metadata))
    (directory / "audit.json").write_text(json.dumps(audit))
    (directory / "conversation.json.zst").write_bytes(zstandard.compress(json.dumps(state).encode()))
