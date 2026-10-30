"""Check HF hidden-state prediction, exact vector alignment, and chunk resume."""

import os
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import AutoModel, DataCollatorWithPadding, PreTrainedTokenizerFast, Trainer, TrainingArguments

from critic_only.data.dataset_loader import load_critic_splits
from critic_only.data.storage import CriticInputStorage
from critic_only.hidden_states.encoder import HiddenStateEncoder
from critic_only.hidden_states.runner import HiddenStateRunner
from critic_only.hidden_states.storage import HiddenStateStorage
from critic_only.data.tokenizer import token_digest
from critic_only.models.models import CriticConfig
from discovery_execution.processing.storage import file_digest, read_manifest, write_json_atomic, write_parquet_atomic


def prediction_trainer(model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, directory: Path, batch_size: int) -> Trainer:
    """Use standard HF prediction on a small CPU model without any training labels."""
    args = TrainingArguments(output_dir=str(directory / "prediction"), per_device_eval_batch_size=batch_size,
                             use_cpu=True, disable_tqdm=True, report_to="none", label_names=[])
    return Trainer(model=HiddenStateEncoder(model), args=args,
                   data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer)


def save_token_chunks(root: Path, rows: list[dict[str, Any]], chunk_size: int) -> dict[str, str]:
    """Write stable input chunks in deliberately unsorted length order."""
    checksums = {}
    for split in ("train", "validation", "test"):
        values = [{**row, "prefix_id": f"{split}-{index}", "input_hash": token_digest(row["input_ids"]),
                   "privileged_input_hash": token_digest(row["privileged_input_ids"]) if row["privileged_input_ids"] else None}
                  for index, row in enumerate(rows)]
        for chunk, start in enumerate(range(0, len(values), chunk_size)):
            path = root / split / "critic_only" / "tokenized" / f"part-{chunk:05d}.parquet"
            write_parquet_atomic(path, pa.Table.from_pylist(values[start:start + chunk_size]))
            checksums[str(path.relative_to(root))] = file_digest(path)
    write_json_atomic(root / "manifest.json", {"tokenization": {"shards": checksums},
                      "features": {"signature": "signature", "complete": False}})
    return checksums


@pytest.mark.parametrize("batch_size", [1, 2])
def test_hidden_states_match_inputs_and_resume(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
    token_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int) -> None:
    """Within-chunk length sorting must preserve original row order and both input versions."""
    model = AutoModel.from_pretrained(critic_settings("ppo-critic").model).eval()
    shards = save_token_chunks(tmp_path, token_rows, 2)
    inputs, outputs = CriticInputStorage(tmp_path), HiddenStateStorage(tmp_path)
    runner = HiddenStateRunner(inputs, outputs, prediction_trainer(model, tokenizer, tmp_path, batch_size))
    runner.run()
    for name in shards:
        sources = pq.read_table(tmp_path / name).to_pylist()
        rows = pq.read_table(outputs.chunk_path(name)).to_pylist()
        assert [row["prefix_id"] for row in rows] == [row["prefix_id"] for row in sources]
        for source, row in zip(sources, rows):
            assert row["labels"] == source["labels"]
            for tokens, vector in [("input_ids", "hidden_state"), ("privileged_input_ids", "privileged_hidden_state")]:
                if source[tokens] is None:
                    assert row[vector] is None
                    continue
                ids = torch.tensor([source[tokens]])
                with torch.no_grad():
                    actual = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, -1]
                torch.testing.assert_close(torch.tensor(row[vector]), actual, atol=1e-6, rtol=1e-5)
    def forbidden(*args: Any, **kwargs: Any) -> None:
        """Completed files must not run Qwen again."""
        raise AssertionError("recomputed hidden states")
    monkeypatch.setattr(model, "forward", forbidden)
    runner.run()
    with pytest.raises(ValueError, match="configuration differs"):
        name = next(iter(shards))
        outputs.validate_chunk(name, inputs.read_chunk(name, shards[name]), "changed")
    HiddenStateStorage(tmp_path).complete_cache()
    loaded, manifest = load_critic_splits(tmp_path, ("test",), "frozen")
    assert len(loaded["test"]) == 2 and manifest["features"]["complete"]


def test_corrupt_input_hash_rejected(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
    token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """A changed sequence cannot be encoded under its old input hash."""
    shards = save_token_chunks(tmp_path, token_rows, 2)
    name = sorted(shards)[0]
    rows = pq.read_table(tmp_path / name).to_pylist()
    rows[0]["input_ids"] = [3, 3]
    write_parquet_atomic(tmp_path / name, pa.Table.from_pylist(rows))
    shards[name] = file_digest(tmp_path / name)
    manifest = read_manifest(tmp_path)
    manifest["tokenization"]["shards"] = shards
    write_json_atomic(tmp_path / "manifest.json", manifest)
    model = AutoModel.from_pretrained(critic_settings("ppo-critic").model)
    with pytest.raises(ValueError, match="token IDs changed"):
        HiddenStateRunner(CriticInputStorage(tmp_path), HiddenStateStorage(tmp_path),
                          prediction_trainer(model, tokenizer, tmp_path, 2)).run()


def test_incomplete_cache_resumes_missing_chunks(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
    token_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a completed chunk and predict only the remaining chunks after interruption."""
    shards = save_token_chunks(tmp_path, token_rows, 1)
    model = AutoModel.from_pretrained(critic_settings("ppo-critic").model)
    inputs, outputs = CriticInputStorage(tmp_path), HiddenStateStorage(tmp_path)
    trainer = prediction_trainer(model, tokenizer, tmp_path, 2)
    runner = HiddenStateRunner(inputs, outputs, trainer)
    name = sorted(shards)[0]
    rows = inputs.read_chunk(name, shards[name])
    outputs.save_chunk(name, runner._encode_chunk(rows), "signature")
    saved = outputs.chunk_path(name).read_bytes()
    with pytest.raises(ValueError, match="missing hidden-state shard"):
        outputs.complete_cache()
    calls = []
    predict = trainer.predict
    def record_prediction(*args: Any, **kwargs: Any) -> Any:
        """Count calls to ensure resume skips the saved chunk."""
        calls.append(True)
        return predict(*args, **kwargs)
    monkeypatch.setattr(trainer, "predict", record_prediction)
    runner.run()
    assert len(calls) == len(shards) - 1
    assert outputs.chunk_path(name).read_bytes() == saved
    outputs.complete_cache()
    loaded, _ = load_critic_splits(tmp_path, ("train",), "frozen")
    assert len(loaded["train"]) == len(set(loaded["train"]["prefix_id"])) == 2


def test_distributed_hf_prediction_preserves_rows(critic_settings: Callable[[str], CriticConfig], tokenizer: PreTrainedTokenizerFast,
    token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Two CPU ranks gather odd-sized paired inputs without duplicating or reordering saved rows."""
    model_path = critic_settings("ppo-critic").model
    shards = save_token_chunks(tmp_path, token_rows, 2)
    tokenizer.save_pretrained(tmp_path / "critic_only" / "tokenizer")
    manifest = read_manifest(tmp_path)
    manifest.pop("features")
    manifest["tokenization"].update(complete=True, settings={"model": model_path, "revision": "main", "hidden_size": 16})
    write_json_atomic(tmp_path / "manifest.json", manifest)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    launcher = tmp_path / "cpu_prediction.py"
    # Simulate a CPU-only host: Accelerate passes accelerator device IDs to Gloo barriers on MPS machines.
    launcher.write_text("import runpy\nfrom unittest.mock import patch\nimport torch\n"
                       "if __name__ == '__main__':\n"
                       "    with patch('torch._C._get_accelerator', return_value=torch.device('cpu')):\n"
                       "        runpy.run_module('critic_only.cli.cache_hidden_states', run_name='__main__')\n")
    command = [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1", f"--master-port={port}",
               "--nproc_per_node=2", str(launcher), "--data", str(tmp_path), "--batch-size", "2", "--cpu"]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_manifest(tmp_path)["features"]["complete"]
    model = AutoModel.from_pretrained(model_path).eval()
    for name in shards:
        sources = pq.read_table(tmp_path / name).to_pylist()
        saved = pq.read_table(HiddenStateStorage(tmp_path).chunk_path(name)).to_pylist()
        assert [row["prefix_id"] for row in saved] == [row["prefix_id"] for row in sources]
        for source, row in zip(sources, saved):
            for token_key, vector_key in (("input_ids", "hidden_state"), ("privileged_input_ids", "privileged_hidden_state")):
                if source[token_key] is None:
                    assert row[vector_key] is None
                    continue
                ids = torch.tensor([source[token_key]])
                with torch.no_grad():
                    expected = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, -1]
                torch.testing.assert_close(torch.tensor(row[vector_key]), expected, atol=1e-6, rtol=1e-5)
