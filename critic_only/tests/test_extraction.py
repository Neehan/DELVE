"""Compare cached vectors with real transformer forwards and verify extraction restarts."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModel, PreTrainedTokenizerFast

from critic_only.cli.extract import publish_feature_manifest
from critic_only.data.extractor import HiddenStateExtractor
from critic_only.data.features import load_feature_splits
from critic_only.data.tokenizer import token_digest
from critic_only.models.models import CriticConfig
from discovery_execution.processing.storage import write_json_atomic


def tokenized_fixture(rows: list[dict[str, Any]]) -> DatasetDict:
    """Attach stable IDs and exact token hashes to two differently sized input views."""
    return DatasetDict({split: Dataset.from_list([
        {**row, "prefix_id": f"{split}-{index}", "input_hash": token_digest(row["input_ids"]),
         "privileged_input_hash": token_digest(row["privileged_input_ids"]) if row["privileged_input_ids"] is not None else None}
        for index, row in enumerate(rows)]) for split in ("train", "validation", "test")})


def test_extraction_matches_direct_forward_and_resumes(critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, token_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saved float32 vectors match the last real token, including padding and privileged views."""
    settings = critic_settings("ppo-critic")
    model = AutoModel.from_pretrained(settings.model).eval()
    datasets = tokenized_fixture(token_rows)
    root = tmp_path / "processed"
    destination = root / "critic_only" / "hidden_states"
    extractor = HiddenStateExtractor(model, tokenizer, 2)
    extractor.extract_shards(datasets, destination, "signature", 0, 1)
    expected = {row["prefix_id"]: row for row in datasets["test"].to_list()}
    rows = pq.read_table(destination / "test" / "part-00000.parquet").to_pylist()
    for row in rows:
        source = expected[row["prefix_id"]]
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
        """Completed shards must not invoke the transformer again."""
        raise AssertionError("recomputed completed shard")
    monkeypatch.setattr(model, "forward", forbidden)
    extractor.extract_shards(datasets, destination, "signature", 0, 1)
    with pytest.raises(ValueError, match="configuration differs"):
        extractor.extract_shards(datasets, destination, "changed", 0, 1)
    write_json_atomic(root / "manifest.json", {"features": {"signature": "signature", "complete": False}})
    publish_feature_manifest(root, datasets)
    loaded, manifest = load_feature_splits(root, ("test",))
    assert len(loaded["test"]) == 2 and manifest["features"]["complete"]


def test_token_corruption_fails_before_forward(critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, token_rows: list[dict[str, Any]], tmp_path: Path) -> None:
    """Feature extraction rejects token IDs changed after their input hashes were recorded."""
    datasets = tokenized_fixture(token_rows)
    rows = datasets["test"].to_list()
    rows[0]["input_ids"] = [3, 3]
    datasets["test"] = Dataset.from_list(rows)
    model = AutoModel.from_pretrained(critic_settings("ppo-critic").model)
    with pytest.raises(ValueError, match="token IDs changed"):
        HiddenStateExtractor(model, tokenizer, 2).extract_shards(datasets, tmp_path / "features", "sig", 0, 1)


def test_shard_ranks_cover_each_example_once(critic_settings: Callable[[str], CriticConfig],
    tokenizer: PreTrainedTokenizerFast, token_rows: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent ranks publish disjoint shards, and the coordinator rejects missing work."""
    monkeypatch.setattr("critic_only.data.extractor.FEATURE_SHARD_ROWS", 1)
    monkeypatch.setattr("critic_only.cli.extract.FEATURE_SHARD_ROWS", 1)
    model = AutoModel.from_pretrained(critic_settings("ppo-critic").model)
    datasets = tokenized_fixture(token_rows)
    root = tmp_path / "processed"
    root.mkdir()
    write_json_atomic(root / "manifest.json", {"features": {"signature": "sig", "complete": False}})
    destination = root / "critic_only" / "hidden_states"
    extractor = HiddenStateExtractor(model, tokenizer, 1)
    extractor.extract_shards(datasets, destination, "sig", 0, 2)
    with pytest.raises(ValueError, match="missing feature shard"):
        publish_feature_manifest(root, datasets)
    extractor.extract_shards(datasets, destination, "sig", 1, 2)
    publish_feature_manifest(root, datasets)
    rows = [row for shard in sorted((destination / "train").glob("*.parquet")) for row in pq.read_table(shard).to_pylist()]
    assert len(rows) == len({row["prefix_id"] for row in rows}) == 2
