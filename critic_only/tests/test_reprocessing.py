"""Verify in-place derived-data replacement preserves raw shards and leaves failures unpublished."""

from pathlib import Path
from typing import Any

import pytest

from critic_only.data.prepared_input_publisher import PreparedInputPublisher
from discovery_execution.processing.storage import file_digest, read_manifest, write_json_atomic


def dataset_files(root: Path, name: str) -> dict[str, Any]:
    """Write minimal manifest-owned files so publication can be checked independently of tokenization."""
    root.mkdir()
    raw = root / "raw.parquet"
    raw.write_text("raw unchanged")
    token = root / f"tokens-{name}.parquet"
    token.write_text(name)
    checkpoint = root / f"checkpoints-{name}.parquet"
    checkpoint.write_text(name)
    manifest = {"trajectories": {raw.name: file_digest(raw)}, "split_sha256": "fixed",
                "tokenization": {"complete": True, "shards": {token.name: file_digest(token)}},
                "analysis_checkpoints": {"path": checkpoint.name, "sha256": file_digest(checkpoint)}}
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def test_publish_replaces_only_derived_data(tmp_path: Path) -> None:
    """After replacement only new derived files remain, with one unchanged copy of the raw shard."""
    root, staging = tmp_path / "dataset", tmp_path / "staging"
    old, new = dataset_files(root, "old"), dataset_files(staging, "new")
    PreparedInputPublisher(root).publish(staging)
    assert read_manifest(root) == new
    assert file_digest(root / "raw.parquet") == old["trajectories"]["raw.parquet"]
    assert sorted(path.name for path in root.glob("*.parquet")) == ["checkpoints-new.parquet", "raw.parquet", "tokens-new.parquet"]


@pytest.mark.parametrize("damage", ["old_file", "new_file", "raw_manifest", "split_manifest", "path_escape"])
def test_failed_publish_preserves_current_dataset(tmp_path: Path, damage: str) -> None:
    """Publication validates ownership and hashes before changing the existing manifest or removing files."""
    root, staging = tmp_path / "dataset", tmp_path / "staging"
    dataset_files(root, "old")
    new = dataset_files(staging, "new")
    if damage == "old_file":
        (root / "tokens-old.parquet").write_text("edited")
    elif damage == "new_file":
        (staging / "tokens-new.parquet").write_text("bad")
    elif damage == "raw_manifest":
        new["trajectories"] = {}
    elif damage == "split_manifest":
        new["split_sha256"] = "changed"
    else:
        new["tokenization"]["shards"] = {"../escape.parquet": "invalid"}
    write_json_atomic(staging / "manifest.json", new)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        PreparedInputPublisher(root).publish(staging)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert after == before
