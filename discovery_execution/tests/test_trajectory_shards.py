"""Check single-pass split buffers retain rows across full and partial shards."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from discovery_execution.processing.storage import file_digest
from discovery_execution.processing.trajectory_shard_writer import TrajectoryShardWriter


def test_partial_final_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flush at the byte threshold and retain the last partial shard in order."""
    rows = [{'trajectory_id': str(index), 'text': 'sample'} for index in range(3)]
    monkeypatch.setattr('discovery_execution.processing.trajectory_shard_writer.PARQUET_SHARD_BYTES',
        len(json.dumps(rows[0]).encode()) * 2)
    writer = TrajectoryShardWriter(tmp_path / 'train')
    for row in rows:
        writer.append(row)
    checksums = writer.finish()
    assert len(checksums) == 2
    restored = []
    for name, digest in checksums.items():
        path = tmp_path / 'train' / name
        assert file_digest(path) == digest
        restored.extend(pq.read_table(path).to_pylist())
    assert restored == rows
    assert writer.finish() == checksums


def test_empty_split_writes_no_shard(tmp_path: Path) -> None:
    """A split without selected trajectories produces neither an empty shard nor a directory."""
    directory = tmp_path / 'validation'
    assert TrajectoryShardWriter(directory).finish() == {}
    assert not directory.exists()
