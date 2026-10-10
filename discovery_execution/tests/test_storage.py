"""Verify conversation resume and exclusive trajectory ownership."""

import json
from pathlib import Path
from typing import Any

import pytest
import zstandard

from discovery_execution.harness.constants import (
    CHECKPOINT_FILENAME, CONVERSATION_FILENAME, OUTPUT_TOKENS_PER_BLOCK, TOTAL_BLOCKS,
)
from discovery_execution.harness.models import RunState
from discovery_execution.harness.storage import Storage, atomic_write


@pytest.fixture
def metadata() -> dict[str, Any]:
    """Use immutable identity settings to exercise matching and mismatched resumes."""
    return {"model": "model", "seed": 0, "compute_multiplier_k": TOTAL_BLOCKS}


def test_resume_saved_state(tmp_path: Path, metadata: dict[str, Any], conversation: RunState) -> None:
    """Reopen a trajectory after process shutdown and preserve its exact continuation state."""
    conversation.messages[0]["content"] += "\n理由: α² + β² = γ²"
    conversation.checkpoints[0].last_message_chars = len(conversation.messages[0]["content"])
    with Storage(tmp_path) as storage:
        assert storage.load(metadata) == RunState.empty()
        storage.save(conversation)
    with Storage(tmp_path) as storage:
        assert storage.load(metadata) == conversation
    saved = json.loads(zstandard.decompress((tmp_path / CONVERSATION_FILENAME).read_bytes()))
    assert saved["messages"] == conversation.messages


@pytest.mark.parametrize("compute_multiplier_k", range(1, TOTAL_BLOCKS + 1))
def test_skip_completed_after_validating_settings(tmp_path: Path, metadata: dict[str, Any], conversation: RunState,
                                                  compute_multiplier_k: int, caplog: pytest.LogCaptureFixture) -> None:
    """Skip a matching completed run but reject changed settings before treating it as reusable."""
    metadata["compute_multiplier_k"] = compute_multiplier_k
    conversation.output_tokens = compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK
    with Storage(tmp_path) as storage:
        storage.load(metadata)
        storage.save(conversation)
        saved = (tmp_path / CONVERSATION_FILENAME).read_bytes()
        with caplog.at_level("INFO"):
            assert storage.load(metadata) == conversation
        assert "Skipped completed trajectory" in caplog.text
        with pytest.raises(ValueError, match="configuration differs"):
            storage.load({**metadata, "model": "different-model"})
        with pytest.raises(ValueError, match="configuration differs"):
            storage.load({**metadata, "compute_multiplier_k": compute_multiplier_k + 1})
        assert (tmp_path / CONVERSATION_FILENAME).read_bytes() == saved
        assert storage.read_metadata() == metadata


def test_exclusive_directory_lock(tmp_path: Path, metadata: dict[str, Any]) -> None:
    """A second writer fails while the first owns the directory, and succeeds after release."""
    with Storage(tmp_path) as storage:
        storage.load(metadata)
        with pytest.raises(BlockingIOError):
            Storage(tmp_path)
    with Storage(tmp_path) as storage:
        assert storage.read_metadata() == metadata


def test_resume_repairs_snapshots(tmp_path: Path, metadata: dict[str, Any], conversation: RunState,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover the committed conversation if saving audit snapshots fails afterward."""
    def fail_snapshots(storage: Storage, state: RunState) -> None:
        """Simulate a crash between committing the conversation and writing snapshots."""
        raise OSError("snapshot write failed")

    with Storage(tmp_path) as storage:
        storage.load(metadata)
        with monkeypatch.context() as patch:
            patch.setattr(Storage, "_save_checkpoints", fail_snapshots)
            with pytest.raises(OSError, match="snapshot write failed"):
                storage.save(conversation)
    assert not (tmp_path / CHECKPOINT_FILENAME.format(k=1)).exists()
    with Storage(tmp_path) as storage:
        assert storage.load(metadata) == conversation
        assert storage.read_checkpoints() == [
            {"output_tokens": conversation.output_tokens, "messages": conversation.messages}
        ]


def test_snapshots_exclude_later_continuation(tmp_path: Path, metadata: dict[str, Any], conversation: RunState) -> None:
    """A continuation after 1x must not change that snapshot, including after resume."""
    with Storage(tmp_path) as storage:
        storage.load(metadata)
        storage.save(conversation)
        snapshots = storage.read_checkpoints()
        conversation.messages[-1]["content"] += " LATER_RESPONSE"
        conversation.output_tokens += 1
        storage.save(conversation)
        assert storage.read_checkpoints() == snapshots
    with Storage(tmp_path) as storage:
        assert storage.load(metadata) == conversation
        assert storage.read_checkpoints() == snapshots


def test_corrupt_conversation_does_not_resume_from_snapshots(tmp_path: Path, metadata: dict[str, Any],
                                                           conversation: RunState) -> None:
    """Only the live conversation determines resume state; snapshots cannot silently replace it."""
    with Storage(tmp_path) as storage:
        storage.load(metadata)
        storage.save(conversation)
    (tmp_path / CONVERSATION_FILENAME).write_bytes(b"invalid compressed data")
    with Storage(tmp_path) as storage:
        with pytest.raises(zstandard.ZstdError):
            storage.load(metadata)


def test_missing_conversation_with_snapshots_fails(tmp_path: Path, metadata: dict[str, Any]) -> None:
    """Missing resume data must not silently restart a trajectory that has audit snapshots."""
    (tmp_path / CHECKPOINT_FILENAME.format(k=1)).write_bytes(zstandard.compress(b"{}"))
    with Storage(tmp_path) as storage:
        with pytest.raises(FileNotFoundError, match="conversation is missing"):
            storage.load(metadata)


def test_resume_writes_only_missing_snapshot(tmp_path: Path, metadata: dict[str, Any], conversation: RunState,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover a failed second snapshot without replacing the already published first boundary."""
    second = tmp_path / CHECKPOINT_FILENAME.format(k=2)
    writes: list[Path] = []
    interrupted = True

    def publish(path: Path, content: bytes) -> None:
        """Fail the second snapshot after the updated conversation has been committed."""
        writes.append(path)
        if path == second and interrupted:
            raise OSError("interrupted second snapshot")
        atomic_write(path, content)

    with Storage(tmp_path) as storage:
        storage.load(metadata)
        storage.save(conversation)
        first_snapshot = storage.read_checkpoints()[0]
        conversation.messages[-1]["content"] += " SECOND_BLOCK"
        conversation.output_tokens = 2 * OUTPUT_TOKENS_PER_BLOCK
        conversation.checkpoints.append(conversation.checkpoints[0].model_copy(update={
            "output_tokens": conversation.output_tokens, "last_message_chars": len(conversation.messages[-1]["content"]),
        }))
        monkeypatch.setattr("discovery_execution.harness.storage.atomic_write", publish)
        with pytest.raises(OSError, match="interrupted second snapshot"):
            storage.save(conversation)
    assert not second.exists()
    interrupted = False
    writes.clear()
    with Storage(tmp_path) as storage:
        assert storage.load(metadata) == conversation
        assert writes == [second]
        snapshots = storage.read_checkpoints()
        assert snapshots == [first_snapshot, {"output_tokens": conversation.output_tokens, "messages": conversation.messages}]
        writes.clear()
        storage.load(metadata)
        assert not writes


@pytest.mark.parametrize("block,tokens", [(2, 2 * OUTPUT_TOKENS_PER_BLOCK), (1, OUTPUT_TOKENS_PER_BLOCK + 1)])
def test_invalid_snapshot_boundary_fails(tmp_path: Path, block: int, tokens: int) -> None:
    """Reject a missing earlier file or a token count that disagrees with the filename."""
    snapshot = {"output_tokens": tokens, "messages": [{"role": "assistant", "content": "answer"}]}
    path = tmp_path / CHECKPOINT_FILENAME.format(k=block)
    path.write_bytes(zstandard.compress(json.dumps(snapshot).encode("utf-8")))
    with Storage(tmp_path) as storage:
        with pytest.raises(ValueError, match="checkpoint boundary"):
            storage.read_checkpoints()
