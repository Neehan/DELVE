"""Save run settings, resumable conversations, and token-boundary audit snapshots."""

import fcntl
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Any

import zstandard

from discovery_execution.harness.constants import (
    CHECKPOINT_FILENAME, CONVERSATION_FILENAME, LOCK_FILENAME, OUTPUT_TOKENS_PER_BLOCK, META_FILENAME, TOTAL_BLOCKS,
)
from discovery_execution.harness.models import RunState


def model_output_directory(root: Path, competition: str, year: int, model: str) -> Path:
    """Locate the shared output subtree for one dataset partition and solver."""
    return root / competition / str(year) / model


class Storage:
    """Resume from the conversation; publish separate snapshots for offline auditing."""

    def __init__(self, directory: Path) -> None:
        """Acquire one seed's process lock before accessing its files."""
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._lock = (directory / LOCK_FILENAME).open("a", encoding="utf-8")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._lock.close()
            raise

    def _bind(self, run_metadata: dict[str, Any]) -> None:
        """Reject a changed run configuration instead of overwriting an existing run."""
        path = self.directory / META_FILENAME
        if path.exists():
            if self.read_metadata() != run_metadata:
                raise ValueError(f"run configuration differs: {self.directory}")
        else:
            write_json(path, run_metadata)

    def read_metadata(self) -> dict[str, Any]:
        """Read run settings and the dataset/problem identifiers."""
        return json.loads((self.directory / META_FILENAME).read_text(encoding="utf-8"))

    def load(self, run_metadata: dict[str, Any]) -> RunState:
        """Validate resume settings and return progress, including an already exhausted budget."""
        self._bind(run_metadata)
        path = self.directory / CONVERSATION_FILENAME
        if not path.exists():
            if any(self.directory.glob(CHECKPOINT_FILENAME.format(k="*"))):
                raise FileNotFoundError(f"audit snapshots exist but the resumable conversation is missing: {path}")
            return RunState.empty()
        state = RunState.model_validate_json(zstandard.decompress(path.read_bytes()))
        token_budget = run_metadata["compute_multiplier_k"] * OUTPUT_TOKENS_PER_BLOCK
        if state.output_tokens > token_budget:
            raise ValueError("saved conversation exceeds the configured token budget")
        if state.checkpoints:
            self._save_checkpoints(state)
        if state.output_tokens == token_budget:
            logging.info("Skipped completed trajectory %s", self.directory)
        return state

    def read_checkpoints(self) -> list[dict[str, Any]]:
        """Read consecutive boundary snapshots, validating names against their token counts."""
        paths = sorted(self.directory.glob(CHECKPOINT_FILENAME.format(k="*")))
        if not paths:
            raise FileNotFoundError(self.directory / CHECKPOINT_FILENAME.format(k=1))
        snapshots = []
        for block, path in enumerate(paths, start=1):
            snapshot = json.loads(zstandard.decompress(path.read_bytes()))
            if (block > TOTAL_BLOCKS or path.name != CHECKPOINT_FILENAME.format(k=block)
                    or snapshot["output_tokens"] != block * OUTPUT_TOKENS_PER_BLOCK):
                raise ValueError(f"missing or invalid checkpoint boundary: {path}")
            snapshots.append(snapshot)
        return snapshots

    def save(self, state: RunState) -> None:
        """Save conversation progress after each response and audit snapshots at token boundaries."""
        atomic_write(self.directory / CONVERSATION_FILENAME,
                     zstandard.compress(state.model_dump_json().encode("utf-8")))
        if state.output_tokens and state.output_tokens % OUTPUT_TOKENS_PER_BLOCK == 0:
            self._save_checkpoints(state)

    def _save_checkpoints(self, state: RunState) -> None:
        """Write missing boundary snapshots; resume repairs interrupted publication without rewriting earlier files."""
        for checkpoint in state.checkpoints:
            block = checkpoint.output_tokens // OUTPUT_TOKENS_PER_BLOCK
            path = self.directory / CHECKPOINT_FILENAME.format(k=block)
            if path.exists():
                continue
            messages = [dict(message) for message in state.messages[:checkpoint.message_count]]
            messages[-1]["content"] = messages[-1]["content"][:checkpoint.last_message_chars]
            snapshot = {"output_tokens": checkpoint.output_tokens, "messages": messages}
            atomic_write(path, zstandard.compress(json.dumps(snapshot, ensure_ascii=False).encode("utf-8")))

    def __enter__(self) -> "Storage":
        """Keep the trajectory lock for the lifetime of the run."""
        return self

    def __exit__(self, exception_type: type[BaseException] | None,
                 exception: BaseException | None, traceback: TracebackType | None) -> None:
        """Release the trajectory lock on success or failure."""
        self.close()

    def close(self) -> None:
        """Release exclusive ownership of this seed."""
        self._lock.close()


def atomic_write(path: Path, content: bytes) -> None:
    """Publish a complete file only after flushing its contents to disk."""
    with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    """Serialize a complete JSON artifact with a trailing newline."""
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
