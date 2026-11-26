"""Read a harness run's metadata, conversation and grades, and check its saved snapshots."""

import json
from pathlib import Path

import zstandard

from discovery_execution.harness.constants import AUDIT_FILENAME, CHECKPOINT_FILENAME, CONVERSATION_FILENAME, META_FILENAME, OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.harness.models import RunState
from discovery_execution.outputs.models import Trajectory


class NativeTrajectoryReader:
    """Use read_trajectory to load a complete harness run and verify its checkpoint text."""

    def __init__(self, root: Path) -> None:
        """Use root as the base directory for relative trajectory IDs."""
        self.root = root

    def read_trajectory(self, directory: Path) -> Trajectory:
        """Return the Trajectory stored in directory; reject incomplete runs or mismatched snapshots.

        Each snapshot must equal the final conversation sliced at its saved boundary.
        """
        metadata_text = (directory / META_FILENAME).read_text()
        audit_text = (directory / AUDIT_FILENAME).read_text()
        state = RunState.model_validate_json(zstandard.decompress((directory / CONVERSATION_FILENAME).read_bytes()))
        trajectory = Trajectory(directory.relative_to(self.root).as_posix(), metadata_text, state, audit_text)
        trajectory.validate()
        for checkpoint in state.checkpoints:
            block = checkpoint.output_tokens // OUTPUT_TOKENS_PER_BLOCK
            messages = [dict(message) for message in state.messages[:checkpoint.message_count]]
            messages[-1]["content"] = messages[-1]["content"][:checkpoint.last_message_chars]
            snapshot = json.loads(zstandard.decompress((directory / CHECKPOINT_FILENAME.format(k=block)).read_bytes()))
            if snapshot != {"messages": messages, "output_tokens": checkpoint.output_tokens}:
                raise ValueError(f"snapshot disagrees with conversation: {directory}")
        return trajectory
