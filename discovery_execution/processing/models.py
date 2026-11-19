"""Describe graded analysis boundaries without changing the original generation checkpoints."""

from typing import Literal

from pydantic import Field

from discovery_execution.harness.models import Checkpoint


class GradedCheckpoint(Checkpoint):
    """Locate and grade one analysis checkpoint in the saved conversation.

    output_tokens is the nominal budget; message_count and last_message_chars locate the actual text. text_tokens counts retokenized assistant tokens since the previous analysis checkpoint. correctness grades the last complete solve/revise answer. Original saved endpoints remain exact; intermediate cuts are approximate.
    """

    correctness: Literal[0, 1]
    boundary_source: Literal["saved", "reconstructed"]
    text_tokens: int = Field(ge=0)
