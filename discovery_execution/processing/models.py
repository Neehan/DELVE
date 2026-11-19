"""Describe graded analysis boundaries without changing the original generation checkpoints."""

from typing import Literal

from discovery_execution.harness.models import Checkpoint


class GradedCheckpoint(Checkpoint):
    """Locate and grade one analysis checkpoint in the saved conversation.

    Inherits output_tokens, message_count and last_message_chars. correctness is the last complete solve/revise answer's grade. boundary_source distinguishes saved or request-end boundaries from reconstructed text cuts.
    """

    correctness: Literal[0, 1]
    boundary_source: Literal["saved", "request", "reconstructed"]
