"""Define the input history, matching reference solution, and final label used before tokenization."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from discovery_execution.harness.constants import MAX_BLOCKS


class PreparationConfig(BaseModel):
    """Choose the saved trajectories and final budget used to create critic examples.

    Fields:
        source_directory: Processed model directory containing full trajectory chunks.
        horizon: Required final solver budget in blocks, measured from the problem start.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_directory: Path
    horizon: int = Field(ge=2, le=MAX_BLOCKS)


class BranchOutcome(BaseModel):
    """Keep a branch's final correctness label separate from the critic input.

    Fields:
        seed: Sampling seed used to generate the continuation.
        final_correctness: 1 if the final answer is correct, otherwise 0.
        path: Saved trajectory ID relative to the original outputs root.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(ge=0)
    final_correctness: Literal[0, 1]
    path: str


class InputPrefix(BaseModel):
    """Describe a shared 1× state and the independent continuations that supervise it.

    Fields:
        prefix_id: Source checkpoint and oracle-condition identifier; independent of child count.
        oracle: Whether the solver history contains an injected sketch.
        problem_key: Competition/year/problem ID used for splitting and scoring.
        fork: Shared source checkpoint in blocks.
        remaining_blocks: Solver budget left after the observed history.
        split: Fixed train, validation, or test assignment for this problem.
        messages: Exact source history plus an optional oracle hint, without child responses.
        phase: Solver phase at the checkpoint: solve, critique, or revise.
        response_finished: Whether the current phase ended naturally; false after a new hint.
        solution_index: Reference solution selected to match the sketch or source-seed round robin.
        reference_solution: Optional full solution supplied only to the critic.
        branches: Independent final outcomes and identities, never included in the prompt.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    prefix_id: str
    oracle: bool
    problem_key: str
    fork: int
    remaining_blocks: int = Field(gt=0)
    split: Literal["train", "validation", "test"]
    messages: list[dict[str, str]]
    phase: Literal["solve", "critique", "revise"]
    response_finished: bool
    solution_index: int = Field(ge=0)
    reference_solution: str | None
    branches: list[BranchOutcome] = Field(min_length=1)
