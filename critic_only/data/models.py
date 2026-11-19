"""Define the input history, matching reference solution, and checkpoint labels used before tokenization."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from discovery_execution.harness.constants import MAX_BLOCKS


class PreparationConfig(BaseModel):
    """Choose the saved trajectories and final budget used to create critic examples.

    Fields:
        source_directory: Processed model directory containing full trajectory chunks.
        horizon: Final training checkpoint in analysis blocks; source generation may continue longer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_directory: Path
    horizon: int = Field(ge=2, le=MAX_BLOCKS)


class BranchOutcome(BaseModel):
    """Keep a branch's checkpoint correctness labels separate from the critic input.

    Fields:
        seed: Sampling seed used to generate the continuation.
        checkpoint_correctness: Exact binary grades at 2× through the final budget, in order.
        path: Saved trajectory ID relative to the original outputs root.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(ge=0)
    checkpoint_correctness: list[Literal[0, 1]] = Field(min_length=1)
    path: str


class InputPrefix(BaseModel):
    """Describe an individual 1× trajectory and its verified future checkpoint outcomes.

    Fields:
        prefix_id: Trajectory path and checkpoint identifier.
        oracle: Whether the solver history contains an injected sketch.
        problem_key: Competition/year/problem ID used for splitting and scoring.
        remaining_blocks: Solver budget left after the observed history.
        split: Fixed train, validation, or test assignment for this problem.
        messages: History through the first analysis checkpoint, including a hint when present.
        solution_index: Reference solution selected to match the sketch or source-seed round robin.
        reference_solution: Optional full solution supplied only to the critic.
        branch: This trajectory’s checkpoint outcomes and identity, never included in the prompt.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    prefix_id: str
    oracle: bool
    problem_key: str
    remaining_blocks: int = Field(gt=0)
    split: Literal["train", "validation", "test"]
    messages: list[dict[str, str]]
    solution_index: int = Field(ge=0)
    reference_solution: str | None
    branch: BranchOutcome
