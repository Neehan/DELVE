"""Validated preparation settings and auditable prefix-level records."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from discovery_execution.harness.constants import MAX_BLOCKS


class PreparationConfig(BaseModel):
    """Select a processed collection and a terminal budget for critic inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_directory: Path
    horizon: int = Field(ge=2, le=MAX_BLOCKS)


class BranchOutcome(BaseModel):
    """Record the final binary grade and the saved trajectory that produced it."""

    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(ge=0)
    final_correctness: Literal[0, 1]
    path: str


class InputPrefix(BaseModel):
    """One branch history after its first block, paired with its final outcome."""

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
    branch: BranchOutcome
