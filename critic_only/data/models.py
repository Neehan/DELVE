"""Validated preparation settings and auditable prefix-level records."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from discovery_execution.harness.constants import MAX_BLOCKS


class PreparationConfig(BaseModel):
    """Select a solver, budget, and deterministic problem-level split."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    outputs_directory: Path
    datasets_directory: Path
    destination: Path
    solver_model: str
    horizon: int = Field(ge=2, le=MAX_BLOCKS)
    validation_fraction: float = Field(gt=0, lt=1)
    split_seed: int = Field(ge=0)
    source_revision: str | None


class BranchOutcome(BaseModel):
    """Retain seed-level outcomes, including later regressions, for resampling studies."""

    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(ge=0)
    outcomes: list[Literal[0, 1]]
    path: str


class InputPrefix(BaseModel):
    """One branch history after its first block, with its own subsequent outcomes."""

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

    @model_validator(mode="after")
    def validate_outcomes(self) -> "InputPrefix":
        """Require one future grade per remaining block, excluding the observed input checkpoint."""
        if len(self.branch.outcomes) != self.remaining_blocks:
            raise ValueError(f"incomplete future outcome vector: {self.prefix_id}")
        return self
