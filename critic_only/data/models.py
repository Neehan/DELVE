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
    test_year: int = Field(gt=0)
    validation_fraction: float = Field(gt=0, lt=1)
    split_seed: int = Field(ge=0)
    source_revision: str


class BranchOutcome(BaseModel):
    """Retain seed-level outcomes, including later regressions, for resampling studies."""

    model_config = ConfigDict(extra="forbid", strict=True)
    condition: Literal["continue", "continue-oracle"]
    seed: int = Field(ge=0)
    solution_index: int = Field(ge=0)
    outcomes: list[Literal[0, 1]]
    path: str


class PrefixRecord(BaseModel):
    """One exact ordinary prefix with independently generated ordinary and optional oracle labels."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prefix_id: str
    problem_key: str
    competition: str
    year: int
    problem_id: str
    solver_model: str
    source_arm: str
    source_seed: int
    fork: int
    remaining_blocks: int = Field(gt=0)
    split: Literal["train", "validation", "test"]
    messages: list[dict[str, str]]
    phase: Literal["solve", "critique", "revise"]
    response_finished: bool
    solution_index: int = Field(ge=0)
    reference_solution: str
    oracle_sketch: str
    branches: list[BranchOutcome]

    @model_validator(mode="after")
    def validate_branches(self) -> "PrefixRecord":
        """Require unique independent branch identities and complete future outcome vectors."""
        identities = [(item.condition, item.seed) for item in self.branches]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate branch: {self.prefix_id}")
        if not any(item.condition == "continue" for item in self.branches):
            raise ValueError(f"prefix has no ordinary branches: {self.prefix_id}")
        if any(len(item.outcomes) != self.remaining_blocks for item in self.branches):
            raise ValueError(f"incomplete future outcome vector: {self.prefix_id}")
        return self
