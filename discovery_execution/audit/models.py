"""Binary checkpoint records shared by audit exports and framework inputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from discovery_execution.harness.constants import COMPETITIONS, SAFE_COMPONENT, SUPPORTED_MODELS


class BinaryGrade(BaseModel):
    """Use DE's correctness.score field with exact binary scores instead of proof grades."""

    model_config = ConfigDict(extra="forbid", strict=True)
    score: Literal[0, 1]


class AuditRecord(BaseModel):
    """One problem/seed/checkpoint row in an arm's compiled audit.jsonl."""

    model_config = ConfigDict(extra="forbid", strict=True)
    dataset: str
    solver_model: str
    experiment: Literal["unaided", "oracle-execution"]
    compute_multiplier_k: int
    inject_sketch_at: int
    solution_index: int
    problem_id: str
    seed: int
    checkpoint_multiplier_k: int
    correctness: BinaryGrade

    @model_validator(mode="after")
    def validate_record(self) -> "AuditRecord":
        """Reject invalid partitions, identities, horizons, and intervention settings."""
        competition, year = self.dataset.split("/")
        if competition not in COMPETITIONS or int(year) < 1 or str(int(year)) != year:
            raise ValueError("invalid audit dataset partition")
        if self.solver_model not in SUPPORTED_MODELS or self.seed < 0:
            raise ValueError("invalid audit model or seed")
        if not SAFE_COMPONENT.fullmatch(self.problem_id):
            raise ValueError("invalid audit problem ID")
        if not 1 <= self.checkpoint_multiplier_k <= self.compute_multiplier_k:
            raise ValueError("invalid audit checkpoint or compute multiplier")
        if not 0 <= self.inject_sketch_at < self.compute_multiplier_k or self.solution_index < 0:
            raise ValueError("invalid audit sketch selection")
        if self.experiment == "unaided" and (self.inject_sketch_at or self.solution_index):
            raise ValueError("unaided audits cannot have sketch settings")
        return self
