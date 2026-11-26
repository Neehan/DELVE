"""One validated trajectory shared by native packing and Parquet consumers."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from discovery_execution.harness.models import RunState
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK


def bank_path(trajectory_id: str, settings: dict[str, Any]) -> Path:
    """Validate native path identity and distinguish parent seeds from child seeds."""
    path = Path(trajectory_id)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe trajectory path")
    expected = Path(settings["competition"]) / str(settings["year"]) / settings["model"] / settings["arm"] / settings["problem_id"]
    directory = expected.parent
    if settings.get("continue_at") is not None:
        expected = expected / f"seed_{settings['source_seed']}" / f"branch_{settings['seed']}"
        directory = directory / f"source_seed_{settings['source_seed']}"
    else:
        expected = expected / f"seed_{settings['seed']}"
    if path != expected:
        raise ValueError(f"trajectory identity does not match path: {trajectory_id}")
    return directory / f"seed_{settings['seed']}.parquet"


def identity_columns(trajectory_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Derive the queryable columns from authoritative native metadata and its validated path."""
    return {"trajectory_id": trajectory_id,
            **{key: metadata[key] for key in ("competition", "year", "model", "problem_id", "seed")},
            "arm": bank_path(trajectory_id, metadata).parts[3],
            "branch_seed": metadata["seed"] if metadata.get("continue_at") is not None else None,
            **{key: metadata.get(key) for key in ("source_seed", "source_arm", "continue_at")}}


def checkpoint_grades(audit: dict[str, Any]) -> list[dict[str, int]]:
    """Expose raw binary grades while retaining complete audit JSON separately."""
    return [{"block": int(label.removesuffix("x")), "score": value["correctness"]["score"]}
            for label, value in audit["checkpoints"].items()]


@dataclass(frozen=True)
class Trajectory:
    """Preserve complete metadata, conversation state, and unmodified native audit text."""

    trajectory_id: str
    metadata: str
    state: RunState
    audit: str

    def bank_path(self) -> Path:
        """Separate root seeds and child seeds from different source trajectories."""
        return bank_path(self.trajectory_id, json.loads(self.metadata))

    def validate(self) -> None:
        """Require a complete curve with exact native token boundaries and audit coverage."""
        metadata, audit = json.loads(self.metadata), json.loads(self.audit)
        budget = metadata["compute_multiplier_k"] * OUTPUT_TOKENS_PER_BLOCK
        if not audit["complete"] or audit["output_tokens"] != budget or self.state.output_tokens != budget:
            raise ValueError(f"incomplete branch audit or conversation: {self.trajectory_id}")
        fork = metadata["continue_at"] if metadata.get("continue_at") is not None else 0
        expected = {block * OUTPUT_TOKENS_PER_BLOCK for block in range(fork + 1, metadata["compute_multiplier_k"] + 1)}
        actual = {checkpoint.output_tokens for checkpoint in self.state.checkpoints}
        grades = {f"{tokens // OUTPUT_TOKENS_PER_BLOCK}x" for tokens in expected}
        if actual != expected or len(actual) != len(self.state.checkpoints) or set(audit["checkpoints"]) != grades:
            raise ValueError(f"missing or duplicate checkpoints: {self.trajectory_id}")
        self.bank_path()

    def to_row(self) -> dict[str, Any]:
        """Expose queryable identities, nested messages/boundaries, and raw checkpoint scores."""
        metadata, audit = json.loads(self.metadata), json.loads(self.audit)
        conversation = self.state.model_dump()
        conversation["requests_json"] = json.dumps(conversation.pop("requests"))
        return {**identity_columns(self.trajectory_id, metadata),
                "metadata": self.metadata, "conversation": conversation, "audit": self.audit,
                "grades": checkpoint_grades(audit)}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Trajectory":
        """Rebuild native state without changing request history or checkpoint boundaries."""
        conversation = dict(row["conversation"])
        conversation["requests"] = json.loads(conversation.pop("requests_json"))
        trajectory = cls(row["trajectory_id"], row["metadata"], RunState.model_validate(conversation), row["audit"])
        trajectory.validate()
        if trajectory.to_row() != row:
            raise ValueError(f"Parquet trajectory columns disagree: {trajectory.trajectory_id}")
        return trajectory
