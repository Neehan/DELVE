"""Reconstruct native binary audit records from a trajectory's saved settings and grades."""

from collections.abc import Iterator
from typing import Any

from discovery_execution.audit.models import AuditRecord


def audit_records(settings: dict[str, Any], result: dict[str, Any]) -> Iterator[AuditRecord]:
    """Keep raw checkpoint correctness and source identities unchanged for DE aggregation."""
    identity = {"dataset": f"{settings['competition']}/{settings['year']}", "solver_model": settings["model"],
                **{key: settings[key] for key in ("experiment", "compute_multiplier_k", "inject_sketch_at",
                                                 "solution_index", "problem_id", "seed")}}
    if settings.get("continue_at") is not None:
        identity.update({key: settings[key] for key in ("source_arm", "source_seed", "continue_at")})
    for label, checkpoint in result["checkpoints"].items():
        yield AuditRecord(**identity, checkpoint_multiplier_k=int(label.removesuffix("x")),
                          correctness=checkpoint["correctness"])
