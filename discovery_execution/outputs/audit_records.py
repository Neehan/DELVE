"""Convert saved audit JSON into checkpoint records for audit exports and DE fitting."""

from collections.abc import Iterator
from typing import Any

from discovery_execution.audit.models import AuditRecord


def iter_audit_records(settings: dict[str, Any], result: dict[str, Any]) -> Iterator[AuditRecord]:
    """Yield one AuditRecord per checkpoint from run settings and saved audit results.

    Scores remain unchanged; cumulative success is computed by the framework loader.
    """
    identity = {"dataset": f"{settings['competition']}/{settings['year']}", "solver_model": settings["model"],
                **{key: settings[key] for key in ("experiment", "compute_multiplier_k", "inject_sketch_at",
                                                 "solution_index", "problem_id", "seed")}}
    if settings.get("continue_at") is not None:
        identity.update({key: settings[key] for key in ("source_arm", "source_seed", "continue_at")})
    for label, checkpoint in result["checkpoints"].items():
        yield AuditRecord(**identity, checkpoint_multiplier_k=int(label.removesuffix("x")),
                          correctness=checkpoint["correctness"])
