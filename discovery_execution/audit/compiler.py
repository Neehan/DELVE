"""Compile per-seed binary audits into DE-style checkpoint rows for one arm."""

import fcntl
import json
from pathlib import Path

from discovery_execution.audit.constants import COMPILED_AUDIT_FILENAME
from discovery_execution.audit.models import AuditRecord
from discovery_execution.harness.constants import AUDIT_FILENAME, LOCK_FILENAME, OUTPUT_TOKENS_PER_BLOCK, RUN_FILENAME
from discovery_execution.harness.storage import atomic_write


class AuditCompiler:
    """Compile all audited seeds under an arm, retaining rows outside any audit CLI filter."""

    def __init__(self, directory: Path) -> None:
        """Select the run arm directory that owns the compiled export."""
        self.directory = directory

    def compile(self) -> Path:
        """Serialize publication and atomically replace the export only after every row validates."""
        destination = self.directory / COMPILED_AUDIT_FILENAME
        with (self.directory / LOCK_FILENAME).open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            rows = self._collect_records()
            content = "".join(row.model_dump_json() + "\n" for row in rows)
            atomic_write(destination, content.encode("utf-8"))
        return destination

    def _collect_records(self) -> list[AuditRecord]:
        """Read current per-seed audits and attach the same identity fields used by DE."""
        rows = []
        for path in sorted(self.directory.glob(f"*/seed_*/{AUDIT_FILENAME}")):
            settings = json.loads((path.parent / RUN_FILENAME).read_text(encoding="utf-8"))
            result = json.loads(path.read_text(encoding="utf-8"))
            expected = {f"{k}x" for k in range(1, result["output_tokens"] // OUTPUT_TOKENS_PER_BLOCK + 1)}
            if set(result["checkpoints"]) != expected or result["output_tokens"] % OUTPUT_TOKENS_PER_BLOCK:
                raise ValueError(f"missing or invalid audit checkpoints: {path}")
            if result["complete"] != (result["output_tokens"] == settings["compute_multiplier_k"] * OUTPUT_TOKENS_PER_BLOCK):
                raise ValueError(f"audit completion does not match its budget: {path}")
            if (path.parent.name != f"seed_{settings['seed']}" or path.parent.parent.name != settings["problem_id"]
                    or self.directory.name != settings["arm"]):
                raise ValueError(f"audit identity does not match directory: {path}")
            identity = {
                "dataset": f"{settings['competition']}/{settings['year']}", "solver_model": settings["model"],
                **{key: settings[key] for key in ("experiment", "compute_multiplier_k", "inject_sketch_at",
                                                 "solution_index", "problem_id", "seed")},
            }
            for label, checkpoint in result["checkpoints"].items():
                rows.append(AuditRecord(**identity, checkpoint_multiplier_k=int(label.removesuffix("x")),
                                        correctness=checkpoint["correctness"]))
        return sorted(rows, key=lambda row: (row.problem_id, row.seed, row.checkpoint_multiplier_k))
