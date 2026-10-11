"""Load binary audited short and initial-sketch oracle runs for DE estimators."""

import logging
from pathlib import Path

from discovery_execution.audit.constants import COMPILED_AUDIT_FILENAME
from discovery_execution.audit.models import AuditRecord
from discovery_execution.frameworks.models import InputSelection, Observation
from discovery_execution.frameworks.utils.constants import MIN_AUDIT_SCORE, MIN_ORACLE_SEEDS, MIN_SHORT_SEEDS, ORACLE_ARM, SHORT_ARM, TOTAL_BUDGET


class DataLoader:
    """Load complete seed curves from explicit partitions; long unaided and delayed oracle arms are excluded."""

    def __init__(self, results: Path) -> None:
        """Inject the root containing trajectory outputs and compiled audits."""
        self.results = results

    def load(self, selection: InputSelection) -> list[Observation]:
        """Use every complete audited seed and report problems below DE's sampling thresholds."""
        observations = []
        skipped = []
        for competition in selection.competitions:
            for year in selection.years:
                dataset = f"{competition}/{year}"
                directory = self.results / dataset / selection.model
                short = self._read_runs(directory / SHORT_ARM)
                oracle = self._read_runs(directory / ORACLE_ARM)
                for problem in sorted(short.keys() | oracle.keys()):
                    short_runs, oracle_runs = short.get(problem, []), oracle.get(problem, [])
                    if len(short_runs) < MIN_SHORT_SEEDS or len(oracle_runs) < MIN_ORACLE_SEEDS:
                        skipped.append(f"{dataset}/{problem} (short={len(short_runs)}, oracle={len(oracle_runs)})")
                        continue
                    observations.append(Observation(
                        dataset=dataset, problem_id=problem, short_attempts=len(short_runs),
                        short_successes=sum(run[1] for run in short_runs), oracle_attempts=len(oracle_runs),
                        oracle_successes=self._cumulative_successes(oracle_runs),
                    ))
        if skipped:
            logging.warning("Need at least %s short and %s oracle seeds; skipped: %s",
                            MIN_SHORT_SEEDS, MIN_ORACLE_SEEDS, ", ".join(skipped))
        return observations

    def _read_runs(self, directory: Path) -> dict[str, list[dict[int, bool]]]:
        """Reject mixed identities and duplicate rows; exclude incomplete seeds from fitting."""
        horizon = 1 if directory.name == SHORT_ARM else TOTAL_BUDGET
        path = directory / COMPILED_AUDIT_FILENAME
        runs: dict[str, dict[int, dict[int, bool]]] = {}
        solution_indices: dict[tuple[str, int], int] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = AuditRecord.model_validate_json(line)
                arm = f"unaided-{row.compute_multiplier_k}x" if row.experiment == "unaided" else f"oracle-at-{row.inject_sketch_at}x-execution"
                expected_directory = self.results / row.dataset / row.solver_model / arm
                if expected_directory != directory or row.compute_multiplier_k != horizon:
                    raise ValueError(f"audit identity does not match {path}")
                identity = (row.problem_id, row.seed)
                if solution_indices.setdefault(identity, row.solution_index) != row.solution_index:
                    raise ValueError(f"solution index changes within a trajectory in {path}")
                checkpoint = row.checkpoint_multiplier_k
                run = runs.setdefault(row.problem_id, {}).setdefault(row.seed, {})
                if checkpoint in run:
                    raise ValueError(f"duplicate audit checkpoint in {path}")
                run[checkpoint] = row.correctness.score >= MIN_AUDIT_SCORE
        expected = set(range(1, horizon + 1))
        complete = {problem: [run for run in seeds.values() if run.keys() == expected]
                    for problem, seeds in runs.items()}
        incomplete = sum(len(runs[problem]) - len(seeds) for problem, seeds in complete.items())
        if incomplete:
            logging.warning("Excluded %s incomplete seeds from %s", incomplete, path)
        return complete

    def _cumulative_successes(self, runs: list[dict[int, bool]]) -> list[int]:
        """Retain success once observed, matching DE's cumulative execution convention."""
        counts = [0] * TOTAL_BUDGET
        for run in runs:
            solved = False
            for checkpoint in range(1, TOTAL_BUDGET + 1):
                solved = solved or run[checkpoint]
                counts[checkpoint - 1] += solved
        return counts
