"""Pair each root trajectory’s analysis prefix with grades from its reconstructed checkpoint table."""

import hashlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from datasets import Dataset, concatenate_datasets
from math_verify.errors import TimeoutException

from discovery_execution.audit.answer_grader import AnswerGrader, validate_grading_rule
from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CONVERSATION_FILENAME,
    META_FILENAME, OUTPUT_TOKENS_PER_BLOCK, DEFAULT_BLOCKS,
)
from discovery_execution.harness.models import Problem, RunState
from critic_only.constants import DEFAULT_SEED, HASH_SCALE, INPUT_PREFIX_BLOCKS, TRAJECTORY_SEEDS
from discovery_execution.processing.constants import MANIFEST_FILENAME, SPLIT_FILENAME
from discovery_execution.processing.checkpoint_reconstructor import CheckpointReconstructor
from discovery_execution.processing.models import GradedCheckpoint
from discovery_execution.processing.storage import checkpoint_snapshot, file_digest
from critic_only.data.models import BranchOutcome, InputPrefix, PreparationConfig


def is_input_trajectory(metadata: dict[str, Any], horizon: int) -> bool:
    """Return whether metadata describes an eligible root run at the original generation horizon.

    Only unaided runs and oracle-at-zero runs supply individual 1× inputs. Continuation arms are excluded.
    """
    if metadata["compute_multiplier_k"] != horizon:
        return False
    experiment = metadata["experiment"]
    return experiment == "unaided" or (experiment == "oracle-execution" and metadata["inject_sketch_at"] == 0)


def selection_fraction(key: str, seed: int) -> float:
    """Map a problem key to a reproducible number for selecting problem subsets.

    Args:
        key: Stable problem-selection key; callers use different prefixes for training and oracle subsets.
        seed: Seed used with the key to produce a reproducible hash.

    Returns:
        float: Deterministic fraction in [0, 1), used to rank or select whole problems.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest) / HASH_SCALE


def select_unaided_examples(dataset: Dataset) -> Dataset:
    """Keep only trajectories where the solver received no oracle sketch.

    Args:
        dataset: Example dataset containing an oracle boolean column.

    Returns:
        Dataset: Rows without solver-visible sketch injections; raises when no unaided rows remain.
    """
    selected = dataset.select([index for index, oracle in enumerate(dataset["oracle"]) if not oracle])
    if not len(selected):
        raise ValueError("no unaided histories available")
    return selected


def select_training_problems(dataset: Dataset, fraction: float) -> Dataset:
    """Choose a reproducible fraction of training problems and retain all their trajectories.

    Selection is independent of row order and the training seed, so model variants use the same problems and retain every available trajectory of each selected problem.

    Args:
        dataset: Training examples containing problem_key; all trajectories of a selected problem are retained.
        fraction: Requested fraction of distinct training problems, in (0, 1].

    Returns:
        Dataset: Deterministic nested subset selected with DEFAULT_SEED; raises if rounding selects no problems.
    """
    if not 0 < fraction <= 1:
        raise ValueError("training problem fraction must be in (0, 1]")
    problem_keys = list(dataset["problem_key"])
    ranked = sorted(set(problem_keys), key=lambda key: (selection_fraction(f"training:{key}", DEFAULT_SEED), key))
    count = int(len(ranked) * fraction)
    if count == 0:
        raise ValueError("training problem fraction selects no problems; increase it")
    selected = set(ranked[:count])
    return dataset.select([index for index, key in enumerate(problem_keys) if key in selected])


class CriticInputProcessor:
    """Yield one 1× analysis history and future binary labels per root trajectory.

    iter_inputs checks identities, saved hints and audits. References follow the saved solution selection or seed round robin.
    """

    def __init__(self, config: PreparationConfig, checkpoints: CheckpointReconstructor) -> None:
        """Check the processed files and index trajectory IDs for later lookup.

        Args:
            config: Processed model directory and analysis horizon.
            checkpoints: Reconstructs and grades the finer analysis boundaries.

        Returns:
            None. Verifies source checksums and indexes saved trajectories and problems for iter_inputs.
        """
        self.config = config
        self.checkpoint_builder = checkpoints
        self.checkpoint_rows: list[dict[str, Any]] = []
        self.manifest = json.loads((config.source_directory / MANIFEST_FILENAME).read_text())
        if file_digest(config.source_directory.parent / SPLIT_FILENAME) != self.manifest["split_sha256"]:
            raise ValueError("problem split assignments changed")
        datasets = []
        for name, digest in self.manifest["trajectories"].items():
            path = config.source_directory / name
            if file_digest(path) != digest:
                raise ValueError(f"processed trajectory shard changed: {path}")
            datasets.append(Dataset.from_parquet(str(path)))
        self.dataset = concatenate_datasets(datasets)
        identities = list(self.dataset["trajectory_id"])
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate saved trajectory identities")
        self.indices = {name: index for index, name in enumerate(identities)}
        self.problem_indices = {key: index for index, key in enumerate(self.dataset["problem_key"])}
        self.problems: dict[str, Problem] = {}

    def iter_inputs(self) -> Iterator[dict[str, Any]]:
        """Yield individual histories for seeds 0–7; duplicate problem/condition/seed identities raise an error."""
        self.checkpoint_rows.clear()
        identities = set()
        for name, text in zip(self.dataset["trajectory_id"], self.dataset["metadata"]):
            metadata = json.loads(text)
            if not is_input_trajectory(metadata, DEFAULT_BLOCKS) or metadata["seed"] not in TRAJECTORY_SEEDS:
                continue
            identity = tuple(metadata[key] for key in ("competition", "year", "problem_id", "model", "experiment", "seed"))
            if identity in identities:
                raise ValueError(f"duplicate saved trajectory: {name}")
            identities.add(identity)
            try:
                yield self._build_input(Path(name), metadata).model_dump()
            except (ValueError, IndexError, KeyError, TimeoutException) as error:
                raise ValueError(f"invalid trajectory {name}: {error}") from error

    def _read_json(self, path: Path) -> Any:
        """Read saved run metadata, conversation, or audit JSON from a Parquet row.

        Args:
            path: Logical saved artifact path: trajectory ID followed by meta.json, conversation.json.zst, or audit.json.

        Returns:
            Any: Decoded JSON object from the corresponding Parquet column; no raw artifact file is opened.
        """
        columns = {META_FILENAME: "metadata", CONVERSATION_FILENAME: "conversation", AUDIT_FILENAME: "audit"}
        column = columns[path.name]
        row = self.dataset.select_columns([column])[self.indices[str(path.parent)]]
        return json.loads(row[column])

    def _load_problem(self, metadata: dict[str, Any]) -> Problem:
        """Read the problem and solutions stored during setup, rather than the current dataset files.

        Args:
            metadata: Run metadata identifying the competition, year, and problem ID.

        Returns:
            Problem: Dataset record frozen during setup, including its statement and reference solutions.
        """
        key = f"{metadata['competition']}/{metadata['year']}/{metadata['problem_id']}"
        if key not in self.problems:
            self.problems[key] = Problem.model_validate_json(self.dataset.select_columns(["problem"])[self.problem_indices[key]]["problem"])
        return self.problems[key]

    def _build_input(self, path: Path, metadata: dict[str, Any]) -> InputPrefix:
        """Pair a reconstructed 1× input with later grades and the solution corresponding to its sketch.

        Args:
            path: Saved root trajectory ID, used to read the conversation, audit and split.
            metadata: Run identity, condition, seed and saved oracle solution index.

        Returns:
            InputPrefix: Observed messages and matching reference, with 2×–H× grades stored separately in branch. Later reasoning is never copied into the input.
        """
        oracle = metadata["experiment"] == "oracle-execution"
        problem = self._load_problem(metadata)
        state, checkpoints = self._graded_checkpoints(path, problem)
        boundary = checkpoints[INPUT_PREFIX_BLOCKS - 1]
        snapshot = checkpoint_snapshot(state, boundary)
        if state.injected != oracle or any(request["injected"] != oracle for request in state.requests):
            raise ValueError(f"input checkpoint lacks the expected oracle state: {path}")
        outcome = BranchOutcome(seed=metadata["seed"], path=str(path),
            checkpoint_correctness=[item.correctness for item in checkpoints[INPUT_PREFIX_BLOCKS:self.config.horizon]])
        if problem.statement not in snapshot["messages"][0]["content"]:
            raise ValueError(f"current problem statement differs from saved solver input: {path}")
        saved_index = metadata["solution_index"] if oracle else None
        solution_index = problem.select_solution_index(metadata["seed"], saved_index)
        solution = problem.solutions[solution_index]
        first_response = next(index for index, message in enumerate(snapshot["messages"]) if message["role"] == "assistant")
        if oracle and not any(message["role"] == "user" and solution.sketch in message["content"]
                              for message in snapshot["messages"][:first_response]):
            raise ValueError(f"current sketch differs from saved intervention: {path}")
        return InputPrefix(prefix_id=f"{path}/checkpoint_{boundary.output_tokens}tokens", oracle=oracle,
            problem_key=f"{problem.competition}/{problem.year}/{problem.problem_id}",
            remaining_blocks=self.config.horizon - INPUT_PREFIX_BLOCKS,
            split=self.dataset.select_columns(["split"])[self.indices[str(path)]]["split"], messages=snapshot["messages"],
            solution_index=solution_index, reference_solution=solution.solution, branch=outcome)

    def _graded_checkpoints(self, path: Path, problem: Problem) -> tuple[RunState, list[GradedCheckpoint]]:
        """Read one source state, validate its audit, and retain every derived checkpoint.

        Args:
            path: Saved trajectory identity used to read the state and audit.
            problem: Frozen problem record supplying the verified answer.

        Returns:
            tuple: Original state and graded analysis boundaries, including checkpoints beyond training's horizon.
        """
        state = RunState.model_validate(self._read_json(path / CONVERSATION_FILENAME))
        audit = self._read_json(path / AUDIT_FILENAME)
        validate_grading_rule(audit, str(path))
        metadata = self._read_json(path / META_FILENAME)
        source_budget = metadata["compute_multiplier_k"] * OUTPUT_TOKENS_PER_BLOCK
        if audit["complete"] is not True or audit["output_tokens"] != source_budget or state.output_tokens != source_budget:
            raise ValueError(f"incomplete branch audit: {path}")
        if source_budget < self.config.horizon * self.checkpoint_builder.interval_tokens:
            raise ValueError("analysis horizon exceeds saved generation budget")
        checkpoints = self.checkpoint_builder.reconstruct(state, audit, AnswerGrader(problem.final_answer))
        self.checkpoint_rows.append({"trajectory_id": str(path), "checkpoints": [item.model_dump() for item in checkpoints]})
        return state, checkpoints
