"""Build a critic input and final correctness label from each saved solver branch.

For example, a branch starting at 2x supplies its history through 3x and its final grade at 5x. Oracle inputs include the actual sketch and the following solver response. The matching reference solution is attached separately for tokenization; later reasoning and grades never enter the prompt.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from datasets import Dataset, concatenate_datasets

from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CONTINUATION_EXPERIMENTS, CONVERSATION_FILENAME,
    META_FILENAME, ORACLE_EXPERIMENTS, OUTPUT_TOKENS_PER_BLOCK,
)
from discovery_execution.harness.models import Problem, RunState
from critic_only.constants import DEFAULT_SEED, HASH_SCALE
from discovery_execution.processing.constants import MANIFEST_FILENAME, SPLIT_FILENAME
from discovery_execution.processing.storage import checkpoint_snapshot, file_digest
from critic_only.data.models import BranchOutcome, InputPrefix, PreparationConfig


def select_input_fork(metadata: dict[str, Any], horizon: int) -> int | None:
    """Find a run's branch point if observing one more block still leaves a future to predict.

    Args:
        metadata: Saved run metadata identifying the experiment, total budget, fork, and sketch injection point.
        horizon: Required final rollout budget in blocks, not the remaining budget after the critic input.

    Returns:
        int | None: Fork block for an eligible run, or None when its budget/experiment is excluded or no future block remains.
    """
    if metadata["compute_multiplier_k"] != horizon:
        return None
    experiment = metadata["experiment"]
    if experiment in CONTINUATION_EXPERIMENTS:
        fork = metadata["continue_at"]
        if not 1 <= fork < horizon:
            raise ValueError("continuation fork must precede the destination horizon")
    elif experiment == "unaided" or (experiment == "oracle-execution" and metadata["inject_sketch_at"] == 0):
        fork = 0
    else:
        return None
    return fork if fork + 1 < horizon else None


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
    """Choose a reproducible fraction of training problems and retain all their branches.

    Selection is independent of row order and the training seed, so model variants use the same problems and retain every available branch of each selected problem.

    Args:
        dataset: Training examples containing problem_key; all branches of a selected problem are retained.
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
    """Select the visible history and final label for each critic example.

    iter_inputs yields one record per branch. Helpers reconstruct the checkpoint, verify the original source history and sketch, and select its matching reference solution. Each branch keeps its own history and final label.
    """

    def __init__(self, config: PreparationConfig) -> None:
        """Check the processed files and index trajectory IDs for later lookup.

        Args:
            config: Processed model directory and requested final rollout horizon.

        Returns:
            None. Verifies source checksums and indexes saved trajectories and problems for iter_inputs.
        """
        self.config = config
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
        """Yield each eligible branch's partial history, matching reference, and final label.

        Returns:
            Iterator[dict[str, Any]]: Serialized InputPrefix records, one per eligible observed branch, each with its own final binary label.
        """
        groups: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        identities = set()
        for name, text in zip(self.dataset["trajectory_id"], self.dataset["metadata"]):
            metadata = json.loads(text)
            fork = select_input_fork(metadata, self.config.horizon)
            if fork is None:
                continue
            problem_key = tuple(metadata[key] for key in ("competition", "year", "problem_id"))
            source_key = (metadata["source_arm"], metadata["source_seed"], fork) if fork else ()
            identity = (*problem_key, *source_key, metadata["experiment"], metadata["seed"])
            if identity in identities:
                raise ValueError(f"duplicate saved trajectory: {name}")
            identities.add(identity)
            if fork == 0:
                solution_index = metadata["solution_index"] if metadata["experiment"] in ORACLE_EXPERIMENTS else None
                yield self._build_input(Path(name), metadata, solution_index).model_dump()
            else:
                groups[problem_key + source_key].append((Path(name), metadata))
        for group in groups.values():
            for record in self._build_branch_inputs(group):
                yield record.model_dump()

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

    def _build_branch_inputs(self, group: list[tuple[Path, dict[str, Any]]]) -> list[InputPrefix]:
        """Build inputs from related branches after checking their shared source and sketch selection.

        Args:
            group: Continuation paths and metadata sharing the same problem, source run, source seed, and fork block.

        Returns:
            list[InputPrefix]: One observed-history record per branch, using the group's matching solution selection.
        """
        _, metadata = group[0]
        root = Path(metadata["competition"]) / str(metadata["year"]) / metadata["model"]
        source = root / metadata["source_arm"] / metadata["problem_id"] / f"seed_{metadata['source_seed']}"
        parent = self._read_json(source / META_FILENAME)
        for name in ("competition", "year", "problem_id", "model"):
            if parent[name] != metadata[name]:
                raise ValueError(f"source identity mismatch: {source}")
        if parent["experiment"] != "unaided" or parent["seed"] != metadata["source_seed"]:
            raise ValueError(f"source is not the selected unaided seed: {source}")
        snapshot, _ = self._read_checkpoint(source, metadata["continue_at"], False)
        problem = self._load_problem(metadata)
        oracle_indices = {item["solution_index"] for _, item in group if item["experiment"] == "continue-oracle"}
        if len(oracle_indices) > 1:
            raise ValueError("oracle branches from one prefix must use the same sketch")
        solution_index = problem.select_solution_index(metadata["source_seed"], next(iter(oracle_indices), None))
        records = []
        intervention = None
        for path, branch_metadata in group:
            for name in ("temperature", "top_p", "top_k"):
                if branch_metadata["backend"][name] != parent["backend"][name]:
                    raise ValueError(f"branch sampling differs from source: {path}")
            record = self._build_input(path, branch_metadata, solution_index)
            self._validate_source_history(record.messages, snapshot["messages"], path)
            if record.oracle:
                candidate = self._validate_oracle_injection(record.messages, snapshot["messages"], problem.solutions[solution_index].sketch, path)
                if intervention is not None and intervention != candidate:
                    raise ValueError(f"oracle branches have different injection messages: {path}")
                intervention = candidate
            records.append(record)
        return records

    def _build_input(self, path: Path, metadata: dict[str, Any], solution_index: int | None) -> InputPrefix:
        """Pair the history one block after the fork with that branch's final answer grade.

        Args:
            path: Saved trajectory ID relative to the original outputs root.
            metadata: Run metadata identifying its fork, oracle condition, seed, and final budget.
            solution_index: Explicit matching solution index, or None to apply the problem's seed-based round robin.

        Returns:
            InputPrefix: History through one block after the fork, matching reference, remaining budget, split, and final label.
        """
        fork = metadata["continue_at"] if metadata["experiment"] in CONTINUATION_EXPERIMENTS else 0
        oracle = metadata["experiment"] in ORACLE_EXPERIMENTS
        outcome = self._read_branch_outcome(path, metadata, fork)
        snapshot, request = self._read_checkpoint(path, fork + 1, oracle)
        problem = self._load_problem(metadata)
        if problem.statement not in snapshot["messages"][0]["content"]:
            raise ValueError(f"current problem statement differs from saved solver input: {path}")
        solution_index = problem.select_solution_index(metadata["seed"], solution_index)
        solution = problem.solutions[solution_index]
        if oracle and not fork:
            if not any(message["role"] == "user" and solution.sketch in message["content"] for message in snapshot["messages"]):
                raise ValueError(f"current sketch differs from saved intervention: {path}")
        key = f"{problem.competition}/{problem.year}/{problem.problem_id}"
        return InputPrefix(prefix_id=f"{outcome.path}/checkpoint_{fork + 1}x", oracle=oracle, problem_key=key,
                           fork=fork, remaining_blocks=self.config.horizon - fork - 1,
                           split=self.dataset.select_columns(["split"])[self.indices[str(path)]]["split"], messages=snapshot["messages"],
                           phase=request["phase"], response_finished=request["finish_reason"] == "stop",
                           solution_index=solution_index, reference_solution=solution.solution, branch=outcome)

    def _validate_source_history(self, messages: list[dict[str, str]], prefix: list[dict[str, str]], path: Path) -> None:
        """Reject a branch that changes the history it was supposed to continue.

        Args:
            messages: Observed branch history after continuation.
            prefix: Source history at the fork; its final assistant message may be extended by an unaided branch.
            path: Trajectory identifier included in validation errors.

        Returns:
            None. Raises ValueError if the branch changes or omits any source-prefix text.
        """
        if (not prefix or len(messages) < len(prefix) or messages[:len(prefix) - 1] != prefix[:-1]
                or messages[len(prefix) - 1]["role"] != prefix[-1]["role"]
                or not messages[len(prefix) - 1]["content"].startswith(prefix[-1]["content"])):
            raise ValueError(f"branch does not preserve source prefix: {path}")

    def _validate_oracle_injection(self, messages: list[dict[str, str]], source: list[dict[str, str]],
                                   sketch: str, path: Path) -> dict[str, str]:
        """Check that the sketch follows the unchanged source history and matches the chosen solution.

        Args:
            messages: Observed oracle branch history, including the injected sketch and subsequent response.
            source: Unmodified source history at the fork.
            sketch: Sketch text belonging to the selected reference solution.
            path: Trajectory identifier included in validation errors.

        Returns:
            dict[str, str]: The saved user message containing the oracle sketch; raises if the injection or source history differs.
        """
        if len(messages) <= len(source):
            raise ValueError(f"oracle branch has no injection message: {path}")
        candidate = messages[len(source)]
        if messages[:len(source)] != source:
            raise ValueError(f"oracle branch changed the pre-injection history: {path}")
        if candidate["role"] != "user" or sketch not in candidate["content"]:
            raise ValueError(f"current sketch differs from saved intervention: {path}")
        return candidate

    def _read_checkpoint(self, path: Path, block: int, oracle: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reconstruct the history and solver phase at the requested block boundary.

        Args:
            path: Saved trajectory ID whose conversation contains checkpoint boundaries.
            block: Absolute checkpoint block to reconstruct, measured from the original problem start.
            oracle: Whether the reconstructed history must contain a solver-visible sketch injection.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: Snapshot with messages/output_tokens, and the last request record at that boundary.
        """
        tokens = block * OUTPUT_TOKENS_PER_BLOCK
        state = RunState.model_validate(self._read_json(path / CONVERSATION_FILENAME))
        if state.output_tokens < tokens:
            raise ValueError(f"checkpoint exceeds saved conversation progress: {path}")
        boundaries = [item for item in state.checkpoints if item.output_tokens == tokens]
        requests = [item for item in state.requests if item["output_tokens"] <= tokens]
        if len(boundaries) != 1 or not requests or requests[-1]["output_tokens"] != tokens:
            raise ValueError(f"conversation lacks the input checkpoint boundary: {path}")
        snapshot = checkpoint_snapshot(state, boundaries[0])
        messages = snapshot["messages"]
        if messages[-1]["role"] != "assistant" or requests[-1]["injected"] != oracle:
            raise ValueError(f"input checkpoint lacks the expected solver response or oracle state: {path}")
        if not oracle and any(item["injected"] for item in requests):
            raise ValueError(f"unaided history contains a sketch injection: {path}")
        return snapshot, requests[-1]

    def _read_branch_outcome(self, path: Path, metadata: dict[str, Any], fork: int) -> BranchOutcome:
        """Read the final answer grade, rejecting incomplete or inconsistent audits.

        Args:
            path: Saved trajectory ID whose audit supplies the final label.
            metadata: Run metadata providing the continuation seed.
            fork: Block where this branch starts; determines the expected audited checkpoints.

        Returns:
            BranchOutcome: Branch seed, trajectory ID, and correctness of the last answer at the configured final horizon.
        """
        audit = self._read_json(path / AUDIT_FILENAME)
        if audit["complete"] is not True or audit["output_tokens"] != self.config.horizon * OUTPUT_TOKENS_PER_BLOCK:
            raise ValueError(f"incomplete branch audit: {path}")
        expected = [f"{k}x" for k in range(fork + 1, self.config.horizon + 1)]
        if set(audit["checkpoints"]) != set(expected):
            raise ValueError(f"missing or unexpected audit checkpoints: {path}")
        return BranchOutcome(seed=metadata["seed"],
                             final_correctness=audit["checkpoints"][expected[-1]]["correctness"]["score"],
                             path=str(path))
