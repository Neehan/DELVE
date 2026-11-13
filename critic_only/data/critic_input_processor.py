"""Group continuations from the same 1× state into one critic input and success-rate target."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from datasets import Dataset, concatenate_datasets

from discovery_execution.audit.answer_grader import validate_grading_rule
from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CONTINUATION_EXPERIMENTS, CONVERSATION_FILENAME,
    META_FILENAME, OUTPUT_TOKENS_PER_BLOCK, SOLVE_PHASE,
)
from discovery_execution.harness.models import Problem, RunState
from critic_only.constants import DEFAULT_SEED, HASH_SCALE, INPUT_PREFIX_BLOCKS, CONTINUATION_SEEDS
from discovery_execution.processing.constants import MANIFEST_FILENAME, SPLIT_FILENAME
from discovery_execution.processing.storage import checkpoint_snapshot, file_digest
from critic_only.data.models import BranchOutcome, InputPrefix, PreparationConfig


def select_input_fork(metadata: dict[str, Any], horizon: int) -> int | None:
    """Return the 1× fork for continuation runs ending at the requested horizon.

    metadata identifies the experiment and fork; horizon is the requested final budget. Root runs provide source histories, not labels. Other forks and budgets return None.
    """
    if metadata["compute_multiplier_k"] != horizon or metadata["experiment"] not in CONTINUATION_EXPERIMENTS:
        return None
    fork = metadata["continue_at"]
    if not 1 <= fork < horizon:
        raise ValueError("continuation fork must precede the destination horizon")
    return fork if fork == INPUT_PREFIX_BLOCKS else None


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
    """Yield one pre-continuation input per source state and oracle condition.

    iter_inputs verifies common histories, hints, sampling settings, and final audits. Each input carries the independent branch outcomes used to estimate its success probability.
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
        """Yield serialized InputPrefix records grouped by source and condition, without exposing child reasoning or grades."""
        groups: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        identities = set()
        for name, text in zip(self.dataset["trajectory_id"], self.dataset["metadata"]):
            metadata = json.loads(text)
            fork = select_input_fork(metadata, self.config.horizon)
            if fork is None or metadata["seed"] not in CONTINUATION_SEEDS:
                continue
            problem_key = tuple(metadata[key] for key in ("competition", "year", "problem_id", "model"))
            source_key = (metadata["source_arm"], metadata["source_seed"], fork)
            identity = (*problem_key, *source_key, metadata["experiment"], metadata["seed"])
            if identity in identities:
                raise ValueError(f"duplicate saved trajectory: {name}")
            identities.add(identity)
            groups[problem_key + source_key].append((Path(name), metadata))
        for key in sorted(groups):
            for record in self._build_branch_inputs(groups[key]):
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
        """Reconstruct one source state and build its unaided and oracle examples.

        group contains paths and metadata with the same problem, source arm, source seed, and fork. Returns one InputPrefix per available condition. Oracle branches must agree on the sketch; its matching solution also supplies the unaided reference view.
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
        snapshot, request = self._read_checkpoint(source, INPUT_PREFIX_BLOCKS, False)
        problem = self._load_problem(metadata)
        if problem.statement not in snapshot["messages"][0]["content"]:
            raise ValueError(f"current problem statement differs from saved solver input: {source}")
        oracle_indices = {item["solution_index"] for _, item in group if item["experiment"] == "continue-oracle"}
        if len(oracle_indices) > 1:
            raise ValueError("oracle branches from one prefix must use the same sketch")
        solution_index = problem.select_solution_index(metadata["source_seed"], next(iter(oracle_indices), None))
        records = []
        for experiment in CONTINUATION_EXPERIMENTS:
            branches = sorted((item for item in group if item[1]["experiment"] == experiment), key=lambda item: item[1]["seed"])
            if not branches:
                continue
            if tuple(item[1]["seed"] for item in branches) != CONTINUATION_SEEDS:
                raise ValueError(f"expected continuation seeds {CONTINUATION_SEEDS}: {source}/{experiment}")
            outcomes, intervention = self._read_matching_branches(branches, parent, snapshot["messages"], problem.solutions[solution_index].sketch)
            oracle = experiment == "continue-oracle"
            messages = snapshot["messages"] + ([intervention] if intervention is not None else [])
            records.append(InputPrefix(
                prefix_id=f"{source}/checkpoint_{INPUT_PREFIX_BLOCKS}x/{experiment}", oracle=oracle,
                problem_key=f"{problem.competition}/{problem.year}/{problem.problem_id}", fork=INPUT_PREFIX_BLOCKS,
                remaining_blocks=self.config.horizon - INPUT_PREFIX_BLOCKS,
                split=self.dataset.select_columns(["split"])[self.indices[str(source)]]["split"], messages=messages,
                phase=SOLVE_PHASE if oracle else request["phase"],
                response_finished=False if oracle else request["finish_reason"] == "stop",
                solution_index=solution_index, reference_solution=problem.solutions[solution_index].solution, branches=outcomes))
        return records

    def _read_matching_branches(self, branches: list[tuple[Path, dict[str, Any]]], parent: dict[str, Any],
                                source_messages: list[dict[str, str]], sketch: str) -> tuple[list[BranchOutcome], dict[str, str] | None]:
        """Verify each child's source and intervention, returning outcomes and the exact injected message.

        branches supplies child paths and metadata; parent supplies sampling settings. source_messages and sketch define the expected input. Returns final outcomes and the shared hint message, if present. Child text is used only for validation.
        """
        outcomes = []
        intervention = None
        for path, metadata in branches:
            for name in ("temperature", "top_p", "top_k"):
                if metadata["backend"][name] != parent["backend"][name]:
                    raise ValueError(f"branch sampling differs from source: {path}")
            oracle = metadata["experiment"] == "continue-oracle"
            snapshot, _ = self._read_checkpoint(path, INPUT_PREFIX_BLOCKS + 1, oracle)
            self._validate_source_history(snapshot["messages"], source_messages, path)
            if oracle:
                candidate = self._validate_oracle_injection(snapshot["messages"], source_messages, sketch, path)
                if intervention is not None and intervention != candidate:
                    raise ValueError(f"oracle branches have different injection messages: {path}")
                intervention = candidate
            outcomes.append(self._read_branch_outcome(path, metadata, INPUT_PREFIX_BLOCKS))
        return outcomes, intervention

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
        validate_grading_rule(audit, str(path))
        if audit["complete"] is not True or audit["output_tokens"] != self.config.horizon * OUTPUT_TOKENS_PER_BLOCK:
            raise ValueError(f"incomplete branch audit: {path}")
        expected = [f"{k}x" for k in range(fork + 1, self.config.horizon + 1)]
        if set(audit["checkpoints"]) != set(expected):
            raise ValueError(f"missing or unexpected audit checkpoints: {path}")
        return BranchOutcome(seed=metadata["seed"],
                             final_correctness=audit["checkpoints"][expected[-1]]["correctness"]["score"],
                             path=str(path))
