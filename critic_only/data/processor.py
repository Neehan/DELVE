"""Select verified post-continuation histories and their terminal labels from shared trajectory shards."""

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
    """Select root runs and continuation forks with one observed block and a nonterminal future."""
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
    """Assign a stable fraction to a problem without depending on its branches or file order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest) / HASH_SCALE


def select_unaided_examples(dataset: Dataset) -> Dataset:
    """Select unaided histories for the baseline's training and all primary evaluations."""
    selected = dataset.select([index for index, oracle in enumerate(dataset["oracle"]) if not oracle])
    if not len(selected):
        raise ValueError("no unaided histories available")
    return selected


def select_training_problems(dataset: Dataset, fraction: float) -> Dataset:
    """Keep a nested, seed-1234 subset of whole problems, rounding the requested count down.

    Selection is independent of row order and the training seed, so model variants
    use the same problems and retain every available branch of each selected problem.
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
    """Yield checked post-continuation histories and matching labels through iter_inputs()."""

    def __init__(self, config: PreparationConfig) -> None:
        """Read a verified Parquet collection and index identities without materializing histories."""
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
        """Yield one observed branch history and its own terminal label, keeping source groups together."""
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
        """Read the exact serialized artifact preserved in its trajectory row."""
        columns = {META_FILENAME: "metadata", CONVERSATION_FILENAME: "conversation", AUDIT_FILENAME: "audit"}
        row = self.dataset[self.indices[str(path.parent)]]
        return json.loads(row[columns[path.name]])

    def _load_problem(self, metadata: dict[str, Any]) -> Problem:
        """Use the reference frozen during setup, never a subsequently edited dataset file."""
        key = f"{metadata['competition']}/{metadata['year']}/{metadata['problem_id']}"
        if key not in self.problems:
            self.problems[key] = Problem.model_validate_json(self.dataset[self.problem_indices[key]]["problem"])
        return self.problems[key]

    def _build_branch_inputs(self, group: list[tuple[Path, dict[str, Any]]]) -> list[InputPrefix]:
        """Validate the common source, then retain each branch's own post-continuation history."""
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
        """Read one block after the fork and pair that history with its own final correctness."""
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
                           split=self.dataset[self.indices[str(path)]]["split"], messages=snapshot["messages"],
                           phase=request["phase"], response_finished=request["finish_reason"] == "stop",
                           solution_index=solution_index, reference_solution=solution.solution, branch=outcome)

    def _validate_source_history(self, messages: list[dict[str, str]], prefix: list[dict[str, str]], path: Path) -> None:
        """Allow an unfinished assistant response to extend, but never alter any source text."""
        if (not prefix or len(messages) < len(prefix) or messages[:len(prefix) - 1] != prefix[:-1]
                or messages[len(prefix) - 1]["role"] != prefix[-1]["role"]
                or not messages[len(prefix) - 1]["content"].startswith(prefix[-1]["content"])):
            raise ValueError(f"branch does not preserve source prefix: {path}")

    def _validate_oracle_injection(self, messages: list[dict[str, str]], source: list[dict[str, str]],
                                   sketch: str, path: Path) -> dict[str, str]:
        """Check the saved injection while retaining the solver's response as critic input."""
        if len(messages) <= len(source):
            raise ValueError(f"oracle branch has no injection message: {path}")
        candidate = messages[len(source)]
        if messages[:len(source)] != source:
            raise ValueError(f"oracle branch changed the pre-injection history: {path}")
        if candidate["role"] != "user" or sketch not in candidate["content"]:
            raise ValueError(f"current sketch differs from saved intervention: {path}")
        return candidate

    def _read_checkpoint(self, path: Path, block: int, oracle: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        """Recover history and phase at the input boundary, excluding all later text and state."""
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
        """Require a complete audit and retain only the final answer grade as the training target."""
        audit = self._read_json(path / AUDIT_FILENAME)
        if audit["complete"] is not True or audit["output_tokens"] != self.config.horizon * OUTPUT_TOKENS_PER_BLOCK:
            raise ValueError(f"incomplete branch audit: {path}")
        expected = [f"{k}x" for k in range(fork + 1, self.config.horizon + 1)]
        if set(audit["checkpoints"]) != set(expected):
            raise ValueError(f"missing or unexpected audit checkpoints: {path}")
        return BranchOutcome(seed=metadata["seed"],
                             final_correctness=audit["checkpoints"][expected[-1]]["correctness"]["score"],
                             path=str(path))
