"""Compile audited DELVE branches once into portable, problem-disjoint Parquet splits."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import zstandard
from datasets import Dataset

from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CHECKPOINT_FILENAME, CONTINUATION_EXPERIMENTS, CONVERSATION_FILENAME,
    META_FILENAME, ORACLE_EXPERIMENTS, OUTPUT_TOKENS_PER_BLOCK,
)
from discovery_execution.harness.models import Problem, RunState
from critic_only.constants import DEFAULT_SEED, HASH_SCALE, MANIFEST_FILENAME, SPLITS, TEST_YEAR
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


class DatasetProcessor:
    """Validate audited branch histories and publish problem-disjoint splits through run()."""

    def __init__(self, config: PreparationConfig) -> None:
        """Keep preparation settings and the input checksums needed to reproduce this artifact."""
        self.config = config
        self.checksums: dict[str, str] = {}
        self.problems: dict[tuple[str, int], dict[str, Problem]] = {}

    def run(self) -> None:
        """Reject overwrites, group branches, and publish the full dataset only after validation."""
        if self.config.destination.exists():
            raise FileExistsError(self.config.destination)
        groups: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        root_runs = []
        identities = set()
        for path in sorted(self.config.outputs_directory.rglob(META_FILENAME)):
            metadata = self._read_json(path)
            if metadata["model"] != self.config.solver_model:
                continue
            fork = select_input_fork(metadata, self.config.horizon)
            if fork is None:
                continue
            problem_key = tuple(metadata[name] for name in ("competition", "year", "problem_id"))
            source_key = (metadata["source_arm"], metadata["source_seed"], fork) if fork else ()
            identity = (*problem_key, *source_key, metadata["experiment"], metadata["seed"])
            if identity in identities:
                raise ValueError(f"duplicate saved trajectory: {path}")
            identities.add(identity)
            if fork == 0:
                root_runs.append((path.parent, metadata))
            else:
                groups[problem_key + source_key].append((path.parent, metadata))
        records = []
        for path, metadata in root_runs:
            solution_index = metadata["solution_index"] if metadata["experiment"] in ORACLE_EXPERIMENTS else None
            records.append(self._build_input(path, metadata, solution_index))
        records.extend(record for group in groups.values() for record in self._build_branch_inputs(group))
        if not records:
            raise ValueError("no matching trajectories with a nonterminal critic input")
        self._publish(records)

    def _read_json(self, path: Path) -> Any:
        """Read and fingerprint exactly the bytes used, without mutating collection artifacts."""
        file_bytes = path.read_bytes()
        self.checksums[str(path.resolve())] = hashlib.sha256(file_bytes).hexdigest()
        if path.suffix == ".zst":
            file_bytes = zstandard.decompress(file_bytes)
        return json.loads(file_bytes)

    def _load_problem(self, metadata: dict[str, Any]) -> Problem:
        """Load each local reference partition once and keep its source checksum."""
        key = metadata["competition"], metadata["year"]
        if key not in self.problems:
            path = self.config.datasets_directory / key[0] / f"{key[1]}.jsonl"
            dataset_bytes = path.read_bytes()
            self.checksums[str(path.resolve())] = hashlib.sha256(dataset_bytes).hexdigest()
            rows = [Problem.model_validate_json(line) for line in dataset_bytes.splitlines() if line.strip()]
            if any((row.competition, row.year) != key for row in rows):
                raise ValueError(f"problem partition disagrees with filename: {path}")
            if len({row.problem_id for row in rows}) != len(rows):
                raise ValueError(f"duplicate problem IDs: {path}")
            self.problems[key] = {row.problem_id: row for row in rows}
        return self.problems[key][metadata["problem_id"]]

    def _build_branch_inputs(self, group: list[tuple[Path, dict[str, Any]]]) -> list[InputPrefix]:
        """Validate the common source, then retain each branch's own post-continuation history."""
        _, metadata = group[0]
        root = self.config.outputs_directory / metadata["competition"] / str(metadata["year"]) / metadata["model"]
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
                           split=self._split(key, problem.year), messages=snapshot["messages"],
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
        snapshot = self._read_json(path / CHECKPOINT_FILENAME.format(k=block))
        state = RunState.model_validate(self._read_json(path / CONVERSATION_FILENAME))
        if state.output_tokens < tokens:
            raise ValueError(f"checkpoint exceeds saved conversation progress: {path}")
        boundaries = [item for item in state.checkpoints if item.output_tokens == tokens]
        requests = [item for item in state.requests if item["output_tokens"] <= tokens]
        if len(boundaries) != 1 or not requests or requests[-1]["output_tokens"] != tokens:
            raise ValueError(f"conversation lacks the input checkpoint boundary: {path}")
        boundary = boundaries[0]
        messages = [dict(item) for item in state.messages[:boundary.message_count]]
        messages[-1]["content"] = messages[-1]["content"][:boundary.last_message_chars]
        if snapshot != {"messages": messages, "output_tokens": tokens}:
            raise ValueError(f"snapshot disagrees with conversation: {path}")
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
                             path=str(path.relative_to(self.config.outputs_directory)))

    def _split(self, key: str, year: int) -> Literal["train", "validation", "test"]:
        """Keep the held-out year untouched and split older problems before expanding prefixes."""
        if year >= TEST_YEAR:
            return "test"
        return "validation" if selection_fraction(key, self.config.split_seed) < self.config.validation_fraction else "train"

    def _publish(self, records: list[InputPrefix]) -> None:
        """Write shareable Parquet and a provenance manifest without publishing partial output."""
        destination = self.config.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "prepared"
            staging.mkdir()
            counts = {}
            for split in SPLITS:
                rows = [row.model_dump() for row in records if row.split == split]
                counts[split] = len(rows)
                if rows:
                    Dataset.from_list(cast(list[dict], rows)).to_parquet(str(staging / f"{split}.parquet"))
            manifest = {"settings": self.config.model_dump(mode="json"), "prefix_counts": counts,
                        "problem_count": len({row.problem_key for row in records}), "input_sha256": self.checksums}
            (staging / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
            staging.rename(destination)
