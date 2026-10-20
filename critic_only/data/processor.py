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
    AUDIT_FILENAME, CHECKPOINT_FILENAME, CONVERSATION_FILENAME, META_FILENAME, OUTPUT_TOKENS_PER_BLOCK,
)
from discovery_execution.harness.models import Problem, RunState
from critic_only.constants import CONDITIONS, HASH_SCALE, MANIFEST_FILENAME, SPLITS
from critic_only.data.models import BranchOutcome, PrefixRecord, PreparationConfig


def selection_fraction(key: str, seed: int) -> float:
    """Assign a stable fraction to a problem without depending on its branches or file order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest) / HASH_SCALE


def select_ordinary_examples(dataset: Dataset) -> Dataset:
    """Select ordinary histories for the baseline's training and all primary evaluations."""
    selected = dataset.select([index for index, oracle in enumerate(dataset["oracle"]) if not oracle])
    if not len(selected):
        raise ValueError("no ordinary histories available")
    return selected


class DatasetProcessor:
    """Run validates exact prefixes and audits, then atomically publishes immutable splits."""

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
        for path in sorted(self.config.outputs_directory.rglob(META_FILENAME)):
            metadata = self._read_json(path)
            if metadata["model"] != self.config.solver_model or metadata["experiment"] not in CONDITIONS:
                continue
            if metadata["compute_multiplier_k"] != self.config.horizon:
                continue
            key = tuple(metadata[name] for name in ("competition", "year", "problem_id", "source_arm", "source_seed", "continue_at"))
            groups[key].append((path.parent, metadata))
        if not groups:
            raise ValueError("no matching continuation trajectories")
        records = [record for group in groups.values() for record in self._build_records(group)]
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

    def _build_records(self, group: list[tuple[Path, dict[str, Any]]]) -> list[PrefixRecord]:
        """Create separate examples before and after the saved sketch intervention."""
        _, metadata = group[0]
        fork = metadata["continue_at"]
        if not 1 <= fork < self.config.horizon:
            raise ValueError("fork must precede the destination horizon")
        root = self.config.outputs_directory / metadata["competition"] / str(metadata["year"]) / metadata["model"]
        source = root / metadata["source_arm"] / metadata["problem_id"] / f"seed_{metadata['source_seed']}"
        parent = self._read_json(source / META_FILENAME)
        for name in ("competition", "year", "problem_id", "model"):
            if parent[name] != metadata[name]:
                raise ValueError(f"source identity mismatch: {source}")
        if parent["experiment"] != "unaided" or parent["seed"] != metadata["source_seed"]:
            raise ValueError(f"source is not the selected unaided seed: {source}")
        snapshot, request = self._source_state(source, fork)
        branches = {condition: [self._branch(path, item, parent, snapshot) for path, item in group
                                if item["experiment"] == condition] for condition in CONDITIONS}
        problem = self._load_problem(metadata)
        if problem.statement not in snapshot["messages"][0]["content"]:
            raise ValueError(f"current problem statement differs from saved solver input: {source}")
        oracle_indices = {item["solution_index"] for _, item in group if item["experiment"] == "continue-oracle"}
        if len(oracle_indices) > 1:
            raise ValueError("oracle branches from one prefix must use the same sketch")
        solution_index = problem.select_solution_index(metadata["source_seed"], next(iter(oracle_indices), None))
        solution = problem.solutions[solution_index]
        intervention = self._oracle_intervention(group, snapshot, solution.sketch)
        key = f"{problem.competition}/{problem.year}/{problem.problem_id}"
        records = []
        for condition in CONDITIONS:
            selected = branches[condition]
            if not selected:
                continue
            oracle = condition == "continue-oracle"
            messages = snapshot["messages"] + [intervention] if oracle else snapshot["messages"]
            records.append(PrefixRecord(prefix_id=f"{key}/{metadata['source_arm']}/{metadata['source_seed']}/{fork}/{condition}",
                            oracle=oracle, problem_key=key,
                            fork=fork, remaining_blocks=self.config.horizon - fork, split=self._split(key, problem.year),
                            messages=messages, phase="solve" if oracle else request["phase"],
                            response_finished=False if oracle else request["finish_reason"] == "stop", solution_index=solution_index,
                            reference_solution=solution.solution, branches=selected))
        return records

    def _oracle_intervention(self, group: list[tuple[Path, dict[str, Any]]], snapshot: dict[str, Any],
                             sketch: str) -> dict[str, str] | None:
        """Read the exact injection message and reject branches that disagree about their starting history."""
        intervention = None
        for path, item in group:
            if item["experiment"] == "continue-oracle":
                first = self._read_json(path / CHECKPOINT_FILENAME.format(k=item["continue_at"] + 1))
                if len(first["messages"]) <= len(snapshot["messages"]):
                    raise ValueError(f"oracle branch has no injection message: {path}")
                candidate = first["messages"][len(snapshot["messages"])]
                if first["messages"][:len(snapshot["messages"])] != snapshot["messages"]:
                    raise ValueError(f"oracle branch changed the pre-injection history: {path}")
                if candidate["role"] != "user" or sketch not in candidate["content"]:
                    raise ValueError(f"current sketch differs from saved intervention: {path}")
                if intervention is not None and candidate != intervention:
                    raise ValueError(f"oracle branches have different injection messages: {path}")
                intervention = candidate
        return intervention

    def _source_state(self, source: Path, fork: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Recover the phase and natural-ending flag at the fork, never from the parent's final state."""
        tokens = fork * OUTPUT_TOKENS_PER_BLOCK
        snapshot = self._read_json(source / CHECKPOINT_FILENAME.format(k=fork))
        state = RunState.model_validate(self._read_json(source / CONVERSATION_FILENAME))
        boundaries = [item for item in state.checkpoints if item.output_tokens == tokens]
        requests = [item for item in state.requests if item["output_tokens"] <= tokens]
        if len(boundaries) != 1 or not requests or requests[-1]["output_tokens"] != tokens:
            raise ValueError(f"source conversation lacks the fork boundary: {source}")
        boundary = boundaries[0]
        messages = [dict(item) for item in state.messages[:boundary.message_count]]
        messages[-1]["content"] = messages[-1]["content"][:boundary.last_message_chars]
        if snapshot != {"messages": messages, "output_tokens": tokens} or any(item["injected"] for item in requests):
            raise ValueError(f"source snapshot disagrees with unaided conversation: {source}")
        return snapshot, requests[-1]

    def _branch(self, path: Path, metadata: dict[str, Any], parent: dict[str, Any],
                source: dict[str, Any]) -> BranchOutcome:
        """Reject incomplete audits, sampling mismatches, and branches that do not share the source text."""
        for key in ("temperature", "top_p", "top_k"):
            if metadata["backend"][key] != parent["backend"][key]:
                raise ValueError(f"branch sampling differs from source: {path}")
        fork = metadata["continue_at"]
        snapshot = self._read_json(path / CHECKPOINT_FILENAME.format(k=fork + 1))
        if snapshot["output_tokens"] != (fork + 1) * OUTPUT_TOKENS_PER_BLOCK:
            raise ValueError(f"wrong first branch checkpoint boundary: {path}")
        messages = snapshot["messages"]
        prefix = source["messages"]
        if (not prefix or len(messages) < len(prefix) or messages[:len(prefix) - 1] != prefix[:-1]
                or messages[len(prefix) - 1]["role"] != prefix[-1]["role"]
                or not messages[len(prefix) - 1]["content"].startswith(prefix[-1]["content"])):
            raise ValueError(f"branch does not preserve source prefix: {path}")
        audit = self._read_json(path / AUDIT_FILENAME)
        if audit["complete"] is not True or audit["output_tokens"] != self.config.horizon * OUTPUT_TOKENS_PER_BLOCK:
            raise ValueError(f"incomplete branch audit: {path}")
        expected = [f"{k}x" for k in range(fork + 1, self.config.horizon + 1)]
        if set(audit["checkpoints"]) != set(expected):
            raise ValueError(f"missing or unexpected audit checkpoints: {path}")
        return BranchOutcome(seed=metadata["seed"],
                             outcomes=[audit["checkpoints"][k]["correctness"]["score"] for k in expected],
                             path=str(path.relative_to(self.config.outputs_directory)))

    def _split(self, key: str, year: int) -> Literal["train", "validation", "test"]:
        """Keep the held-out year untouched and split older problems before expanding prefixes."""
        if year >= self.config.test_year:
            return "test"
        return "validation" if selection_fraction(key, self.config.split_seed) < self.config.validation_fraction else "train"

    def _publish(self, records: list[PrefixRecord]) -> None:
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
