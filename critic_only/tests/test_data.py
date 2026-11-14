"""Verify exact branch histories, reference selection, fixed splits, and cached input identity."""

import hashlib
import json
import shutil
from typing import Any

import pytest
import zstandard
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from critic_only.constants import CRITIC_TARGET, TRAJECTORY_SEEDS
from critic_only.data.models import PreparationConfig
from critic_only.data.dataset_loader import input_signature
from critic_only.data.storage import CriticInputStorage
from critic_only.data.critic_input_processor import CriticInputProcessor, is_input_trajectory, select_training_problems
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.trajectory_processor import TrajectoryProcessor
from discovery_execution.processing.problem_splits import ProblemSplits


def compile_fixture(config: CollectionFixture) -> CriticInputProcessor:
    """Compile real raw artifacts, then select critic inputs from the shared Parquet collection."""
    splits = ProblemSplits(config.datasets_directory, config.destination.parent, 1)
    TrajectoryProcessor(config.outputs_directory, config.destination, config.solver_model, splits).compile_trajectories("fixture", TRAJECTORY_SEEDS)
    return CriticInputProcessor(PreparationConfig(source_directory=config.destination, horizon=config.horizon))


def test_exact_fixed_splits(prepared_inputs: CollectionFixture) -> None:
    """Repeated processing reuses exact assignments and rejects changed problem identities."""
    folder = prepared_inputs.destination.parent / "processed"
    splits = ProblemSplits(prepared_inputs.datasets_directory, folder, 1)
    splits.save_assignments()
    assert list(splits.assignments.values()).count("train") == 1
    assert list(splits.assignments.values()).count("validation") == 1
    assert list(splits.assignments.values()).count("test") == 1
    assert ProblemSplits(prepared_inputs.datasets_directory, folder, 1).assignments == splits.assignments
    for path in prepared_inputs.datasets_directory.rglob("*.jsonl"):
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    recreated = ProblemSplits(prepared_inputs.datasets_directory, folder.parent / "recreated", 1)
    assert recreated.assignments == splits.assignments
    with pytest.raises(ValueError, match="size differs"):
        ProblemSplits(prepared_inputs.datasets_directory, folder, 2)


def test_training_subsets_keep_problem_groups() -> None:
    """Training fractions select nested problem sets and retain every branch of each chosen problem."""
    data = Dataset.from_list([{"problem_key": f"p-{problem}", "branch": branch} for problem in range(8) for branch in range(4)])
    small, large = select_training_problems(data, 0.25), select_training_problems(data, 0.5)
    assert len(small) == 8 and len(large) == 16
    assert set(small["problem_key"]) <= set(large["problem_key"])
    with pytest.raises(ValueError):
        select_training_problems(data, 0)


def test_individual_collection(prepared_inputs: CollectionFixture, tokenizer: PreTrainedTokenizerFast,
                               encoding_settings: dict[str, Any]) -> None:
    """Keep each root seed's observed history and own final grade; exclude continuation arms and later text."""
    processor = compile_fixture(prepared_inputs)
    rows = list(processor.iter_inputs())
    assert len(rows) == len({row["prefix_id"] for row in rows}) == 32
    for row in rows:
        assert "continue" not in row["branch"]["path"]
        assert row["branch"]["final_correctness"] == row["branch"]["seed"] % 2
        assert row["remaining_blocks"] == 2
        assert "FINAL FUTURE" not in json.dumps(row["messages"])
        assert row["messages"][-1]["content"] == f"Thinking seed {row['branch']['seed']}"
        assert (row["split"] == "test") == ("/2026/" in row["problem_key"])
    settings = {**encoding_settings, "target": CRITIC_TARGET}
    formatter = PrefixTokenizer(tokenizer)
    encoded = Dataset.from_list([formatter.encode(row, settings) for row in rows])
    storage = CriticInputStorage(prepared_inputs.destination)
    shards = storage.save_inputs(encoded, tokenizer, settings)
    assert len(storage.load_splits(("test",), storage.load_metadata()["tokenization"])["test"]) == 16
    for name, digest in shards.items():
        for row in storage.read_chunk(name, digest).to_list():
            assert row["labels"] == row["branch"]["final_correctness"]
            assert len(row["input_ids"]) == len(row["solver_mask"])
            assert row["solver_mask"][-1] and row["privileged_solver_mask"][-1]
    with pytest.raises(FileExistsError):
        storage.save_inputs(encoded, tokenizer, settings)


def test_future_outcomes_do_not_change_input_tokens(prepared_inputs: CollectionFixture,
    tokenizer: PreTrainedTokenizerFast, encoding_settings: dict[str, Any]) -> None:
    """Changing the terminal outcome changes only the target, never prompt tokens, masks or identity."""
    row = next(compile_fixture(prepared_inputs).iter_inputs())
    formatter = PrefixTokenizer(tokenizer)
    before = formatter.encode(row, encoding_settings)
    row["branch"]["final_correctness"] = 1 - row["branch"]["final_correctness"]
    after = formatter.encode(row, encoding_settings)
    assert before["labels"] != after["labels"]
    for key in ("prefix_id", "input_ids", "privileged_input_ids", "solver_mask", "privileged_solver_mask"):
        assert before[key] == after[key]


@pytest.mark.parametrize("saved_index", [0, 1])
def test_saved_solution_overrides_round_robin(prepared_inputs: CollectionFixture, saved_index: int) -> None:
    """Oracle references follow saved hint selection; unaided references follow their own root seed."""
    for path in prepared_inputs.datasets_directory.rglob("*.jsonl"):
        problems = [json.loads(line) for line in path.read_text().splitlines()]
        for problem in problems:
            problem["solutions"].append({"source": "ai_generated", "solution": "Alternate proof.", "sketch": "Count the complement."})
        path.write_text("".join(json.dumps(problem) + "\n" for problem in problems))
    for path in prepared_inputs.outputs_directory.rglob("meta.json"):
        metadata = json.loads(path.read_text())
        if metadata["experiment"] != "oracle-execution":
            continue
        metadata["solution_index"] = saved_index
        path.write_text(json.dumps(metadata))
        if saved_index:
            for artifact in path.parent.glob("*.zst"):
                data = json.loads(zstandard.decompress(artifact.read_bytes()))
                for message in data["messages"]:
                    message["content"] = message["content"].replace("Use symmetry.", "Count the complement.")
                artifact.write_bytes(zstandard.compress(json.dumps(data).encode()))
    for row in compile_fixture(prepared_inputs).iter_inputs():
        expected = saved_index if row["oracle"] else row["branch"]["seed"] % 2
        assert row["solution_index"] == expected
        assert row["reference_solution"] == ("Alternate proof." if expected else "The answer is one.")


def test_later_seeds_ignored_before_artifacts(prepared_inputs: CollectionFixture) -> None:
    """Incomplete seeds 8–15 must not affect preparation, while every selected root stays independent."""
    for path in list(prepared_inputs.outputs_directory.rglob("meta.json")):
        metadata = json.loads(path.read_text())
        if metadata["experiment"] not in ("unaided", "oracle-execution") or metadata["seed"] != 0:
            continue
        for seed in range(8, 16):
            directory = path.parent.with_name(f"seed_{seed}")
            directory.mkdir()
            (directory / "meta.json").write_text(json.dumps({**metadata, "seed": seed}))
    assert len(list(compile_fixture(prepared_inputs).iter_inputs())) == 32


def test_duplicate_root_seed_rejected(prepared_inputs: CollectionFixture) -> None:
    """Two paths for one problem/condition/seed cannot silently duplicate training data."""
    source = next(path.parent for path in prepared_inputs.outputs_directory.rglob("meta.json")
                  if json.loads(path.read_text())["experiment"] == "unaided")
    shutil.copytree(source, source.with_name("duplicate"))
    with pytest.raises(ValueError, match="duplicate saved trajectory"):
        list(compile_fixture(prepared_inputs).iter_inputs())


@pytest.mark.parametrize("damage", ["grading_rule", "audit", "sketch", "injected"])
def test_invalid_root_rejected(prepared_inputs: CollectionFixture, damage: str) -> None:
    """Reject stale grades, incomplete outcomes and hints that differ from the saved oracle selection."""
    root = next(path.parent for path in prepared_inputs.outputs_directory.rglob("meta.json")
                if json.loads(path.read_text())["experiment"] == "oracle-execution")
    if damage in ("grading_rule", "audit"):
        path = root / "audit.json"
        data = json.loads(path.read_text())
        if damage == "grading_rule":
            del data["grading_rule"]
        else:
            data["complete"] = False
        path.write_text(json.dumps(data))
    else:
        for path in root.glob("*.zst"):
            data = json.loads(zstandard.decompress(path.read_bytes()))
            if damage == "sketch":
                data["messages"][1]["content"] = "Wrong hint."
            elif "requests" in data:
                data["requests"][0]["injected"] = False
            path.write_bytes(zstandard.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError):
        list(compile_fixture(prepared_inputs).iter_inputs())


@pytest.mark.parametrize("experiment,injection,expected", [("unaided", None, True), ("oracle-execution", 0, True),
    ("oracle-execution", 1, False), ("continue", 1, False), ("continue-oracle", 1, False)])
def test_root_selection(experiment: str, injection: int | None, expected: bool) -> None:
    """Only roots with no hint or a hint at zero enter the 1× input experiment."""
    metadata = {"experiment": experiment, "inject_sketch_at": injection, "compute_multiplier_k": 5}
    assert is_input_trajectory(metadata, 5) == expected
    assert not is_input_trajectory(metadata, 3)


def test_input_signature_separates_reference_only_runs() -> None:
    """Old paired-view checkpoints must not resume under the reference-only objective."""
    manifest = {"tokenization": {"model": "fixture", "rows": 4}}
    paired_signature = hashlib.sha256(json.dumps(manifest["tokenization"], sort_keys=True).encode()).hexdigest()
    reference_signature = input_signature(manifest, "lora")
    assert reference_signature != paired_signature
    assert input_signature({**manifest, "features": {"rows": 4}}, "lora") == reference_signature


def test_legacy_processed_audits_cannot_supply_labels(prepared_inputs: CollectionFixture) -> None:
    """Reusing existing trajectory Parquet must not bypass the new grading-rule check."""
    processor = compile_fixture(prepared_inputs)

    def remove_rule(row: dict[str, Any]) -> dict[str, str]:
        """Return a legacy audit column for this fixture row without changing its outcomes."""
        audit = json.loads(row["audit"])
        del audit["grading_rule"]
        return {"audit": json.dumps(audit)}

    processor.dataset = processor.dataset.map(remove_rule)
    with pytest.raises(ValueError, match="regrade"):
        list(processor.iter_inputs())
