"""Verify exact branch histories, reference selection, fixed splits, and cached input identity."""

import json
from typing import Any, cast

import pytest
import zstandard
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from critic_only.data.models import PreparationConfig
from critic_only.data.storage import CriticInputStorage
from critic_only.data.critic_input_processor import CriticInputProcessor, select_input_fork, select_training_problems
from critic_only.data.tokenizer import PrefixTokenizer, token_digest
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.trajectory_processor import TrajectoryProcessor
from discovery_execution.processing.problem_splits import ProblemSplits


def compile_fixture(config: CollectionFixture) -> CriticInputProcessor:
    """Compile real raw artifacts, then select critic inputs from the shared Parquet collection."""
    splits = ProblemSplits(config.datasets_directory, config.destination.parent, 1)
    TrajectoryProcessor(config.outputs_directory, config.destination, config.solver_model, splits).compile_trajectories("fixture")
    return CriticInputProcessor(PreparationConfig(source_directory=config.destination, horizon=config.horizon))


def test_shared_collection_and_tokenization(prepared_inputs: CollectionFixture, tokenizer: PreTrainedTokenizerFast,
                                            encoding_settings: dict[str, Any]) -> None:
    """Keep shared full histories while tokenized critic inputs exclude all future reasoning and grades."""
    processor = compile_fixture(prepared_inputs)
    rows = list(processor.iter_inputs())
    assert len(rows) == 10
    assert all("FINAL FUTURE" not in json.dumps(row["messages"]) for row in rows)
    branches = [row for row in rows if row["fork"] == 1]
    assert all(row["branch"]["final_correctness"] == row["branch"]["seed"] for row in branches)
    assert all(row["phase"] == "critique" and row["response_finished"] for row in branches)
    assert all(row["split"] == "test" for row in rows if "/2026/" in row["problem_key"])
    assert any("FINAL FUTURE" in text for text in processor.dataset["conversation"])
    destination = prepared_inputs.destination
    examples = Dataset.from_list(rows)
    encoded = examples.map(PrefixTokenizer(tokenizer).encode, fn_kwargs={"settings": encoding_settings},
                           remove_columns=examples.column_names)
    storage = CriticInputStorage(destination)
    shards = storage.save_inputs(encoded, tokenizer, encoding_settings)
    test_paths = [str(destination / name) for name in shards if name.startswith("test/")]
    cached = Dataset.from_parquet(cast(Any, test_paths))
    assert len(cached) == 5
    for row in cached.to_list():
        assert row["input_hash"] == token_digest(row["input_ids"])
        assert row["privileged_input_hash"] == token_digest(row["privileged_input_ids"])
        assert row["labels"] in (0, 1)
    with pytest.raises(FileExistsError):
        storage.save_inputs(encoded, tokenizer, encoding_settings)



@pytest.mark.parametrize("damage", ["snapshot", "audit", "source", "sketch", "sampling", "phase"])
def test_invalid_collection_rejected(prepared_inputs: CollectionFixture, damage: str) -> None:
    """Reject mismatched snapshots, outcomes, source histories, sketches, and branch settings."""
    branch = next(path for path in prepared_inputs.outputs_directory.rglob("branch_0") if "continue-oracle" in str(path))
    if damage == "audit":
        path = branch / "audit.json"
        data = json.loads(path.read_text())
        data["complete"] = False
        path.write_text(json.dumps(data))
    elif damage == "sampling":
        path = branch / "meta.json"
        data = json.loads(path.read_text())
        data["backend"]["top_k"] += 1
        path.write_text(json.dumps(data))
    else:
        paths = [branch / "checkpoint_2x.json.zst"] if damage == "snapshot" else list(branch.glob("*.zst"))
        for path in paths:
            data = json.loads(zstandard.decompress(path.read_bytes()))
            if damage == "phase":
                if "requests" in data:
                    data["requests"][0]["injected"] = False
            else:
                index = 0 if damage == "source" else 2 if damage == "sketch" else -1
                data["messages"][index]["content"] = "Wrong text."
            path.write_bytes(zstandard.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError):
        list(compile_fixture(prepared_inputs).iter_inputs())


@pytest.mark.parametrize("saved_index", [0, 1])
def test_saved_solution_overrides_round_robin(prepared_inputs: CollectionFixture, saved_index: int) -> None:
    """Oracle branches and paired unaided references follow the saved selection, not branch seed."""
    for path in prepared_inputs.datasets_directory.rglob("*.jsonl"):
        problems = [json.loads(line) for line in path.read_text().splitlines()]
        for problem in problems:
            problem["solutions"].append({"source": "ai_generated", "solution": "Alternate proof.", "sketch": "Count the complement."})
        path.write_text("".join(json.dumps(problem) + "\n" for problem in problems))
    for path in prepared_inputs.outputs_directory.rglob("meta.json"):
        metadata = json.loads(path.read_text())
        if metadata["experiment"] == "continue-oracle":
            metadata["solution_index"] = saved_index
            path.write_text(json.dumps(metadata))
            if saved_index:
                for artifact in path.parent.glob("*.zst"):
                    data = json.loads(zstandard.decompress(artifact.read_bytes()))
                    for message in data["messages"]:
                        message["content"] = message["content"].replace("Use symmetry.", "Count the complement.")
                    artifact.write_bytes(zstandard.compress(json.dumps(data).encode()))
    rows = list(compile_fixture(prepared_inputs).iter_inputs())
    for row in rows:
        if row["fork"]:
            assert row["solution_index"] == saved_index
            assert row["reference_solution"] == ("Alternate proof." if saved_index else "The answer is one.")


def test_reference_and_history_are_frozen(prepared_inputs: CollectionFixture, tokenizer: PreTrainedTokenizerFast,
                                         encoding_settings: dict[str, Any]) -> None:
    """Later dataset edits cannot silently replace the solution associated with a processed trajectory."""
    processor = compile_fixture(prepared_inputs)
    for path in prepared_inputs.datasets_directory.rglob("*.jsonl"):
        path.write_text("changed after setup")
    row = next(row for row in processor.iter_inputs() if row["oracle"])
    encoder = PrefixTokenizer(tokenizer)
    captured: list[list[dict[str, str]]] = []
    original = encoder._encode_messages
    def capture(messages: list[dict[str, str]], max_length: int, prefix_id: str) -> list[int]:
        """Capture the exact rendered ordinary and privileged views."""
        captured.append(messages)
        return original(messages, max_length, prefix_id)
    encoder._encode_messages = capture
    encoding_settings["critic_prompt"] = "Saved critic instructions. " + encoding_settings["critic_prompt"]
    encoding_settings["reference_prompt"] = "Saved reference instructions. " + encoding_settings["reference_prompt"]
    encoded = encoder.encode(row, encoding_settings)
    assert encoded["labels"] == row["branch"]["final_correctness"]
    assert all(messages[0]["role"] == "system" for messages in captured)
    assert all(messages[0]["content"].startswith("Saved critic instructions.") for messages in captured)
    assert "Saved reference instructions." in captured[1][0]["content"]
    assert "Saved reference instructions." not in captured[0][0]["content"]
    assert captured[0][1] == captured[1][1]
    assert "Use symmetry." in captured[0][1]["content"]
    assert "Observed continue-oracle" in captured[0][1]["content"]
    assert "The answer is one." not in captured[0][0]["content"]
    assert "The answer is one." in captured[1][0]["content"]
    assert "not available to the solver" in captured[1][0]["content"]
    assert "FINAL FUTURE" not in json.dumps(captured)
    with pytest.raises(ValueError, match="truncation"):
        encoder.encode(row, {**encoding_settings, "max_length": 1})


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


@pytest.mark.parametrize("experiment,fork,expected", [("continue", 1, 1), ("continue-oracle", 3, 3),
    ("continue", 4, None), ("unaided", None, 0), ("oracle-execution", 0, 0), ("oracle-execution", 1, None)])
def test_nonterminal_input_selection(experiment: str, fork: int | None, expected: int | None) -> None:
    """Keep one observed block after each fork, excluding terminal and delayed root-oracle examples."""
    metadata = {"experiment": experiment, "continue_at": fork, "inject_sketch_at": fork, "compute_multiplier_k": 5}
    assert select_input_fork(metadata, 5) == expected
    assert select_input_fork(metadata, 3) is None
