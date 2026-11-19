"""Verify exact branch histories, reference selection, fixed splits, and cached input identity."""

import hashlib
import json
import shutil
from typing import Any

import pytest
import zstandard
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from critic_only.constants import CRITIC_TARGET, TRAJECTORY_SEEDS
from critic_only.data.models import PreparationConfig
from critic_only.data.dataset_loader import input_signature
from critic_only.data.storage import CriticInputStorage
from critic_only.data.critic_input_processor import CriticInputProcessor, is_input_trajectory, select_training_problems
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.tests.models import CollectionFixture
from discovery_execution.processing.trajectory_processor import TrajectoryProcessor
from discovery_execution.processing.checkpoint_reconstructor import CheckpointReconstructor
from discovery_execution.processing.problem_splits import ProblemSplits
from discovery_execution.audit.answer_grader import AnswerGrader
from discovery_execution.audit.constants import CRITIQUE_PROMPT_PATH
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK


def compile_fixture(config: CollectionFixture) -> CriticInputProcessor:
    """Compile real raw artifacts, then select critic inputs from the shared Parquet collection."""
    splits = ProblemSplits(config.datasets_directory, config.destination.parent, 1)
    TrajectoryProcessor(config.outputs_directory, config.destination, config.solver_model, splits).compile_trajectories("fixture", TRAJECTORY_SEEDS)
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    return CriticInputProcessor(PreparationConfig(source_directory=config.destination, horizon=config.horizon),
        CheckpointReconstructor(tokenizer, OUTPUT_TOKENS_PER_BLOCK))


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
        assert row["branch"]["checkpoint_correctness"] == [1, row["branch"]["seed"] % 2]
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
            assert row["labels"] == row["branch"]["checkpoint_correctness"]
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
    row["branch"]["checkpoint_correctness"] = [1 - value for value in row["branch"]["checkpoint_correctness"]]
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
    reference_signature = input_signature(manifest)
    assert reference_signature != paired_signature
    assert input_signature({**manifest, "features": {"rows": 4}}) == reference_signature


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


@pytest.mark.parametrize("final_response,expected", [(r"Revision: \boxed{2}", 0),
    (r"Revision: \boxed{1}", 1), (r"Still working: \boxed{", 1)])
def test_verified_outcome_and_mid_response_boundary(prepared_inputs: CollectionFixture,
    final_response: str, expected: int) -> None:
    """The corrected grader supplies the final target, while first-block input excludes later text in the same message."""
    root = next(path.parent for path in prepared_inputs.outputs_directory.rglob("meta.json")
                if json.loads(path.read_text())["experiment"] == "unaided")
    first = [{"role": "user", "content": "Find the number."},
             {"role": "assistant", "content": r"At checkpoint: \boxed{1}"}]
    later = [first[0], {"role": "assistant", "content": first[1]["content"] + " FUTURE APPENDED TEXT"},
             {"role": "user", "content": CRITIQUE_PROMPT_PATH.read_text().strip()},
             {"role": "assistant", "content": r"Critique only: \boxed{9}"}]
    final = later + [{"role": "user", "content": "Revise your answer."},
                     {"role": "assistant", "content": final_response}]
    snapshots = [{"messages": history, "output_tokens": block * OUTPUT_TOKENS_PER_BLOCK}
                 for block, history in enumerate((first, later, final), start=1)]
    state_path = root / "conversation.json.zst"
    state = json.loads(zstandard.decompress(state_path.read_bytes()))
    state["messages"] = final
    state["checkpoints"] = [{"output_tokens": snapshot["output_tokens"], "message_count": len(snapshot["messages"]),
                             "last_message_chars": len(snapshot["messages"][-1]["content"])} for snapshot in snapshots]
    state_path.write_bytes(zstandard.compress(json.dumps(state).encode()))
    for block, snapshot in enumerate(snapshots, start=1):
        (root / f"checkpoint_{block}x.json.zst").write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
    audit = AnswerGrader("1").grade_checkpoints(snapshots, prepared_inputs.horizon)
    assert audit["checkpoints"]["2x"]["correctness"]["score"] == 1
    (root / "audit.json").write_text(json.dumps(audit))
    identity = str(root.relative_to(prepared_inputs.outputs_directory))
    row = next(row for row in compile_fixture(prepared_inputs).iter_inputs() if row["branch"]["path"] == identity)
    assert row["messages"] == first
    assert row["branch"]["checkpoint_correctness"] == [1, expected]


@pytest.mark.parametrize("damage", ["duplicate_boundary", "missing_boundary", "missing_request", "invalid_score"])
def test_processed_trajectory_boundaries_rejected(prepared_inputs: CollectionFixture, damage: str) -> None:
    """Corrupt checkpoint identities or non-binary labels cannot become a plausible training row."""
    processor = compile_fixture(prepared_inputs)

    def damage_root(row: dict[str, Any]) -> dict[str, str]:
        """Change semantic fields after packaging to exercise the critic reader's own validation."""
        if json.loads(row["metadata"])["experiment"] != "unaided":
            return {"conversation": row["conversation"], "audit": row["audit"]}
        state, audit = json.loads(row["conversation"]), json.loads(row["audit"])
        if damage == "duplicate_boundary":
            state["checkpoints"].append(state["checkpoints"][0])
        elif damage == "missing_boundary":
            state["checkpoints"] = state["checkpoints"][1:]
        elif damage == "missing_request":
            state["requests"] = state["requests"][1:]
        else:
            audit["checkpoints"]["3x"]["correctness"]["score"] = 0.5
        return {"conversation": json.dumps(state), "audit": json.dumps(audit)}

    processor.dataset = processor.dataset.map(damage_root)
    with pytest.raises(ValueError):
        list(processor.iter_inputs())
