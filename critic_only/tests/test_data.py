"""Guard against prefix leakage, cumulative labels, split leakage, and silent truncation."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import zstandard
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from critic_only.constants import CRITIC_PROMPT_PATH, MANIFEST_FILENAME, REFERENCE_PROMPT_PATH
from critic_only.data.models import PreparationConfig
from critic_only.data.processor import DatasetProcessor, select_input_fork, select_training_problems, selection_fraction
from critic_only.tests.conftest import write_run
from critic_only.data.tokenizer import PrefixTokenizer


def test_training_problem_subsets_are_nested_and_keep_branches() -> None:
    """Select exact problem counts independent of row order, retaining both conditions together."""
    dataset = Dataset.from_list([
        {"problem_key": f"problem-{problem}", "oracle": oracle, "seed": seed}
        for problem in range(8) for oracle in (False, True) for seed in range(2)
    ])
    quarter = select_training_problems(dataset, 0.25)
    half = select_training_problems(dataset, 0.5)
    reordered = select_training_problems(dataset.select(list(reversed(range(len(dataset))))), 0.5)
    assert len(quarter) == 8 and len(half) == 16
    assert set(quarter["problem_key"]) < set(half["problem_key"])
    assert set(half["problem_key"]) == set(reordered["problem_key"])
    for key in set(half["problem_key"]):
        assert [row for row in half.to_list() if row["problem_key"] == key] == [
            row for row in dataset.to_list() if row["problem_key"] == key
        ]
    assert select_training_problems(dataset, 1.0).to_list() == dataset.to_list()
    with pytest.raises(ValueError, match="selects no problems"):
        select_training_problems(dataset, 0.01)


def test_prepare_and_tokenize(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast, tmp_path: Path) -> None:
    """Retain regressions, mask only training oracle supervision, and round-trip cached tokens."""
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet")).to_list()
    row = next(dict(row) for row in rows if row["fork"] == 1 and not row["oracle"] and row["branch"]["seed"] == 0)
    assert row["branch"]["final_correctness"] == 0
    assert row["messages"][-1]["content"] == "Observed continue branch 0"
    assert len(rows) == 5
    assert row["remaining_blocks"] == 1 and "horizon" not in row
    settings = {"model": "tiny", "revision": "local", "max_length": 256,
                "oracle_supervision_fraction": 0, "seed": 42}
    processor = PrefixTokenizer(tokenizer)
    encoded = processor.encode(row, settings)
    assert not {"fork", "attention_mask", "horizon"} & encoded.keys()
    assert len(encoded["privileged_input_ids"]) > len(encoded["input_ids"])
    assert "reference_solution" not in encoded
    assert encoded["labels"] == 0
    assert encoded["remaining_blocks"] == 1
    row["split"] = "train"
    assert processor._retain_example(row, settings)
    oracle_row = {**next(dict(row) for row in rows if row["oracle"]), "split": "train"}
    assert not processor._retain_example(oracle_row, settings)
    destination = tmp_path / "tokenized"
    processor.run(prepared_inputs.destination, destination, settings)
    saved = next(item for item in cast(Dataset, load_from_disk(str(destination))["test"]).to_list() if item["prefix_id"] == row["prefix_id"])
    assert saved["input_ids"] == encoded["input_ids"]
    assert saved["privileged_input_ids"] == encoded["privileged_input_ids"]
    assert json.loads((destination / MANIFEST_FILENAME).read_text())["input_sha256"]
    with pytest.raises(FileExistsError):
        processor.run(prepared_inputs.destination, destination, settings)


def test_branch_snapshots_are_read_once(prepared_inputs: PreparationConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefix validation and oracle extraction share each checkpoint read and retain matching histories."""
    processor = DatasetProcessor(prepared_inputs)
    read = Mock(wraps=processor._read_json)
    monkeypatch.setattr(processor, "_read_json", read)
    processor.run()
    paths = [call.args[0] for call in read.call_args_list]
    for snapshot in prepared_inputs.outputs_directory.rglob("checkpoint_2x.json.zst"):
        assert paths.count(snapshot) == (1 if "continue" in str(snapshot) else 0)


def test_oracle_history_contains_solver_response(prepared_inputs: PreparationConfig,
                                                  tokenizer: PreTrainedTokenizerFast) -> None:
    """Expose the sketch and branch-specific execution, but no later text, grades, or sibling outcomes."""
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet")).to_list()
    oracle = [dict(row) for row in rows if row["oracle"]]
    assert len(oracle) == 2 and oracle[0]["messages"] != oracle[1]["messages"]
    processor = PrefixTokenizer(tokenizer)
    for row in oracle:
        assert row["messages"][-2] == {"role": "user", "content": "Use symmetry."}
        assert row["messages"][-1]["content"] == f"Observed continue-oracle branch {row['branch']['seed']}"
        assert row["phase"] == "critique" and row["response_finished"] is True
        assert all("FINAL FUTURE" not in message["content"] for message in row["messages"])
        encoded = processor.encode(row, {"max_length": 256})
        assert encoded["labels"] == row["branch"]["seed"]
        assert encoded["remaining_blocks"] == 1
    assert len({row["prefix_id"] for row in rows}) == len(rows)


def test_different_oracle_messages_are_rejected(prepared_inputs: PreparationConfig,
                                               replace_saved_text: Callable[[Path, str, str], None]) -> None:
    """Require matched interventions when collecting branches from the same source."""
    path = next(path for path in prepared_inputs.outputs_directory.rglob("branch_0") if "continue-oracle-at-1x" in str(path))
    replace_saved_text(path, "Use symmetry.", "Use symmetry. A different instruction.")
    with pytest.raises(ValueError, match="different injection messages"):
        DatasetProcessor(prepared_inputs).run()


@pytest.mark.parametrize("damage", ["prefix", "incomplete", "nonbinary", "missing", "sampling"])
def test_invalid_collection_fails(prepared_inputs: PreparationConfig, damage: str) -> None:
    """Incomplete, mislabeled, or different-policy branches must not silently become training targets."""
    branch = next(prepared_inputs.outputs_directory.rglob("branch_0"))
    if damage == "prefix":
        path = branch / "checkpoint_2x.json.zst"
        snapshot = json.loads(zstandard.decompress(path.read_bytes()))
        snapshot["messages"][0]["content"] = "different problem"
        path.write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
    elif damage == "sampling":
        path = branch / "meta.json"
        metadata = json.loads(path.read_text())
        metadata["backend"]["temperature"] = 0.6
        path.write_text(json.dumps(metadata))
    else:
        path = branch / "audit.json"
        audit = json.loads(path.read_text())
        if damage == "incomplete":
            audit["complete"] = False
        elif damage == "nonbinary":
            audit["checkpoints"]["3x"]["correctness"]["score"] = 0.5
        else:
            del audit["checkpoints"]["3x"]
        path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        DatasetProcessor(prepared_inputs).run()
    assert not prepared_inputs.destination.exists()


def test_split_depends_only_on_problem(prepared_inputs: PreparationConfig) -> None:
    """Repeated calls and different prefixes cannot move a problem between partitions."""
    processor = DatasetProcessor(prepared_inputs)
    key = "aime/2025/AIME-2025-I-1"
    assert processor._split(key, 2025) == processor._split(key, 2025)
    assert processor._split(key, 2026) == "test"
    assert selection_fraction(key, 42) != selection_fraction(key, 43)


def test_no_truncation(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast) -> None:
    """A long input fails rather than training on a different state than the labels describe."""
    DatasetProcessor(prepared_inputs).run()
    row = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[0])
    processor = PrefixTokenizer(tokenizer)
    settings = {"max_length": 1, "oracle_supervision_fraction": 1,
                "seed": 42}
    with pytest.raises(ValueError, match="truncation"):
        processor.encode(row, settings)


def test_all_branches_and_independent_oracle_selection(prepared_inputs: PreparationConfig,
                                                       tokenizer: PreTrainedTokenizerFast) -> None:
    """Keep all branches in train and evaluation; select oracle problems independently of the split."""
    DatasetProcessor(prepared_inputs).run()
    row = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[0])
    processor = PrefixTokenizer(tokenizer)
    settings = {"max_length": 256, "oracle_supervision_fraction": 0.25,
                "seed": 42}
    assert processor.encode(row, settings)["labels"] == row["branch"]["final_correctness"]
    row["split"] = "train"
    key = next(f"problem-{i}" for i in range(1000)
               if selection_fraction(f"problem-{i}", 42) > 0.5
               and selection_fraction(f"oracle:problem-{i}", 42) < 0.25)
    row["problem_key"] = key
    encoded = processor.encode(row, settings)
    assert encoded["labels"] == row["branch"]["final_correctness"]
    assert processor._retain_example({**row, "oracle": True}, settings)


@pytest.mark.parametrize("oracle_index", [None, 0, 1])
@pytest.mark.parametrize("source_seed", [0, 1, 3, 4])
def test_reference_pair_follows_collection(prepared_inputs: PreparationConfig, oracle_index: int | None,
                                          source_seed: int, replace_saved_text: Callable[[Path, str, str], None]) -> None:
    """Keep a saved oracle selection, or use source-seed round robin when oracle branches are absent."""
    for path in prepared_inputs.datasets_directory.rglob("*.jsonl"):
        problem = json.loads(path.read_text())
        problem["solutions"].append({"source": "ai_generated", "solution": "Alternate proof.",
                                     "sketch": "Count the complement."})
        path.write_text(json.dumps(problem) + "\n")
    for path in prepared_inputs.outputs_directory.rglob("meta.json"):
        metadata = json.loads(path.read_text())
        if metadata["experiment"] == "unaided":
            metadata["seed"] = source_seed
        else:
            metadata["source_seed"] = source_seed
        if metadata["experiment"] == "continue-oracle" and oracle_index is not None:
            metadata["solution_index"] = oracle_index
            replace_saved_text(path.parent, "Use symmetry.", "Count the complement." if oracle_index else "Use symmetry.")
        path.write_text(json.dumps(metadata))
    for directory in list(prepared_inputs.outputs_directory.rglob("seed_0")):
        directory.rename(directory.with_name(f"seed_{source_seed}"))
    if oracle_index is None:
        for directory in list(prepared_inputs.outputs_directory.rglob("continue-oracle-at-1x")):
            directory.rename(prepared_inputs.destination.parent / f"withheld-oracle-{directory.parent.parent.name}")
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet")).to_list()
    row = next(dict(row) for row in rows if row["fork"] == 1 and not row["oracle"])
    selected = source_seed % 2 if oracle_index is None else oracle_index
    assert row["solution_index"] == selected
    assert row["reference_solution"] == ("Alternate proof." if selected else "The answer is one.")
    if oracle_index is not None:
        injected = next(dict(row) for row in rows if row["oracle"])
        assert injected["messages"][-2]["content"] == ("Count the complement." if selected else "Use symmetry.")


def test_optional_reference_encoding(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast,
                                     tmp_path: Path) -> None:
    """Cache null for absent references and reject truncation even when only the privileged view is too long."""
    DatasetProcessor(prepared_inputs).run()
    row = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[0])
    processor = PrefixTokenizer(tokenizer)
    settings = {"max_length": 256, "oracle_supervision_fraction": 1, "seed": 1234}
    paired = processor.encode(row, settings)
    assert len(paired["privileged_input_ids"]) > len(paired["input_ids"])
    with pytest.raises(ValueError, match="truncation"):
        processor.encode(row, {**settings, "max_length": len(paired["input_ids"])})
    row["reference_solution"] = None
    plain = processor.encode(row, settings)
    assert plain["privileged_input_ids"] is None
    assert plain["input_ids"] == paired["input_ids"] and plain["labels"] == paired["labels"]
    source = tmp_path / "missing-references"
    source.mkdir()
    Dataset.from_list([row]).to_parquet(str(source / "test.parquet"))
    (source / MANIFEST_FILENAME).write_text((prepared_inputs.destination / MANIFEST_FILENAME).read_text())
    processor.run(source, tmp_path / "cache", settings)
    assert load_from_disk(str(tmp_path / "cache"))["test"][0]["privileged_input_ids"] is None


def test_chat_prompt_preserves_history_and_separates_reference(prepared_inputs: PreparationConfig,
                                                               tokenizer: PreTrainedTokenizerFast,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unfinished reasoning and sketches verbatim, with references restricted to the privileged view."""
    DatasetProcessor(prepared_inputs).run()
    row = next(dict(row) for row in Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet")).to_list() if row["oracle"])
    row["messages"][1]["content"] = "<think>unfinished reasoning with {symbols}"
    template = Mock(wraps=tokenizer.apply_chat_template)
    monkeypatch.setattr(tokenizer, "apply_chat_template", template)
    processor = PrefixTokenizer(tokenizer)
    assert processor.system_prompt == CRITIC_PROMPT_PATH.read_text().strip()
    processor.encode(row, {"max_length": 512})
    plain, privileged = [call.args[0] for call in template.call_args_list]
    assert [message["role"] for message in plain] == ["system", "user"]
    assert plain[1] == privileged[1]
    assert "Remaining budget: 1 blocks" in plain[0]["content"]
    for message in row["messages"]:
        assert message["content"] in plain[1]["content"]
    assert row["reference_solution"] not in plain[1]["content"]
    assert row["reference_solution"] not in plain[0]["content"]
    assert "calibrate" not in plain[0]["content"]
    reference = REFERENCE_PROMPT_PATH.read_text().strip().format(solution=row["reference_solution"])
    assert privileged[0]["content"] == plain[0]["content"] + "\n\n" + reference
    assert "not available to the solver" in privileged[0]["content"]
    for call in template.call_args_list:
        assert call.kwargs == {"tokenize": True, "add_generation_prompt": False,
                               "truncation": False, "return_dict": False}


def test_root_oracle_runs_and_round_robin(prepared_inputs: PreparationConfig) -> None:
    """Treat full-budget oracle-at-zero seeds as independent 1x inputs with their saved reference selection."""
    for path in prepared_inputs.datasets_directory.rglob("*.jsonl"):
        problem = json.loads(path.read_text())
        problem["solutions"].append({"source": "ai_generated", "solution": "Alternate proof.", "sketch": "Count the complement."})
        path.write_text(json.dumps(problem) + "\n")
    for path in list(prepared_inputs.outputs_directory.rglob("meta.json")):
        metadata = json.loads(path.read_text())
        if metadata["experiment"] != "unaided":
            continue
        for seed in range(2):
            sketch = "Count the complement." if seed else "Use symmetry."
            initial = [{"role": "user", "content": f"Find the number. Verified sketch: {sketch}"}]
            histories = [initial + [{"role": "assistant", "content": f"Root {seed} block {block}"}]
                         for block in range(1, 4)]
            # Saved conversations append to this response; preserve earlier snapshot character boundaries.
            for block in range(1, 3):
                histories[block][-1]["content"] = histories[block - 1][-1]["content"] + f" block {block + 1}"
            directory = path.parent.parent.parent.parent / "oracle-at-0x-execution" / metadata["problem_id"] / f"seed_{seed}"
            write_run(directory, {**metadata, "experiment": "oracle-execution", "inject_sketch_at": 0,
                                   "solution_index": seed, "seed": seed}, histories, [1, 1, seed])
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet")).to_list()
    roots = [row for row in rows if row["fork"] == 0 and row["oracle"]]
    assert len(roots) == 2
    for row in roots:
        assert row["remaining_blocks"] == 2
        assert row["solution_index"] == row["branch"]["seed"]
        assert row["branch"]["final_correctness"] == row["branch"]["seed"]
        assert row["reference_solution"] == ("Alternate proof." if row["solution_index"] else "The answer is one.")
        assert row["messages"][-1]["content"] == f"Root {row['branch']['seed']} block 1"


@pytest.mark.parametrize("experiment,fork,expected", [("continue", 1, 1), ("continue-oracle", 3, 3),
                                                     ("continue", 4, None), ("unaided", None, 0),
                                                     ("oracle-execution", 0, 0), ("oracle-execution", 1, None)])
def test_nonterminal_input_selection(experiment: str, fork: int | None, expected: int | None) -> None:
    """A 5x budget accepts observed inputs at 1x through 4x and excludes terminal or delayed-root-oracle inputs."""
    metadata = {"experiment": experiment, "continue_at": fork, "inject_sketch_at": fork, "compute_multiplier_k": 5}
    assert select_input_fork(metadata, 5) == expected
    assert select_input_fork(metadata, 3) is None


def test_terminal_forks_are_not_read(prepared_inputs: PreparationConfig) -> None:
    """A fork whose next checkpoint is terminal must not create a trivial final-answer example."""
    source = next(prepared_inputs.outputs_directory.rglob("branch_0"))
    metadata = json.loads((source / "meta.json").read_text())
    metadata["continue_at"] = 2
    terminal = source.parent / "terminal"
    terminal.mkdir()
    (terminal / "meta.json").write_text(json.dumps(metadata))
    DatasetProcessor(prepared_inputs).run()
    assert len(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))) == 5


def test_duplicate_branch_identity_rejected(prepared_inputs: PreparationConfig) -> None:
    """Do not count copied trajectories as additional independent training examples."""
    source = next(prepared_inputs.outputs_directory.rglob("branch_0"))
    copied = source.parent / "copy"
    copied.mkdir()
    (copied / "meta.json").write_bytes((source / "meta.json").read_bytes())
    with pytest.raises(ValueError, match="duplicate.*trajectory"):
        DatasetProcessor(prepared_inputs).run()


def test_input_checkpoint_excludes_later_phase_and_text(prepared_inputs: PreparationConfig) -> None:
    """Recover stop status and phase at 2x even though the completed conversation ends in a different phase."""
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet")).to_list()
    for row in rows:
        assert all("FINAL FUTURE" not in message["content"] for message in row["messages"])
        if row["fork"] == 1:
            assert row["phase"] == "critique" and row["response_finished"] is True
            assert row["branch"]["final_correctness"] == row["branch"]["seed"]
        assert row["split"] == "test"


def test_eight_branches_produce_eight_inputs_per_condition(prepared_inputs: PreparationConfig,
                                                          tokenizer: PreTrainedTokenizerFast) -> None:
    """Keep eight distinct post-1x histories and individual labels for each intervention condition."""
    for path in list(prepared_inputs.outputs_directory.rglob("branch_0/meta.json")):
        metadata = json.loads(path.read_text())
        first = json.loads(zstandard.decompress((path.parent / "checkpoint_2x.json.zst").read_bytes()))
        prefix = first["messages"][:-1]
        for seed in range(2, 8):
            text = f"Observed branch {seed}"
            histories = [prefix + [{"role": "assistant", "content": text}],
                         prefix + [{"role": "assistant", "content": text + " FINAL FUTURE"}]]
            write_run(path.parent.with_name(f"branch_{seed}"), {**metadata, "seed": seed}, histories, [1, seed % 2])
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet")).to_list()
    encoder = PrefixTokenizer(tokenizer)
    for oracle in (False, True):
        examples = [row for row in rows if row["fork"] == 1 and row["oracle"] == oracle]
        assert len(examples) == 8
        assert len({row["messages"][-1]["content"] for row in examples}) == 8
        for row in examples:
            assert encoder.encode(row, {"max_length": 256})["labels"] == row["branch"]["seed"] % 2


def test_duplicate_root_run_rejected(prepared_inputs: PreparationConfig) -> None:
    """Copied root runs must not silently double one seed's training weight."""
    original = next(path for path in prepared_inputs.outputs_directory.rglob("meta.json")
                    if json.loads(path.read_text())["experiment"] == "unaided")
    copied = original.parent.with_name("copied_seed")
    copied.mkdir()
    for path in original.parent.iterdir():
        (copied / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="duplicate.*trajectory"):
        DatasetProcessor(prepared_inputs).run()


def test_checkpoint_ahead_of_saved_progress_rejected(prepared_inputs: PreparationConfig) -> None:
    """Do not accept a checkpoint beyond the conversation's recorded token progress."""
    path = next(prepared_inputs.outputs_directory.rglob("branch_0/conversation.json.zst"))
    state = json.loads(zstandard.decompress(path.read_bytes()))
    state["output_tokens"] = 0
    path.write_bytes(zstandard.compress(json.dumps(state).encode()))
    with pytest.raises(ValueError, match="checkpoint.*progress"):
        DatasetProcessor(prepared_inputs).run()
