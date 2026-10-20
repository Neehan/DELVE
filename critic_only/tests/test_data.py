"""Guard against prefix leakage, cumulative labels, split leakage, and silent truncation."""

import json
from pathlib import Path

import pytest
import zstandard
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from critic_only.constants import MANIFEST_FILENAME, REFERENCE_PROMPT
from critic_only.data.models import PreparationConfig
from critic_only.data.processor import DatasetProcessor, selection_fraction
from critic_only.data.tokenizer import PrefixTokenizer


def test_prepare_and_tokenize(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast, tmp_path: Path) -> None:
    """Retain regressions, mask only training oracle supervision, and round-trip cached tokens."""
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))
    row = dict(rows[0])
    assert row["branches"][0]["outcomes"] == [1, 0]
    assert row["messages"][-1]["content"] == "Thinking"
    assert len(rows) == 2 and len(row["branches"]) == 2
    assert row["remaining_blocks"] == 2 and "horizon" not in row
    settings = {"backbone": "tiny", "revision": "local", "max_length": 256,
                "oracle_supervision_fraction": 0, "branch_limit": None, "seed": 42}
    processor = PrefixTokenizer(tokenizer)
    encoded = processor.encode(row, settings)
    assert not {"remaining_blocks", "fork", "attention_mask", "horizon"} & encoded.keys()
    assert len(encoded["privileged_input_ids"]) > len(encoded["input_ids"])
    assert "reference_solution" not in encoded
    assert encoded["labels"] == [[2, 2], [1, 2]]
    row["split"] = "train"
    assert processor._retain_example(row, settings)
    oracle_row = {**dict(rows[1]), "split": "train"}
    assert not processor._retain_example(oracle_row, settings)
    destination = tmp_path / "tokenized"
    processor.run(prepared_inputs.destination, destination, settings)
    assert load_from_disk(str(destination))["test"][0]["input_ids"] == encoded["input_ids"]
    assert load_from_disk(str(destination))["test"][0]["privileged_input_ids"] == encoded["privileged_input_ids"]
    assert json.loads((destination / MANIFEST_FILENAME).read_text())["input_sha256"]
    with pytest.raises(FileExistsError):
        processor.run(prepared_inputs.destination, destination, settings)


def test_oracle_history_contains_only_saved_intervention(prepared_inputs: PreparationConfig,
                                                         tokenizer: PreTrainedTokenizerFast) -> None:
    """Preserve the exact saved intervention without leaking any generated branch continuation."""
    injection = "Verified sketch: Use symmetry. Follow this sketch."
    for path in prepared_inputs.outputs_directory.rglob("checkpoint_2x.json.zst"):
        snapshot = json.loads(zstandard.decompress(path.read_bytes()))
        if "continue-oracle-at-1x" in str(path):
            snapshot["messages"][-1]["content"] = injection
            audit_path = path.parent / "audit.json"
            audit = json.loads(audit_path.read_text())
            for checkpoint in audit["checkpoints"].values():
                checkpoint["correctness"]["score"] = 0
            audit_path.write_text(json.dumps(audit))
        snapshot["messages"].append({"role": "assistant", "content": "FUTURE ANSWER MUST NOT ENTER CRITIC"})
        path.write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))
    ordinary, oracle = dict(rows[0]), dict(rows[1])
    assert not ordinary["oracle"] and oracle["oracle"]
    assert oracle["messages"] == ordinary["messages"] + [{"role": "user", "content": injection}]
    assert oracle["phase"] == "solve" and oracle["response_finished"] is False
    assert all("FUTURE ANSWER" not in message["content"] for row in (ordinary, oracle) for message in row["messages"])
    settings = {"max_length": 256, "oracle_supervision_fraction": 1, "branch_limit": None, "seed": 1234}
    processor = PrefixTokenizer(tokenizer)
    plain, injected = processor.encode(dict(ordinary), settings), processor.encode(dict(oracle), settings)
    assert not plain["oracle"] and injected["oracle"]
    assert len(injected["input_ids"]) > len(plain["input_ids"])
    assert plain["labels"] == [[2, 2], [1, 2]]
    assert injected["labels"] == [[0, 2], [0, 2]]


def test_different_oracle_messages_cannot_share_labels(prepared_inputs: PreparationConfig) -> None:
    """Reject pooling branches with different interventions even when they contain the same sketch."""
    path = next(path for path in prepared_inputs.outputs_directory.rglob("checkpoint_2x.json.zst")
                if "continue-oracle-at-1x" in str(path))
    snapshot = json.loads(zstandard.decompress(path.read_bytes()))
    snapshot["messages"][-1]["content"] += " A different instruction."
    path.write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
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
                "branch_limit": None, "seed": 42}
    with pytest.raises(ValueError, match="truncation"):
        processor.encode(row, settings)


def test_training_branch_limit_and_independent_coverage(prepared_inputs: PreparationConfig,
                                                       tokenizer: PreTrainedTokenizerFast) -> None:
    """Oracle coverage must use a different hash stream from the problem split; evaluation keeps all branches."""
    DatasetProcessor(prepared_inputs).run()
    row = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[0])
    processor = PrefixTokenizer(tokenizer)
    settings = {"max_length": 256, "oracle_supervision_fraction": 0.25,
                "branch_limit": 1, "seed": 42}
    assert processor.encode(row, settings)["labels"][0][1] == 2
    row["split"] = "train"
    key = next(f"problem-{i}" for i in range(1000)
               if selection_fraction(f"problem-{i}", 42) > 0.5
               and selection_fraction(f"oracle:problem-{i}", 42) < 0.25)
    row["problem_key"] = key
    encoded = processor.encode(row, settings)
    assert encoded["labels"][0][1] == 1
    assert processor._retain_example({**row, "oracle": True}, settings)


@pytest.mark.parametrize("oracle_index", [None, 0, 1])
@pytest.mark.parametrize("source_seed", [0, 1, 3, 4])
def test_reference_pair_follows_collection(prepared_inputs: PreparationConfig, oracle_index: int | None,
                                          source_seed: int) -> None:
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
            snapshot_path = path.parent / "checkpoint_2x.json.zst"
            snapshot = json.loads(zstandard.decompress(snapshot_path.read_bytes()))
            snapshot["messages"][-1]["content"] = "Count the complement." if oracle_index else "Use symmetry."
            snapshot_path.write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
        path.write_text(json.dumps(metadata))
    for directory in list(prepared_inputs.outputs_directory.rglob("seed_0")):
        directory.rename(directory.with_name(f"seed_{source_seed}"))
    if oracle_index is None:
        for directory in list(prepared_inputs.outputs_directory.rglob("continue-oracle-at-1x")):
            directory.rename(prepared_inputs.destination.parent / f"withheld-oracle-{directory.parent.parent.name}")
    DatasetProcessor(prepared_inputs).run()
    row = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[0])
    selected = source_seed % 2 if oracle_index is None else oracle_index
    assert row["solution_index"] == selected
    assert row["reference_solution"] == ("Alternate proof." if selected else "The answer is one.")
    if oracle_index is not None:
        injected = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[1])
        assert injected["messages"][-1]["content"] == ("Count the complement." if selected else "Use symmetry.")


def test_optional_reference_encoding(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast,
                                     tmp_path: Path) -> None:
    """Cache null for absent references and reject truncation even when only the privileged view is too long."""
    DatasetProcessor(prepared_inputs).run()
    row = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[0])
    processor = PrefixTokenizer(tokenizer)
    settings = {"max_length": 256, "oracle_supervision_fraction": 1, "branch_limit": None, "seed": 1234}
    paired = processor.encode(row, settings)
    assert paired["privileged_input_ids"][-len(tokenizer.encode(REFERENCE_PROMPT.format(solution=row["reference_solution"]))):] == (
        tokenizer.encode(REFERENCE_PROMPT.format(solution=row["reference_solution"])))
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
