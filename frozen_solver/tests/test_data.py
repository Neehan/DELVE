"""Guard against prefix leakage, cumulative labels, split leakage, and silent truncation."""

import json
from pathlib import Path

import pytest
import zstandard
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from frozen_solver.constants import MANIFEST_FILENAME
from frozen_solver.data.models import PreparationConfig
from frozen_solver.data.processor import DatasetProcessor, selection_fraction
from frozen_solver.data.tokenizer import PrefixTokenizer


def test_prepare_and_tokenize(prepared_inputs: PreparationConfig, tokenizer: PreTrainedTokenizerFast, tmp_path: Path) -> None:
    """Retain regressions, mask only training oracle supervision, and round-trip cached tokens."""
    DatasetProcessor(prepared_inputs).run()
    rows = Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))
    row = dict(rows[0])
    assert row["branches"][0]["outcomes"] == [1, 0]
    assert row["messages"][-1]["content"] == "Thinking"
    assert len(row["branches"]) == 4
    settings = {"backbone": "tiny", "revision": "local", "max_length": 256, "reference_mode": "none",
                "oracle_fraction": 0, "branch_limit": None, "seed": 42}
    processor = PrefixTokenizer(tokenizer)
    encoded = processor.encode(row, settings)
    assert encoded["labels"] == [[[2, 2], [1, 2]], [[2, 2], [1, 2]]]
    row["split"] = "train"
    assert processor.encode(row, settings)["labels"][1] == [[0, 0], [0, 0]]
    destination = tmp_path / "tokenized"
    processor.run(prepared_inputs.destination, destination, settings)
    assert load_from_disk(str(destination))["test"][0]["input_ids"] == encoded["input_ids"]
    assert json.loads((destination / MANIFEST_FILENAME).read_text())["input_sha256"]
    with pytest.raises(FileExistsError):
        processor.run(prepared_inputs.destination, destination, settings)


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
    settings = {"max_length": 1, "reference_mode": "none", "oracle_fraction": 1,
                "branch_limit": None, "seed": 42}
    with pytest.raises(ValueError, match="truncation"):
        processor.encode(row, settings)


def test_training_branch_limit_and_independent_coverage(prepared_inputs: PreparationConfig,
                                                       tokenizer: PreTrainedTokenizerFast) -> None:
    """Oracle coverage must use a different hash stream from the problem split; evaluation keeps all branches."""
    DatasetProcessor(prepared_inputs).run()
    row = dict(Dataset.from_parquet(str(prepared_inputs.destination / "test.parquet"))[0])
    processor = PrefixTokenizer(tokenizer)
    settings = {"max_length": 256, "reference_mode": "none", "oracle_fraction": 0.25,
                "branch_limit": 1, "seed": 42}
    assert processor.encode(row, settings)["labels"][0][0][1] == 2
    row["split"] = "train"
    key = next(f"problem-{i}" for i in range(1000)
               if selection_fraction(f"problem-{i}", 42) > 0.5
               and selection_fraction(f"oracle:problem-{i}", 42) < 0.25)
    row["problem_key"] = key
    encoded = processor.encode(row, settings)
    assert encoded["labels"][0][0][1] == 1 and encoded["labels"][1][0][1] == 1
