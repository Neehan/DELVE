"""Save tokenized critic inputs as Parquet chunks and read them for Qwen encoding or LoRA training."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from critic_only.constants import CRITIC_SUBDIRECTORY, INPUT_CHUNK_ROWS, TOKENIZED_SUBDIRECTORY, TOKENIZER_SUBDIRECTORY
from critic_only.data.tokenizer import token_digest
from discovery_execution.processing.constants import MANIFEST_FILENAME, SPLITS
from discovery_execution.processing.storage import file_digest, load_parquet_splits, read_manifest, write_json_atomic, write_parquet_atomic


class CriticInputStorage:
    """Own the saved token inputs and tokenizer.

    save_inputs writes chunks by split. read_chunk checks and loads one chunk for encoding; load_splits opens partitions for LoRA. load_tokenizer and load_metadata restore the saved tokenizer and settings.
    """

    def __init__(self, directory: Path) -> None:
        """Choose the processed model directory used for reading and writing.

        Args:
            directory: Model root containing manifest.json and split directories.

        Returns:
            None. Stores the root path without loading data.
        """
        self.directory = directory

    def load_metadata(self) -> dict[str, Any]:
        """Read model, token, and hidden-state settings from manifest.json.

        Returns:
            dict[str, Any]: Current model, input, and optional hidden-state metadata.
        """
        return read_manifest(self.directory)

    def save_inputs(self, encoded: Dataset, tokenizer: PreTrainedTokenizerBase, settings: dict[str, Any]) -> dict[str, str]:
        """Write token chunks by split, save the tokenizer, and record file hashes.

        Args:
            encoded: Tokenized examples with split assignments, both input versions, IDs, hashes, and labels.
            tokenizer: Tokenizer used to produce encoded, saved for padding and model loading later.
            settings: Pinned model revision, prompts, dimensions, and input limits.

        Returns:
            dict[str, str]: Model-relative chunk paths and checksums, also recorded in the completed tokenization section.
        """
        checksums = {}
        for split in SPLITS:
            partition = encoded.filter(lambda assignment: assignment == split, input_columns=["split"], load_from_cache_file=False).remove_columns("split")
            folder = self.directory / split / CRITIC_SUBDIRECTORY / TOKENIZED_SUBDIRECTORY
            folder.mkdir(parents=True, exist_ok=True)
            for chunk, start in enumerate(range(0, len(partition), INPUT_CHUNK_ROWS)):
                path = folder / f"part-{chunk:05d}.parquet"
                if path.exists():
                    raise FileExistsError(path)
                rows = partition.select(range(start, min(start + INPUT_CHUNK_ROWS, len(partition))))
                write_parquet_atomic(path, rows.with_format("arrow")[:])
                checksums[str(path.relative_to(self.directory))] = file_digest(path)
        tokenizer.save_pretrained(self.directory / CRITIC_SUBDIRECTORY / TOKENIZER_SUBDIRECTORY)
        manifest = read_manifest(self.directory)
        manifest["tokenization"] = {"settings": settings, "complete": True, "shards": checksums}
        write_json_atomic(self.directory / MANIFEST_FILENAME, manifest)
        return checksums

    def read_chunk(self, name: str, digest: str) -> Dataset:
        """Read one token chunk and reject changed file contents or token sequences.

        Args:
            name: Model-relative token chunk path.
            digest: Expected file checksum from the tokenization manifest.

        Returns:
            Dataset: Ordered examples; changed file bytes or token hashes raise ValueError.
        """
        path = self.directory / name
        if file_digest(path) != digest:
            raise ValueError(f"token chunk changed: {path}")
        dataset = Dataset.from_parquet(str(path))
        for row in cast(Iterable[dict[str, Any]], dataset):
            for column, hash_column in (("input_ids", "input_hash"), ("privileged_input_ids", "privileged_input_hash")):
                if row[column] is not None and token_digest(row[column]) != row[hash_column]:
                    raise ValueError(f"cached token IDs changed: {row['prefix_id']}")
        return dataset

    def load_splits(self, splits: tuple[str, ...], section: dict[str, Any]) -> DatasetDict:
        """Open the requested token splits using memory-mapped Hugging Face datasets.

        Args:
            splits: Requested train, validation, or test partitions.
            section: Completed tokenization section from this collection's manifest.

        Returns:
            DatasetDict: Verified token datasets for the requested partitions.
        """
        return load_parquet_splits(self.directory, splits, section, False)

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Restore the tokenizer saved beside the prepared input chunks.

        Returns:
            PreTrainedTokenizerBase: Matching tokenizer with right padding for critic batches.
        """
        return AutoTokenizer.from_pretrained(self.directory / CRITIC_SUBDIRECTORY / TOKENIZER_SUBDIRECTORY, padding_side="right")
