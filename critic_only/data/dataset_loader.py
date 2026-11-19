"""Load verified token splits and identify the critic inputs used by a training run."""

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict

from critic_only.constants import CRITIC_INPUT_VIEW
from critic_only.data.storage import CriticInputStorage


def load_critic_splits(directory: Path, splits: tuple[str, ...]) -> tuple[DatasetDict, dict[str, Any]]:
    """Load the requested token splits without loading the language model.

    Args:
        directory: Processed model directory with its manifest and split directories.
        splits: Requested train, validation, or test partitions.

    Returns:
        tuple[DatasetDict, dict[str, Any]]: Verified datasets and model manifest. Token datasets use the library's memory-mapped Arrow cache.
    """
    inputs = CriticInputStorage(directory)
    manifest = inputs.load_metadata()
    datasets = inputs.load_splits(splits, manifest["tokenization"])

    return datasets, manifest


def input_signature(manifest: dict[str, Any]) -> str:
    """Hash the cache and selected input view so resume cannot mix training objectives.

    Args:
        manifest: Processed model metadata with a completed tokenization section.

    Returns:
        str: Hash of the tokenization section and selected reference view; unrelated manifest entries do not change it.
    """
    section = manifest["tokenization"]
    return hashlib.sha256(json.dumps({"cache": section, "input_view": CRITIC_INPUT_VIEW}, sort_keys=True).encode()).hexdigest()
