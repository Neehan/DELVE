"""Choose cached vectors for frozen training or cached tokens for LoRA training."""

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict

from critic_only.data.storage import CriticInputStorage
from critic_only.hidden_states.storage import HiddenStateStorage


def load_critic_splits(directory: Path, splits: tuple[str, ...], training_mode: str) -> tuple[DatasetDict, dict[str, Any]]:
    """Load the requested splits as vectors or tokens, depending on training mode.

    Args:
        directory: Processed model directory with its manifest and split directories.
        splits: Requested train, validation, or test partitions.
        training_mode: frozen reads cached vectors; lora reads exact token sequences.

    Returns:
        tuple[DatasetDict, dict[str, Any]]: Verified datasets and model manifest. Token datasets use the library's memory-mapped Arrow cache; vectors are loaded into RAM.
    """
    inputs = CriticInputStorage(directory)
    manifest = inputs.load_metadata()
    if training_mode == "frozen":
        datasets = HiddenStateStorage(directory).load_splits(splits, manifest["features"])
    else:
        datasets = inputs.load_splits(splits, manifest["tokenization"])

    return datasets, manifest


def input_signature(manifest: dict[str, Any], training_mode: str) -> str:
    """Hash the selected cache metadata so a resumed run can detect changed inputs.

    Args:
        manifest: Processed model metadata with a completed token or hidden-state section.
        training_mode: frozen or lora.

    Returns:
        str: Hash of the selected input section. Adding an optional hidden-state cache does not change a LoRA run's identity.
    """
    section = manifest["features" if training_mode == "frozen" else "tokenization"]
    return hashlib.sha256(json.dumps(section, sort_keys=True).encode()).hexdigest()
