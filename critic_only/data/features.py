"""Load verified cached vectors into memory and batch paired critic views."""

import json
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, DatasetDict
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from critic_only.constants import FEATURE_SUBDIRECTORY, CRITIC_SUBDIRECTORY
from discovery_execution.processing.constants import MANIFEST_FILENAME
from discovery_execution.processing.storage import file_digest


def load_feature_splits(directory: Path, splits: tuple[str, ...]) -> tuple[DatasetDict, dict[str, Any]]:
    """Require a completed extraction and verify its shards before loading them into RAM."""
    manifest = json.loads((directory / MANIFEST_FILENAME).read_text())
    extraction = manifest["features"]
    if not extraction["complete"]:
        raise ValueError("feature extraction is incomplete")
    partitions: dict[str, list[str]] = {}
    root = directory / CRITIC_SUBDIRECTORY / FEATURE_SUBDIRECTORY
    for name, digest in extraction["shards"].items():
        split = Path(name).parts[0]
        if split not in splits:
            continue
        path = root / name
        if file_digest(path) != digest:
            raise ValueError(f"feature shard changed: {path}")
        partitions.setdefault(split, []).append(str(path))
    datasets = DatasetDict({split: Dataset.from_parquet(cast(Any, paths), keep_in_memory=True) for split, paths in partitions.items()})
    for dataset in datasets.values():
        identifiers = list(dataset["prefix_id"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate feature example IDs")
    return datasets, manifest


def collate_features(rows: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Stack cached vectors, retaining one label per example and its optional reference view."""
    indices = [index for index, row in enumerate(rows) if row["privileged_hidden_state"] is not None]
    views = [row["hidden_state"] for row in rows] + [rows[index]["privileged_hidden_state"] for index in indices]
    metadata = rows + [rows[index] for index in indices]
    return {"labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32),
            "hidden_state": torch.stack([torch.as_tensor(view, dtype=torch.float32) for view in views]),
            "privileged_indices": torch.tensor(indices, dtype=torch.long),
            "oracle": torch.tensor([row["oracle"] for row in metadata], dtype=torch.bool),
            "remaining_blocks": torch.tensor([row["remaining_blocks"] for row in metadata], dtype=torch.long)}


class FeatureDataset(TorchDataset):
    """Materialize selected feature columns once as CPU tensors for inexpensive MLP minibatches."""

    def __init__(self, dataset: Dataset) -> None:
        """Retain optional reference availability without decoding Arrow vectors on every epoch."""
        self.hidden_states = torch.from_numpy(dataset.with_format("numpy", columns=["hidden_state"])[:]["hidden_state"])
        self.reference_available = dataset.select_columns(["privileged_hidden_state"]).with_format("arrow")[:]["privileged_hidden_state"].is_valid().to_pylist()
        self.privileged_hidden_states = torch.zeros_like(self.hidden_states)
        indices = [index for index, present in enumerate(self.reference_available) if present]
        if indices:
            selected = dataset.select(indices).with_format("numpy", columns=["privileged_hidden_state"])
            self.privileged_hidden_states[indices] = torch.from_numpy(selected[:]["privileged_hidden_state"])
        self.labels = list(dataset["labels"])
        self.oracle = list(dataset["oracle"])
        self.remaining_blocks = list(dataset["remaining_blocks"])

    def __len__(self) -> int:
        """Return the number of branch examples, before adding optional reference views."""
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return tensor views and the branch metadata needed by the paired loss."""
        return {"hidden_state": self.hidden_states[index], "privileged_hidden_state": self.privileged_hidden_states[index] if self.reference_available[index] else None,
                "labels": self.labels[index], "oracle": self.oracle[index], "remaining_blocks": self.remaining_blocks[index]}
