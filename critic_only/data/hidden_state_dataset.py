"""Load cached Qwen vectors into CPU tensors once and serve them to the training DataLoader."""

from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset


class HiddenStateDataset(TorchDataset):
    """Keep the selected hidden vectors in CPU RAM throughout training.

    Indexing returns one trajectory's vectors, label, sketch flag, and remaining budget. CriticBatcher combines these records into minibatches, so training does not reread Parquet files each step.
    """

    def __init__(self, dataset: Dataset) -> None:
        """Convert the selected vectors to CPU tensors once, before training starts.

        Args:
            dataset: Selected feature rows; each has ordinary and optional privileged vectors plus label and budget metadata.

        Returns:
            None. Stores vectors as CPU tensors and preserves the availability of each reference-conditioned vector.
        """
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
        """Count trajectories, counting a reference pair as one example.

        Returns:
            int: Number of branch examples; a reference-conditioned view does not count as another example.
        """
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one trajectory's cached vectors and training label.

        Args:
            index: Zero-based position of a branch example in the selected dataset.

        Returns:
            dict[str, Any]: Ordinary vector, optional reference vector, binary label, oracle flag, and remaining blocks for CriticBatcher.
        """
        return {"hidden_state": self.hidden_states[index], "privileged_hidden_state": self.privileged_hidden_states[index] if self.reference_available[index] else None,
                "labels": self.labels[index], "oracle": self.oracle[index], "remaining_blocks": self.remaining_blocks[index]}
