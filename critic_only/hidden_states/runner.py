"""Generate resumable hidden-state chunks using Hugging Face prediction and existing storage."""

from typing import Any, cast

import numpy as np
from datasets import Dataset
from transformers import Trainer

from critic_only.data.storage import CriticInputStorage
from critic_only.hidden_states.storage import HiddenStateStorage


class HiddenStateRunner:
    """Ask Trainer to encode each unfinished chunk, then save vectors in original row order.

    run verifies completed chunks before reusing them. _encode_chunk combines both reference versions and sorts by length to reduce padding. Trainer distributes each chunk's batches across devices; only the primary process writes files.
    """

    def __init__(self, inputs: CriticInputStorage, outputs: HiddenStateStorage, trainer: Trainer) -> None:
        """Connect prepared inputs, hidden-state storage, and HF prediction.

        Args:
            inputs: Token storage with exact sequences and file hashes.
            outputs: Hidden-state storage in the same processed model directory.
            trainer: HF Trainer wrapping HiddenStateEncoder and a right-padding collator.

        Returns:
            None.
        """
        if inputs.directory != outputs.directory:
            raise ValueError("token and hidden-state storage must share the same collection")
        self.inputs, self.outputs, self.trainer = inputs, outputs, trainer

    def run(self) -> None:
        """Encode incomplete chunks and validate completed ones without recomputing them.

        Returns:
            None. Saves aligned vectors and labels. Every distributed rank visits the same chunks so Trainer can gather predictions safely.
        """
        manifest = self.inputs.load_metadata()
        signature = manifest["features"]["signature"]
        for name, digest in sorted(manifest["tokenization"]["shards"].items()):
            rows = self.inputs.read_chunk(name, digest)
            if self.outputs.has_chunk(name):
                self.outputs.validate_chunk(name, rows, signature)
            else:
                records = self._encode_chunk(rows)
                if self.trainer.is_world_process_zero():
                    self.outputs.save_chunk(name, records, signature)
                    print(f"Saved {self.outputs.chunk_path(name)} examples={len(rows)}", flush=True)
            self.trainer.accelerator.wait_for_everyone()

    def _encode_chunk(self, dataset: Dataset) -> list[dict[str, Any]]:
        """Predict both reference versions together and map vectors back to their original rows.

        Args:
            dataset: One verified token chunk in saved row order.

        Returns:
            list[dict[str, Any]]: Original metadata and labels with ordinary/reference vectors; missing references remain None.
        """
        rows = dataset.to_list()
        reference_indices = [index for index, row in enumerate(rows) if row["privileged_input_ids"] is not None]
        sequences = [row["input_ids"] for row in rows] + [rows[index]["privileged_input_ids"] for index in reference_indices]
        order = np.argsort([len(tokens) for tokens in sequences], kind="stable")
        inputs = Dataset.from_dict({"input_ids": [sequences[index] for index in order]})
        dataloader_count = len(self.trainer.accelerator._dataloaders)
        try:
            predictions = np.asarray(self.trainer.predict(cast(Any, inputs)).predictions)
        finally:
            # Accelerate retains prediction loaders; release this chunk's loaders while keeping Qwen prepared.
            del self.trainer.accelerator._dataloaders[dataloader_count:]
            self.trainer.callback_handler.eval_dataloader = None
        if predictions.ndim != 2 or len(predictions) != len(sequences) or not np.isfinite(predictions).all():
            raise ValueError("encoding must return one finite hidden vector per input")
        vectors = predictions[np.argsort(order)].tolist()
        references = dict(zip(reference_indices, vectors[len(rows):]))
        records = []
        for index, row in enumerate(rows):
            metadata = {key: value for key, value in row.items() if key not in {"input_ids", "privileged_input_ids", "length"}}
            records.append({**metadata, "hidden_state": vectors[index], "privileged_hidden_state": references.get(index)})
        return records
