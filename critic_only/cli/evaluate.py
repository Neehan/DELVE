"""Evaluate saved critics on independent problems without invoking the frozen solver."""

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
from datasets import Dataset
from datasets.utils import disable_progress_bars
from scipy.special import expit
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput

from transformers.utils.logging import disable_progress_bar

from critic_only.constants import CONFIG_FILENAME, DEFAULT_BATCH_SIZE
from discovery_execution.processing.constants import SPLITS
from critic_only.data.features import FeatureDataset, collate_features, load_feature_splits
from critic_only.data.processor import select_unaided_examples
from critic_only.eval.metrics import evaluation_report
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig
from critic_only.trainer.base_trainer import BaseTrainer


def save_evaluation(result: PredictionOutput, dataset: Dataset, args: argparse.Namespace) -> None:
    """Write separate input-view scores; missing privileged predictions are null, never substituted."""
    logits, privileged_logits, available = cast(tuple[np.ndarray, np.ndarray, np.ndarray], result.predictions)
    labels = np.asarray(result.label_ids)
    mask = np.asarray(available, dtype=bool)
    keys = dataset["problem_key"]
    report = {"checkpoint": str(args.checkpoint), "split": args.split,
              "without_reference": evaluation_report(logits, labels, keys), "with_reference": None}
    if mask.any():
        report["with_reference"] = evaluation_report(privileged_logits[mask], labels[mask],
                                                     [key for key, present in zip(keys, mask) if present])
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    privileged = [probability.tolist() if present else None
                  for probability, present in zip(expit(privileged_logits), mask)]
    Dataset.from_dict({"prefix_id": dataset["prefix_id"], "problem_key": keys,
                       "remaining_blocks": dataset["remaining_blocks"],
                       "probability": expit(logits).tolist(), "privileged_probability": privileged,
                       "final_correctness": labels.tolist()}).to_parquet(str(args.destination / "predictions.parquet"))


def main() -> None:
    """Evaluate both input views separately; branch outcome labels never enter either input."""
    disable_progress_bars()
    disable_progress_bar()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    settings = CriticConfig.model_validate_json((args.checkpoint / CONFIG_FILENAME).read_text())
    datasets, cache = load_feature_splits(args.data, (args.split,))
    if cache["features"]["signature"] != settings.feature_signature:
        raise ValueError("evaluation features differ from training")
    dataset = select_unaided_examples(datasets[args.split])
    model = create_critic(settings)
    model.load_from_weights(args.checkpoint)
    training = TrainingArguments(output_dir=str(args.destination), use_cpu=args.cpu, bf16=False,
                                 per_device_eval_batch_size=args.batch_size, remove_unused_columns=False,
                                 label_names=["labels"], report_to="none", disable_tqdm=True)
    trainer = BaseTrainer(model=model, args=training, data_collator=collate_features)
    result = trainer.predict(FeatureDataset(dataset))
    if trainer.is_world_process_zero():
        save_evaluation(result, dataset, args)


if __name__ == "__main__":
    main()
