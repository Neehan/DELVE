"""Evaluate a saved critic and write metrics.json and predictions.parquet.

Use unaided trajectories from the requested split and report scores with and without critic-only reference solutions separately. Frozen mode reads vectors; LoRA mode reads tokens and loads Qwen.
"""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset
from datasets.utils import disable_progress_bars
from scipy.special import expit
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput

from transformers.utils.logging import disable_progress_bar

from critic_only.constants import CONFIG_FILENAME, DEFAULT_BATCH_SIZE, DEFAULT_LORA_BATCH_SIZE
from discovery_execution.processing.constants import SPLITS
from critic_only.data.hidden_state_dataset import HiddenStateDataset
from critic_only.data.critic_batcher import CriticBatcher
from critic_only.data.storage import CriticInputStorage
from critic_only.data.dataset_loader import load_critic_splits, input_signature
from critic_only.data.critic_input_processor import select_unaided_examples
from critic_only.eval.metrics import evaluation_report
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig
from critic_only.trainer.base_trainer import BaseTrainer


def save_evaluation(result: PredictionOutput, dataset: Dataset, args: argparse.Namespace) -> None:
    """Save per-trajectory probabilities and aggregate scores for both reference versions.

    Args:
        result: Trainer predictions: ordinary logits, privileged logits, availability mask, and binary labels.
        dataset: Evaluated examples in prediction order, including IDs, problem keys, and remaining budgets.
        args: CLI settings naming the checkpoint, evaluated split, and report destination.

    Returns:
        None. Writes metrics.json and per-example predictions.parquet to the report directory.
    """
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
    """Load the checkpoint, predict on unaided examples, and save evaluation results.

    Returns:
        None. Loads the trained critic, evaluates the requested prepared split, and writes prediction and metric files.
    """
    disable_progress_bars()
    disable_progress_bar()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    settings = CriticConfig.model_validate_json((args.checkpoint / CONFIG_FILENAME).read_text())
    if args.batch_size is None:
        args.batch_size = DEFAULT_LORA_BATCH_SIZE if settings.training_mode == "lora" else DEFAULT_BATCH_SIZE
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    datasets, cache = load_critic_splits(args.data, (args.split,), settings.training_mode)
    if input_signature(cache, settings.training_mode) != settings.data_signature:
        raise ValueError("evaluation inputs differ from training")
    dataset = select_unaided_examples(datasets[args.split])
    model = create_critic(settings, args.cpu)
    model.load_from_weights(args.checkpoint)
    training = TrainingArguments(output_dir=str(args.destination), use_cpu=args.cpu, bf16=settings.training_mode == "lora" and not args.cpu,
                                 per_device_eval_batch_size=args.batch_size, remove_unused_columns=False,
                                 label_names=["labels"], report_to="none", disable_tqdm=True)
    tokenizer = CriticInputStorage(args.data).load_tokenizer() if settings.training_mode == "lora" else None
    trainer = BaseTrainer(model=model, args=training, data_collator=CriticBatcher(tokenizer))
    result = trainer.predict(cast(Any, HiddenStateDataset(dataset) if settings.training_mode == "frozen" else dataset))
    if trainer.is_world_process_zero():
        save_evaluation(result, dataset, args)


if __name__ == "__main__":
    main()
