"""Evaluate saved critics on independent problems without invoking the frozen solver."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from datasets.utils import disable_progress_bars
from scipy.special import expit
from transformers import AutoTokenizer, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from transformers.utils.logging import disable_progress_bar

from critic_only.constants import CONFIG_FILENAME, DEFAULT_BATCH_SIZE, MANIFEST_FILENAME, SPLITS, TOKENIZED_DIRECTORY
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.data.processor import select_ordinary_examples
from critic_only.eval.metrics import evaluation_report
from critic_only.models.loader import create_critic_from_pretrained
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
                       "probabilities": expit(logits).tolist(), "privileged_probabilities": privileged,
                       "counts": labels.tolist()}).to_parquet(str(args.destination / "predictions.parquet"))


def main() -> None:
    """Evaluate both input views separately; branch outcome labels never enter either input."""
    disable_progress_bars()
    disable_progress_bar()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TOKENIZED_DIRECTORY)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    settings = CriticConfig.model_validate_json((args.checkpoint / CONFIG_FILENAME).read_text())
    cache = json.loads((args.data / MANIFEST_FILENAME).read_text())
    if (cache["settings"]["backbone"], cache["settings"]["revision"]) != (settings.backbone, settings.revision):
        raise ValueError("evaluation cache uses a different backbone/tokenizer revision")
    if cache["preparation"]["settings"]["horizon"] != settings.horizon:
        raise ValueError("evaluation horizon differs from training")
    dataset = select_ordinary_examples(cast(DatasetDict, load_from_disk(str(args.data)))[args.split])
    model = create_critic_from_pretrained(settings, torch.float32 if args.cpu else torch.bfloat16)
    model.load_from_weights(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.data / "tokenizer", padding_side="right")
    training = TrainingArguments(output_dir=str(args.destination), use_cpu=args.cpu, bf16=not args.cpu,
                                 per_device_eval_batch_size=args.batch_size, remove_unused_columns=False,
                                 label_names=["labels"], report_to="none", disable_tqdm=True)
    trainer = BaseTrainer(model=model, args=training, data_collator=PrefixTokenizer(tokenizer).pad_batch)
    result = trainer.predict(cast(Any, dataset))
    if trainer.is_world_process_zero():
        save_evaluation(result, dataset, args)


if __name__ == "__main__":
    main()
