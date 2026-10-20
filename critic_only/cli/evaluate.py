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

from transformers.utils.logging import disable_progress_bar

from critic_only.constants import CONFIG_FILENAME, DEFAULT_BATCH_SIZE, MANIFEST_FILENAME, SPLITS, TOKENIZED_DIRECTORY
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.eval.metrics import evaluation_report
from critic_only.models.de_critic import DECritic
from critic_only.models.ordinary import Ordinary
from critic_only.models.oracle_supervised import OracleSupervised
from critic_only.models.models import CriticConfig
from critic_only.trainer.base_trainer import BaseTrainer


def main() -> None:
    """Write predictions and proper scoring rules; evaluation never uses reference labels as input."""
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
    if (cache["settings"]["reference_mode"] != settings.reference_mode
            or cache["preparation"]["settings"]["horizon"] != settings.horizon):
        raise ValueError("evaluation prompt or horizon differs from training")
    dataset = cast(DatasetDict, load_from_disk(str(args.data)))[args.split]
    critic_type = (DECritic if settings.variant == "de" else
                   OracleSupervised if settings.variant == "oracle-supervised" else Ordinary)
    model = critic_type.create_from_pretrained(settings, torch.float32 if args.cpu else torch.bfloat16)
    model.load_from_weights(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.data / "tokenizer", padding_side="right")
    training = TrainingArguments(output_dir=str(args.destination), use_cpu=args.cpu, bf16=not args.cpu,
                                 per_device_eval_batch_size=args.batch_size, remove_unused_columns=False,
                                 label_names=["labels"], report_to="none", disable_tqdm=True)
    trainer = BaseTrainer(model=model, args=training, data_collator=PrefixTokenizer(tokenizer).pad_batch)
    result = trainer.predict(cast(Any, dataset))
    logits, labels = np.asarray(result.predictions), np.asarray(result.label_ids)
    if trainer.is_world_process_zero():
        args.destination.mkdir(parents=True, exist_ok=True)
        report = evaluation_report(logits, labels, dataset["problem_key"])
        report.update(checkpoint=str(args.checkpoint), split=args.split, reference_mode=cache["settings"]["reference_mode"])
        (args.destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        Dataset.from_dict({"prefix_id": dataset["prefix_id"], "problem_key": dataset["problem_key"],
                           "probabilities": expit(logits).tolist(), "counts": labels.tolist()}).to_parquet(
                               str(args.destination / "predictions.parquet"))


if __name__ == "__main__":
    main()
