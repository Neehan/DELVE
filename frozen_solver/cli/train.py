"""Train matched LoRA critics with Hugging Face Trainer on cached prefixes."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from datasets import DatasetDict, load_from_disk
from dotenv import load_dotenv
from transformers import AutoTokenizer, TrainingArguments, set_seed

from frozen_solver.constants import (
    DEFAULT_ACCUMULATION_STEPS, DEFAULT_ATTENTION, DEFAULT_BATCH_SIZE, DEFAULT_CORRECTION_WEIGHT,
    DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_LOGGING_STEPS, DEFAULT_LORA_ALPHA, DEFAULT_LORA_RANK,
    DEFAULT_ORACLE_WEIGHT, DEFAULT_SEED, DEFAULT_VARIANT, DEFAULT_WARMUP_RATIO, DEFAULT_WEIGHT_DECAY,
    MANIFEST_FILENAME, TOKENIZED_DIRECTORY, TRAINING_DIRECTORY, VARIANTS,
)
from frozen_solver.data.collator import PrefixCollator
from frozen_solver.eval.metrics import compute_metrics
from frozen_solver.models.checkpoint import build_critic
from frozen_solver.models.models import CriticConfig
from frozen_solver.trainer.critic_trainer import CriticTrainer


def parse_arguments() -> argparse.Namespace:
    """Expose experiment choices without reimplementing distributed launch configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TOKENIZED_DIRECTORY)
    parser.add_argument("--output", type=Path, default=TRAINING_DIRECTORY)
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--oracle-weight", type=float, default=DEFAULT_ORACLE_WEIGHT)
    parser.add_argument("--correction-weight", type=float, default=DEFAULT_CORRECTION_WEIGHT)
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default=DEFAULT_ATTENTION)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--accumulation-steps", type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true", help="float32 CPU smoke tests only")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.learning_rate) <= 0:
        parser.error("training sizes and learning rate must be positive")
    return args


def training_arguments(args: argparse.Namespace) -> TrainingArguments:
    """Delegate batching, distributed reduction, scheduling, evaluation, and recovery to HF."""
    return TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs,
                             learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
                             per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation_steps,
                             bf16=not args.cpu, use_cpu=args.cpu, optim="adamw_torch" if args.cpu else "adamw_torch_fused",
                             warmup_steps=DEFAULT_WARMUP_RATIO, weight_decay=DEFAULT_WEIGHT_DECAY,
                             eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="terminal_nll", greater_is_better=False,
                             logging_steps=DEFAULT_LOGGING_STEPS, report_to="none", seed=args.seed,
                             remove_unused_columns=False, label_names=["labels"],
                             train_sampling_strategy="group_by_length", ddp_find_unused_parameters=False)


def bind_training_run(args: argparse.Namespace, manifest: bytes, settings: CriticConfig, world_size: int) -> None:
    """Reject accidental overwrites and changed data/settings when explicitly resuming."""
    options = {key: value for key, value in vars(args).items() if key not in {"resume", "output", "data"}}
    options["data"] = str(args.data)
    options["world_size"] = world_size
    record = {"options": options, "critic": settings.model_dump(), "cache_sha256": hashlib.sha256(manifest).hexdigest()}
    path = args.output / "training.json"
    if args.resume is not None:
        if json.loads(path.read_text()) != record:
            raise ValueError("saved training settings or cached data differ")
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError("resume checkpoint must belong to this output directory")
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty; use --resume or a new directory: {args.output}")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    """Train on older problems, select using validation, and leave the test split untouched."""
    load_dotenv()
    args = parse_arguments()
    set_seed(args.seed)
    raw = (args.data / MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(raw)
    cached = manifest["settings"]
    settings = CriticConfig(backbone=cached["backbone"], revision=cached["revision"], variant=args.variant,
                            reference_mode=cached["reference_mode"],
                            horizon=manifest["preparation"]["settings"]["horizon"], lora_rank=args.rank,
                            lora_alpha=args.lora_alpha, oracle_weight=args.oracle_weight,
                            correction_weight=args.correction_weight, attention=args.attention)
    dataset = cast(DatasetDict, load_from_disk(str(args.data)))
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("training requires nonempty problem-disjoint train and validation splits")
    if args.variant != "direct" and cached["oracle_fraction"] == 0:
        raise ValueError("oracle-supervised variants require nonzero oracle coverage")
    if args.variant != "direct" and not any(sum(pair[1] for pair in row[1]) for row in dataset["train"]["labels"]):
        raise ValueError("no training prefix has oracle outcomes under this coverage selection")
    tokenizer = AutoTokenizer.from_pretrained(args.data / "tokenizer", padding_side="right")
    model = build_critic(settings, torch.float32 if args.cpu else torch.bfloat16)
    if cached["max_length"] > cast(Any, model.backbone).config.max_position_embeddings:
        raise ValueError("cached maximum length exceeds the critic backbone's context limit")
    trainer = CriticTrainer(model=model, args=training_arguments(args), data_collator=PrefixCollator(tokenizer),
                            train_dataset=dataset["train"], eval_dataset=dataset["validation"], compute_metrics=compute_metrics)
    with trainer.args.main_process_first():
        if trainer.is_world_process_zero():
            bind_training_run(args, raw, settings, trainer.args.world_size)
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output / "best"))
    trainer.save_state()


if __name__ == "__main__":
    main()
