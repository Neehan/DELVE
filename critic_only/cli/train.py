"""Train matched LoRA critics with Hugging Face Trainer on cached prefixes."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from datasets import DatasetDict, load_from_disk
from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, TrainingArguments, set_seed

from transformers.utils.logging import disable_progress_bar

from critic_only.constants import (
    DEFAULT_ACCUMULATION_STEPS, DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_LOGGING_STEPS, DEFAULT_LORA_ALPHA, DEFAULT_LORA_RANK,
    DEFAULT_ORACLE_WEIGHT, DEFAULT_SEED, DEFAULT_TRAINING_PROBLEM_FRACTION, DEFAULT_VARIANT,
    DEFAULT_WARMUP_RATIO, DEFAULT_WEIGHT_DECAY,
    MANIFEST_FILENAME, TOKENIZED_DIRECTORY, TRAINING_DIRECTORY, VARIANTS,
)
from critic_only.data.processor import select_training_problems, select_unaided_examples
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.eval.metrics import compute_metrics
from critic_only.models.loader import create_critic_from_pretrained
from critic_only.models.models import CriticConfig
from critic_only.trainer.base_trainer import BaseTrainer


def parse_arguments() -> argparse.Namespace:
    """Expose experiment choices without reimplementing distributed launch configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TOKENIZED_DIRECTORY)
    parser.add_argument("--output", type=Path, default=TRAINING_DIRECTORY)
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--oracle-weight", type=float, default=DEFAULT_ORACLE_WEIGHT)
    parser.add_argument("--training-problem-fraction", type=float, default=DEFAULT_TRAINING_PROBLEM_FRACTION,
                        help="fraction of training problems to retain for ablations; validation/test stay unchanged")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--accumulation-steps", type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true", help="float32 CPU smoke tests only")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.logging_steps, args.learning_rate) <= 0:
        parser.error("training sizes and learning rate must be positive")
    if not 0 < args.training_problem_fraction <= 1:
        parser.error("--training-problem-fraction must be in (0, 1]")
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
                             logging_strategy="steps", logging_steps=args.logging_steps, logging_first_step=True,
                             disable_tqdm=True, report_to="none", seed=args.seed,
                             remove_unused_columns=False, label_names=["labels"],
                             train_sampling_strategy="group_by_length", ddp_find_unused_parameters=False)


def validate_training_run(args: argparse.Namespace, manifest: bytes, settings: CriticConfig, world_size: int) -> dict[str, Any]:
    """Check output ownership and resume settings before loading weights; return the record to save later."""
    options = {key: value for key, value in vars(args).items() if key not in {"resume", "output", "data"}}
    options["data"] = str(args.data)
    options["world_size"] = world_size
    record = {"options": options, "critic": settings.model_dump(), "cache_sha256": hashlib.sha256(manifest).hexdigest()}
    cache = json.loads(manifest)
    record["data"] = {"tokenization": cache["settings"], "preparation": cache["preparation"]["settings"]}
    path = args.output / "training.json"
    if args.resume is not None:
        if not args.resume.is_dir():
            raise FileNotFoundError(args.resume)
        if json.loads(path.read_text()) != record:
            raise ValueError("saved training settings or cached data differ")
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError("resume checkpoint must belong to this output directory")
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty; use --resume or a new directory: {args.output}")
    return record


def main() -> None:
    """Train on older problems, select using validation, and leave the test split untouched."""
    load_dotenv()
    disable_progress_bars()
    disable_progress_bar()
    args = parse_arguments()
    set_seed(args.seed)
    manifest_bytes = (args.data / MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    cached = manifest["settings"]
    settings = CriticConfig(model=cached["model"], revision=cached["revision"], variant=args.variant,
                            horizon=manifest["preparation"]["settings"]["horizon"], lora_rank=args.rank,
                            lora_alpha=args.lora_alpha, oracle_weight=args.oracle_weight)
    dataset = cast(DatasetDict, load_from_disk(str(args.data)))
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("training requires nonempty problem-disjoint train and validation splits")
    dataset["train"] = select_training_problems(dataset["train"], args.training_problem_fraction)
    if args.variant != "ppo-critic" and cached["oracle_supervision_fraction"] == 0:
        raise ValueError("PPOOracleCritic and DECritic require nonzero oracle coverage")
    if args.variant != "ppo-critic" and not any(dataset["train"]["oracle"]):
        raise ValueError("no training prefix has oracle outcomes under this coverage selection")
    if args.variant == "ppo-critic":
        dataset["train"] = select_unaided_examples(dataset["train"])
    dataset["validation"] = select_unaided_examples(dataset["validation"])
    training = training_arguments(args)
    record = validate_training_run(args, manifest_bytes, settings, training.world_size) if training.process_index == 0 else None
    model_config = AutoConfig.from_pretrained(settings.model, revision=settings.revision)
    if cached["max_length"] > model_config.get_text_config().max_position_embeddings:
        raise ValueError("cached maximum length exceeds the critic model's context limit")
    tokenizer = AutoTokenizer.from_pretrained(args.data / "tokenizer", padding_side="right")
    model = create_critic_from_pretrained(settings, torch.float32 if args.cpu else torch.bfloat16)
    trainer = BaseTrainer(model=model, args=training, data_collator=PrefixTokenizer(tokenizer).pad_batch,
                          train_dataset=dataset["train"], eval_dataset=dataset["validation"],
                          compute_metrics=partial(compute_metrics, problem_keys=list(dataset["validation"]["problem_key"])))
    with trainer.args.main_process_first():
        if trainer.is_world_process_zero():
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output / "best"))
    trainer.save_state()


if __name__ == "__main__":
    main()
