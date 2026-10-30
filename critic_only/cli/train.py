"""Train a critic using cached Qwen vectors or Qwen with LoRA adapters.

Average losses with and without reference solutions. Evaluate on unaided validation trajectories, save resumable checkpoints, and keep the best validation model. The test split is not loaded.
"""

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from transformers import TrainingArguments, set_seed

from transformers.utils.logging import disable_progress_bar

from critic_only.constants import (
    DEFAULT_ACCUMULATION_STEPS, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    DEFAULT_LOGGING_STEPS, DEFAULT_HIDDEN_SIZE, DEFAULT_ORACLE_WEIGHT, DEFAULT_SEED,
    DEFAULT_TRAINING_PROBLEM_FRACTION, DEFAULT_VARIANT, DEFAULT_WARMUP_RATIO, DEFAULT_WEIGHT_DECAY,
    TRAINING_DIRECTORY, VARIANTS, DEFAULT_ORACLE_SUPERVISION_FRACTION,
    TRAINING_MODES, DEFAULT_TRAINING_MODE, DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_BATCH_SIZE, DEFAULT_LORA_ACCUMULATION_STEPS,
)
from critic_only.data.critic_input_processor import select_training_problems, select_unaided_examples, selection_fraction
from critic_only.data.hidden_state_dataset import HiddenStateDataset
from critic_only.data.critic_batcher import CriticBatcher
from critic_only.data.storage import CriticInputStorage
from critic_only.data.dataset_loader import load_critic_splits, input_signature
from critic_only.eval.metrics import compute_metrics
from critic_only.models.loader import create_critic
from critic_only.models.models import CriticConfig
from critic_only.trainer.base_trainer import BaseTrainer


def parse_arguments() -> argparse.Namespace:
    """Read and validate training options before loading data or model weights.

    Returns:
        argparse.Namespace: Validated cache paths, critic variant, training settings, and optional resume checkpoint.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=TRAINING_DIRECTORY)
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--oracle-supervision-fraction", type=float, default=DEFAULT_ORACLE_SUPERVISION_FRACTION)
    parser.add_argument("--oracle-weight", type=float, default=DEFAULT_ORACLE_WEIGHT)
    parser.add_argument("--training-problem-fraction", type=float, default=DEFAULT_TRAINING_PROBLEM_FRACTION,
                        help="fraction of training problems to retain for ablations; validation/test stay unchanged")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true", help="use CPU execution for small local checks")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default=DEFAULT_TRAINING_MODE)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    args = parser.parse_args()
    configure_training_mode(args)
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.logging_steps, args.learning_rate) <= 0:
        parser.error("training sizes and learning rate must be positive")
    if args.hidden_size <= 0 or args.oracle_weight < 0 or not 0 <= args.oracle_supervision_fraction <= 1:
        parser.error("invalid MLP size, oracle weight, or oracle-supervision-fraction")
    if not 0 < args.training_problem_fraction <= 1:
        parser.error("--training-problem-fraction must be in (0, 1]")
    return args



def configure_training_mode(args: argparse.Namespace) -> None:
    """Fill in batch and adapter defaults for frozen or LoRA training.

    Args:
        args: Parsed CLI settings, updated in place before training validation.

    Returns:
        None. Frozen mode uses large vector batches; LoRA uses small token batches with accumulation.
    """
    lora = args.training_mode == "lora"
    if not lora and (args.lora_rank is not None or args.lora_alpha is not None):
        raise ValueError("LoRA options require --training-mode lora")
    if args.batch_size is None:
        args.batch_size = DEFAULT_LORA_BATCH_SIZE if lora else DEFAULT_BATCH_SIZE
    if args.accumulation_steps is None:
        args.accumulation_steps = DEFAULT_LORA_ACCUMULATION_STEPS if lora else DEFAULT_ACCUMULATION_STEPS
    if lora:
        args.lora_rank = DEFAULT_LORA_RANK if args.lora_rank is None else args.lora_rank
        args.lora_alpha = DEFAULT_LORA_ALPHA if args.lora_alpha is None else args.lora_alpha
        if min(args.lora_rank, args.lora_alpha) <= 0:
            raise ValueError("LoRA rank and alpha must be positive")


def training_arguments(args: argparse.Namespace) -> TrainingArguments:
    """Configure Hugging Face training, validation, logging, and checkpoints.

    Args:
        args: Validated CLI settings for optimization, batching, output, device, and logging.

    Returns:
        TrainingArguments: Hugging Face configuration for training, epoch evaluation, checkpoint selection, and resume.
    """
    return TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs,
                             learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
                             per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation_steps,
                             bf16=args.training_mode == "lora" and not args.cpu, use_cpu=args.cpu, optim="adamw_torch" if args.cpu else "adamw_torch_fused",
                             warmup_steps=DEFAULT_WARMUP_RATIO, weight_decay=DEFAULT_WEIGHT_DECAY,
                             eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="terminal_nll", greater_is_better=False,
                             logging_strategy="steps", logging_steps=args.logging_steps, logging_first_step=True,
                             disable_tqdm=True, report_to="none", seed=args.seed,
                             remove_unused_columns=False, label_names=["labels"],
                             ddp_find_unused_parameters=False)


def validate_training_run(args: argparse.Namespace, settings: CriticConfig, world_size: int) -> dict[str, Any]:
    """Reject an occupied output directory or a resume with changed settings.

    Args:
        args: CLI settings, including the output directory and optional resume checkpoint.
        settings: Critic configuration and selected input-cache identity.
        world_size: Number of distributed training processes.

    Returns:
        dict[str, Any]: Reproducibility record to write to training.json; raises if an existing run is incompatible.
    """
    options = {key: value for key, value in vars(args).items() if key not in {"resume", "output", "data"}}
    options["data"] = str(args.data)
    options["world_size"] = world_size
    record = {"options": options, "critic": settings.model_dump()}
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
    """Train the selected critic and save the model with the best validation score.

    Returns:
        None. Trains the selected critic, writes metrics and resumable checkpoints, and saves the best validation model.
    """
    load_dotenv()
    disable_progress_bars()
    disable_progress_bar()
    args = parse_arguments()
    set_seed(args.seed)
    dataset, manifest = load_critic_splits(args.data, ("train", "validation"), args.training_mode)
    cached = manifest["tokenization"]["settings"]
    settings = CriticConfig(model=cached["model"], revision=cached["revision"], variant=args.variant,
                            horizon=cached["horizon"], input_size=cached["hidden_size"],
                            hidden_size=args.hidden_size, data_signature=input_signature(manifest, args.training_mode),
                            training_mode=args.training_mode, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                            oracle_weight=args.oracle_weight)
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("training requires nonempty problem-disjoint train and validation splits")
    dataset["train"] = select_training_problems(dataset["train"], args.training_problem_fraction)
    dataset["train"] = dataset["train"].filter(lambda oracle, key: not oracle or
        selection_fraction(f"oracle:{key}", DEFAULT_SEED) < args.oracle_supervision_fraction,
        input_columns=["oracle", "problem_key"])
    if args.variant != "ppo-critic" and not any(dataset["train"]["oracle"]):
        raise ValueError("no training prefix has oracle outcomes under this coverage selection")
    if args.variant == "ppo-critic":
        dataset["train"] = select_unaided_examples(dataset["train"])
    dataset["validation"] = select_unaided_examples(dataset["validation"])
    training = training_arguments(args)
    record = validate_training_run(args, settings, training.world_size) if training.process_index == 0 else None
    model = create_critic(settings, args.cpu)
    tokenizer = CriticInputStorage(args.data).load_tokenizer() if args.training_mode == "lora" else None
    trainer = BaseTrainer(model=model, args=training, data_collator=CriticBatcher(tokenizer),
                          train_dataset=HiddenStateDataset(dataset["train"]) if args.training_mode == "frozen" else dataset["train"],
                          eval_dataset=HiddenStateDataset(dataset["validation"]) if args.training_mode == "frozen" else dataset["validation"],
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
