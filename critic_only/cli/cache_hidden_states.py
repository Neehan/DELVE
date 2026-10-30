"""Run frozen Qwen once on saved critic inputs and cache its final-token vectors.

Save both input versions, with and without reference solutions, in files matching the token chunks. Frozen MLP experiments reuse these vectors; LoRA experiments use the tokens directly.
"""

import argparse
from pathlib import Path

import torch
from accelerate import PartialState
from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from transformers import AutoModel, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.utils.logging import disable_progress_bar

from critic_only.constants import CRITIC_SUBDIRECTORY, DEFAULT_EXTRACTION_BATCH_SIZE, ENCODING_LOADER_WORKERS, ENCODING_PREFETCH_BATCHES
from critic_only.data.storage import CriticInputStorage
from critic_only.hidden_states.encoder import HiddenStateEncoder
from critic_only.hidden_states.runner import HiddenStateRunner
from critic_only.hidden_states.storage import HiddenStateStorage


def parse_arguments() -> argparse.Namespace:
    """Read the prepared-data path, encoding batch size, and device option.

    Returns:
        argparse.Namespace: Prepared data directory, forward batch size, and CPU flag. Input text and model identity come from setup.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EXTRACTION_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true", help="FP32 CPU encoding for small smoke tests")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    return args


def main() -> None:
    """Run HF prediction across available devices and save each completed chunk.

    Returns:
        None. Saves both critic input representations without generating text or updating model weights.
    """
    load_dotenv()
    disable_progress_bars()
    disable_progress_bar()
    args = parse_arguments()
    distributed = PartialState(cpu=args.cpu)
    inputs = CriticInputStorage(args.data)
    outputs = HiddenStateStorage(args.data)
    if distributed.is_main_process:
        outputs.prepare_encoding(args.cpu)
    distributed.wait_for_everyone()
    manifest = inputs.load_metadata()
    settings = manifest["tokenization"]["settings"]
    tokenizer = inputs.load_tokenizer()
    if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
        raise ValueError("hidden-state encoding requires right padding and a pad token")
    model = AutoModel.from_pretrained(settings["model"], revision=settings["revision"],
                                      dtype=torch.float32 if args.cpu else torch.bfloat16,
                                      attn_implementation=manifest["features"]["settings"]["attention"])
    prediction = TrainingArguments(output_dir=str(args.data / CRITIC_SUBDIRECTORY),
        per_device_eval_batch_size=args.batch_size, use_cpu=args.cpu, bf16=not args.cpu,
        dataloader_num_workers=ENCODING_LOADER_WORKERS, dataloader_prefetch_factor=ENCODING_PREFETCH_BATCHES,
        dataloader_multiprocessing_context="spawn", dataloader_pin_memory=not args.cpu,
        accelerator_config={"non_blocking": not args.cpu}, label_names=[],
        disable_tqdm=True, report_to="none")
    trainer = Trainer(model=HiddenStateEncoder(model), args=prediction,
                      data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer)
    HiddenStateRunner(inputs, outputs, trainer).run()
    distributed.wait_for_everyone()
    if distributed.is_main_process:
        outputs.complete_cache()
        print(f"Saved critic hidden states: {args.data}", flush=True)


if __name__ == "__main__":
    main()
