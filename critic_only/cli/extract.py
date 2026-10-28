"""Cache exact critic tokens, then extract frozen Qwen vectors with resumable GPU shards."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import torch
from accelerate import PartialState
from datasets import DatasetDict, load_from_disk
from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from critic_only.constants import (
    CPU_ATTENTION, CRITIC_PROMPT_PATH, CRITIC_SUBDIRECTORY, DEFAULT_ATTENTION, DEFAULT_EXTRACTION_BATCH_SIZE,
    DEFAULT_HORIZON, DEFAULT_MAX_LENGTH, FEATURE_SHARD_ROWS, FEATURE_SUBDIRECTORY, REFERENCE_PROMPT_PATH,
    TOKENIZED_SUBDIRECTORY,
)
from discovery_execution.processing.constants import MANIFEST_FILENAME
from critic_only.data.extractor import HiddenStateExtractor
from critic_only.data.models import PreparationConfig
from critic_only.data.processor import CriticInputProcessor
from critic_only.data.tokenizer import PrefixTokenizer
from discovery_execution.harness.constants import MAX_BLOCKS, SUPPORTED_MODELS
from discovery_execution.processing.storage import file_digest, write_json_atomic


def parse_arguments() -> argparse.Namespace:
    """Select a processed model directory and extraction batch size; model identity comes from setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EXTRACTION_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true", help="FP32 CPU extraction for small smoke tests")
    args = parser.parse_args()
    if not 2 <= args.horizon <= MAX_BLOCKS or min(args.max_length, args.batch_size) <= 0:
        parser.error("invalid horizon, max-length, or batch-size")
    return args


def prepare_token_cache(args: argparse.Namespace) -> None:
    """Freeze prompt/model settings before tokenization; reuse only matching completed token caches."""
    path = args.data / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    processor = CriticInputProcessor(PreparationConfig(source_directory=args.data, horizon=args.horizon))
    requested = {"horizon": args.horizon, "max_length": args.max_length,
                 "critic_prompt": CRITIC_PROMPT_PATH.read_text().strip(),
                 "reference_prompt": REFERENCE_PROMPT_PATH.read_text().strip()}
    if "tokenization" in manifest:
        settings = manifest["tokenization"]["settings"]
        if any(settings[key] != value for key, value in requested.items()):
            raise ValueError("cached critic prompt or tokenization settings differ; use a new processed directory")
    else:
        model = SUPPORTED_MODELS[manifest["solver_model"]]
        revision = HfApi().model_info(model).sha
        if revision is None:
            raise ValueError("model revision could not be resolved")
        settings = {**requested, "model": model, "revision": revision}
        manifest["tokenization"] = {"settings": settings, "complete": False}
        write_json_atomic(path, manifest)
    destination = args.data / CRITIC_SUBDIRECTORY / TOKENIZED_SUBDIRECTORY
    if not destination.exists():
        tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["revision"], padding_side="right")
        PrefixTokenizer(tokenizer).cache_tokens(processor, destination, settings)
    checksums = {str(file.relative_to(destination)): file_digest(file) for file in sorted(destination.rglob("*")) if file.is_file()}
    if manifest["tokenization"]["complete"] and checksums != manifest["tokenization"]["files"]:
        raise ValueError("token cache contents changed")
    manifest["tokenization"].update(complete=True, files=checksums)
    write_json_atomic(path, manifest)


def prepare_extraction(args: argparse.Namespace) -> None:
    """Bind cached features to exact tokens, transformer revision, precision, and pooling convention."""
    path = args.data / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    settings = manifest["tokenization"]["settings"]
    config = AutoConfig.from_pretrained(settings["model"], revision=settings["revision"]).get_text_config()
    if args.max_length > config.max_position_embeddings:
        raise ValueError("cached maximum length exceeds the model context limit")
    token_signature = hashlib.sha256(json.dumps(manifest["tokenization"], sort_keys=True).encode()).hexdigest()
    extraction = {"tokenization_sha256": token_signature, "precision": "float32" if args.cpu else "bfloat16",
                  "attention": CPU_ATTENTION if args.cpu else DEFAULT_ATTENTION,
                  "pooling": "last_non_padding_token", "hidden_size": config.hidden_size,
                  "shard_rows": FEATURE_SHARD_ROWS}
    signature = hashlib.sha256(json.dumps(extraction, sort_keys=True).encode()).hexdigest()
    if "features" in manifest and manifest["features"]["signature"] != signature:
        raise ValueError("feature extraction settings differ")
    if "features" not in manifest:
        manifest["features"] = {"signature": signature, "settings": extraction, "complete": False, "shards": {}}
        write_json_atomic(path, manifest)


def publish_feature_manifest(directory: Path, datasets: DatasetDict) -> None:
    """Require every expected shard before marking the model-level feature cache complete."""
    path = directory / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    root = directory / CRITIC_SUBDIRECTORY / FEATURE_SUBDIRECTORY
    checksums = {}
    for split, dataset in datasets.items():
        for index in range((len(dataset) + FEATURE_SHARD_ROWS - 1) // FEATURE_SHARD_ROWS):
            shard = root / str(split) / f"part-{index:05d}.parquet"
            if not shard.is_file():
                raise ValueError(f"missing feature shard: {shard}")
            checksums[str(shard.relative_to(root))] = file_digest(shard)
    previous = manifest["features"]
    if previous["complete"] and checksums != previous["shards"]:
        raise ValueError("completed feature cache changed")
    manifest["features"].update(complete=True, shards=checksums)
    write_json_atomic(path, manifest)


def main() -> None:
    """Use Accelerate for rank/device placement; each GPU writes distinct completed shards."""
    load_dotenv()
    disable_progress_bars()
    disable_progress_bar()
    args = parse_arguments()
    distributed = PartialState(cpu=args.cpu)
    if distributed.is_main_process:
        prepare_token_cache(args)
        prepare_extraction(args)
    distributed.wait_for_everyone()
    manifest = json.loads((args.data / MANIFEST_FILENAME).read_text())
    settings = manifest["tokenization"]["settings"]
    tokenized = args.data / CRITIC_SUBDIRECTORY / TOKENIZED_SUBDIRECTORY
    datasets = cast(DatasetDict, load_from_disk(str(tokenized)))
    tokenizer = AutoTokenizer.from_pretrained(tokenized / "tokenizer", padding_side="right")
    model = AutoModel.from_pretrained(settings["model"], revision=settings["revision"],
                                      dtype=torch.float32 if args.cpu else torch.bfloat16,
                                      attn_implementation=manifest["features"]["settings"]["attention"])
    model.to(distributed.device)
    extractor = HiddenStateExtractor(model, tokenizer, args.batch_size)
    extractor.extract_shards(datasets, args.data / CRITIC_SUBDIRECTORY / FEATURE_SUBDIRECTORY,
                             manifest["features"]["signature"], distributed.process_index, distributed.num_processes)
    distributed.wait_for_everyone()
    if distributed.is_main_process:
        publish_feature_manifest(args.data, datasets)
        print(f"Saved critic features: {args.data}", flush=True)


if __name__ == "__main__":
    main()
