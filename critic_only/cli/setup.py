"""Prepare local training data from saved, audited solver trajectories.

Download outputs if requested, reuse fixed problem splits, and write full trajectories as Parquet chunks. Then build critic inputs with and without matching reference solutions and save their token IDs. This command does not load Qwen weights or upload files.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from datasets import Dataset
from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer

from critic_only.constants import DEFAULT_HORIZON, DEFAULT_MAX_LENGTH, CRITIC_PROMPT_PATH, REFERENCE_PROMPT_PATH
from critic_only.data.models import PreparationConfig
from critic_only.data.critic_input_processor import CriticInputProcessor
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.data.storage import CriticInputStorage

from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CHECKPOINT_FILENAME, CONVERSATION_FILENAME, DATASETS_DIRECTORY, META_FILENAME,
    SUPPORTED_MODELS, MAX_BLOCKS,
)
from discovery_execution.processing.constants import PROCESSED_DIRECTORY, TRAIN_PROBLEMS
from discovery_execution.processing.trajectory_processor import TrajectoryProcessor
from discovery_execution.processing.problem_splits import ProblemSplits


def download_outputs(repository: str, revision: str, solver: str) -> tuple[Path, str]:
    """Download one solver's raw output files at a fixed Hugging Face commit.

    Args:
        repository: Hugging Face dataset repository containing solver outputs.
        revision: Requested repository revision, resolved to an immutable commit before downloading.
        solver: Solver identifier used in the output directory structure.

    Returns:
        tuple[Path, str]: Local snapshot directory and its repository@commit identifier.
    """
    resolved = HfApi().dataset_info(repository, revision=revision).sha
    if resolved is None:
        raise ValueError("HF output revision could not be resolved")
    names = [META_FILENAME, CONVERSATION_FILENAME, AUDIT_FILENAME, CHECKPOINT_FILENAME.format(k="*")]
    patterns = [f"*/*/{solver}/**/{name}" for name in names]
    root = Path(snapshot_download(repository, repo_type="dataset", revision=resolved, allow_patterns=patterns))
    return root, f"{repository}@{resolved}"


def parse_arguments() -> argparse.Namespace:
    """Read setup options and reject invalid paths, budgets, or source selections.

    Returns:
        argparse.Namespace: Validated source, dataset directory, destination, and solver selection.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--outputs-directory", type=Path)
    source.add_argument("--outputs-repo")
    parser.add_argument("--revision", help="required HF outputs revision; omit for local files")
    parser.add_argument("--datasets-directory", type=Path, default=DATASETS_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=PROCESSED_DIRECTORY)
    parser.add_argument("--solver-model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()
    if not 2 <= args.horizon <= MAX_BLOCKS or args.max_length <= 0:
        parser.error("invalid horizon or maximum input length")
    if args.outputs_repo and not args.revision:
        parser.error("--outputs-repo requires --revision")
    if args.outputs_directory is not None:
        if args.revision is not None:
            parser.error("--revision applies only to --outputs-repo")
        if not args.outputs_directory.is_dir():
            parser.error("outputs directory does not exist")
    if not args.datasets_directory.is_dir():
        parser.error("datasets directory does not exist")
    if (args.destination / args.solver_model).exists():
        parser.error("processed model directory already exists")
    return args



def tokenization_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Record the exact model revision, prompt text, and input limits for this dataset.

    Args:
        args: Setup CLI settings specifying solver, final horizon, and maximum input length.

    Returns:
        dict[str, Any]: Pinned model revision, input dimension, prompts, and tokenization limits.
    """
    model = SUPPORTED_MODELS[args.solver_model]
    revision = HfApi().model_info(model).sha
    if revision is None:
        raise ValueError("model revision could not be resolved")
    config = AutoConfig.from_pretrained(model, revision=revision).get_text_config()
    if args.max_length > config.max_position_embeddings:
        raise ValueError("maximum input length exceeds the model context limit")
    return {"model": model, "revision": revision, "hidden_size": config.hidden_size,
            "horizon": args.horizon, "max_length": args.max_length,
            "critic_prompt": CRITIC_PROMPT_PATH.read_text().strip(),
            "reference_prompt": REFERENCE_PROMPT_PATH.read_text().strip()}


def prepare_critic_inputs(directory: Path, settings: dict[str, Any]) -> None:
    """Select branch histories, tokenize both reference versions, and save them locally.

    Args:
        directory: Completed shared trajectory collection.
        settings: Pinned model/tokenizer identity, prompts, and horizon from tokenization_settings.

    Returns:
        None. Saves split-specific token chunks, the tokenizer, and a completed tokenization manifest section.
    """
    processor = CriticInputProcessor(PreparationConfig(source_directory=directory, horizon=settings["horizon"]))
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["revision"], padding_side="right")
    formatter = PrefixTokenizer(tokenizer)
    with TemporaryDirectory(dir=directory) as temporary:
        examples = Dataset.from_generator(processor.iter_inputs, cache_dir=temporary)
        if not len(examples):
            raise ValueError("no eligible nonterminal critic inputs")
        encoded = examples.map(formatter.encode, fn_kwargs={"settings": settings},
                               remove_columns=examples.column_names, load_from_cache_file=False)
        CriticInputStorage(directory).save_inputs(encoded, tokenizer, settings)


def main() -> None:
    """Create problem splits, pack raw trajectories, and save tokenized critic examples.

    Returns:
        None. Writes fixed problem splits, full trajectory chunks, tokenized critic inputs, and their metadata locally.
    """
    load_dotenv()
    disable_progress_bars()
    args = parse_arguments()
    settings = tokenization_settings(args)
    splits = ProblemSplits(args.datasets_directory, args.destination, TRAIN_PROBLEMS)
    source, revision = args.outputs_directory, None
    if args.outputs_repo:
        source, revision = download_outputs(args.outputs_repo, args.revision, args.solver_model)
    processor = TrajectoryProcessor(source, args.destination / args.solver_model, args.solver_model, splits)
    processor.compile_trajectories(revision)
    prepare_critic_inputs(processor.destination, settings)
    print(f"Saved trajectories and critic inputs: {processor.destination}", flush=True)


if __name__ == "__main__":
    main()
