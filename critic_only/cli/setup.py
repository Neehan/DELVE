"""Compile local or pinned Hugging Face trajectories into shared Parquet splits; never upload."""

import argparse
from pathlib import Path

from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CHECKPOINT_FILENAME, CONVERSATION_FILENAME, DATASETS_DIRECTORY, META_FILENAME,
    SUPPORTED_MODELS,
)
from discovery_execution.processing.constants import PROCESSED_DIRECTORY, TRAIN_PROBLEMS
from discovery_execution.processing.processor import TrajectoryProcessor
from discovery_execution.processing.splits import ProblemSplits


def download_outputs(repository: str, revision: str, solver: str) -> tuple[Path, str]:
    """Resolve one immutable revision and download only the selected model's collection artifacts."""
    resolved = HfApi().dataset_info(repository, revision=revision).sha
    if resolved is None:
        raise ValueError("HF output revision could not be resolved")
    names = [META_FILENAME, CONVERSATION_FILENAME, AUDIT_FILENAME, CHECKPOINT_FILENAME.format(k="*")]
    patterns = [f"*/*/{solver}/**/{name}" for name in names]
    root = Path(snapshot_download(repository, repo_type="dataset", revision=resolved, allow_patterns=patterns))
    return root, f"{repository}@{resolved}"


def parse_arguments() -> argparse.Namespace:
    """Choose a collection and local destination without loading a model or tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--outputs-directory", type=Path)
    source.add_argument("--outputs-repo")
    parser.add_argument("--revision", help="required HF outputs revision; omit for local files")
    parser.add_argument("--datasets-directory", type=Path, default=DATASETS_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=PROCESSED_DIRECTORY)
    parser.add_argument("--solver-model", choices=SUPPORTED_MODELS, required=True)
    args = parser.parse_args()
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


def main() -> None:
    """Publish reusable trajectory shards and a single fixed problem assignment locally."""
    load_dotenv()
    disable_progress_bars()
    args = parse_arguments()
    splits = ProblemSplits(args.datasets_directory, args.destination, TRAIN_PROBLEMS)
    source, revision = args.outputs_directory, None
    if args.outputs_repo:
        source, revision = download_outputs(args.outputs_repo, args.revision, args.solver_model)
    processor = TrajectoryProcessor(source, args.destination / args.solver_model, args.solver_model, splits)
    processor.compile_trajectories(revision)
    print(f"Saved shared trajectories: {processor.destination}", flush=True)


if __name__ == "__main__":
    main()
