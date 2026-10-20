"""Process DELVE outputs into Parquet and a tokenized training cache in one command."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from discovery_execution.harness.constants import CHECKPOINT_FILENAME, CONVERSATION_FILENAME, DATASETS_DIRECTORY, META_FILENAME, SUPPORTED_MODELS
from critic_only.constants import (
    CONDITIONS, DEFAULT_BRANCH_LIMIT, DEFAULT_HORIZON, DEFAULT_MAX_LENGTH, DEFAULT_ORACLE_SUPERVISION_FRACTION,
    DEFAULT_SEED, DEFAULT_TEST_YEAR, DEFAULT_VALIDATION_FRACTION,
    PREPARED_SUBDIRECTORY, PROCESSING_DIRECTORY, TOKENIZED_SUBDIRECTORY,
)
from critic_only.data.models import PreparationConfig
from critic_only.data.processor import DatasetProcessor
from critic_only.data.tokenizer import PrefixTokenizer


def download_outputs(repository: str, revision: str, solver: str, horizon: int) -> tuple[Path, str]:
    """Download metadata first, then only the exact source and first-branch snapshots needed."""
    resolved = HfApi().dataset_info(repository, revision=revision).sha
    if resolved is None:
        raise ValueError("HF output revision could not be resolved")
    root = Path(snapshot_download(repository, repo_type="dataset", revision=resolved,
                                  allow_patterns=[f"**/{solver}/**/{META_FILENAME}"]))
    files = set()
    for path in root.rglob(META_FILENAME):
        metadata = json.loads(path.read_text())
        if metadata["model"] != solver or metadata["experiment"] not in CONDITIONS or metadata["compute_multiplier_k"] != horizon:
            continue
        fork = metadata["continue_at"]
        model_root = root / metadata["competition"] / str(metadata["year"]) / solver
        source = model_root / metadata["source_arm"] / metadata["problem_id"] / f"seed_{metadata['source_seed']}"
        needed = (path.parent / "audit.json", path.parent / CHECKPOINT_FILENAME.format(k=fork + 1),
                  source / CONVERSATION_FILENAME, source / CHECKPOINT_FILENAME.format(k=fork))
        files.update(str(item.relative_to(root)) for item in needed)
    if not files:
        raise ValueError("no matching continuation metadata in HF snapshot")
    snapshot_download(repository, repo_type="dataset", revision=resolved, allow_patterns=sorted(files))
    return root, resolved


def parse_arguments() -> argparse.Namespace:
    """Collect source, split, and tokenizer settings for one preprocessing job."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--outputs-directory", type=Path)
    source.add_argument("--outputs-repo")
    parser.add_argument("--revision", required=True, help="HF revision, or a descriptive local collection revision")
    parser.add_argument("--datasets-directory", type=Path, default=DATASETS_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=PROCESSING_DIRECTORY)
    parser.add_argument("--solver-model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backbone-revision", required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--oracle-supervision-fraction", type=float, default=DEFAULT_ORACLE_SUPERVISION_FRACTION)
    parser.add_argument("--branch-limit", type=int, default=DEFAULT_BRANCH_LIMIT, help="training branches per prefix and condition")
    args = parser.parse_args()
    if args.max_length < 1 or not 0 <= args.oracle_supervision_fraction <= 1:
        parser.error("max-length must be positive and oracle-supervision-fraction must be in [0, 1]")
    if args.branch_limit is not None and args.branch_limit < 1:
        parser.error("branch-limit must be positive")
    return args


def main() -> None:
    """Publish both datasets together after successful validation and tokenization."""
    load_dotenv()
    disable_progress_bars()
    disable_progress_bar()
    args = parse_arguments()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    revision = HfApi().model_info(args.backbone, revision=args.backbone_revision).sha
    if revision is None:
        raise ValueError("HF backbone revision could not be resolved")
    tokenizer = PrefixTokenizer(AutoTokenizer.from_pretrained(args.backbone, revision=revision, padding_side="right"))
    settings = {"backbone": args.backbone, "revision": revision, "max_length": args.max_length,
                "oracle_supervision_fraction": args.oracle_supervision_fraction,
                "branch_limit": args.branch_limit, "seed": args.seed}
    directory, revision = args.outputs_directory, args.revision
    print("Processing saved trajectories and audits.", flush=True)
    if args.outputs_repo:
        directory, revision = download_outputs(args.outputs_repo, revision, args.solver_model, args.horizon)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.destination.parent) as temporary:
        staging = Path(temporary)
        config = PreparationConfig(outputs_directory=directory, datasets_directory=args.datasets_directory,
                                   destination=staging / PREPARED_SUBDIRECTORY, solver_model=args.solver_model,
                                   horizon=args.horizon, test_year=args.test_year, validation_fraction=args.validation_fraction,
                                   split_seed=args.seed, source_revision=f"{args.outputs_repo or 'local'}@{revision}")
        DatasetProcessor(config).run()
        print("Tokenizing critic inputs and saving labels.", flush=True)
        tokenizer.run(config.destination, staging / TOKENIZED_SUBDIRECTORY, settings)
        staging.rename(args.destination)
    print(f"Saved processed data: {args.destination}", flush=True)


if __name__ == "__main__":
    main()
