"""Compile local or pinned Hugging Face DELVE outputs into reusable prefix-label Parquet."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

from discovery_execution.harness.constants import CHECKPOINT_FILENAME, CONVERSATION_FILENAME, DATASETS_DIRECTORY, META_FILENAME, SUPPORTED_MODELS
from frozen_solver.cli.logging import disable_progress_bars
from frozen_solver.constants import CONDITIONS, DEFAULT_HORIZON, DEFAULT_SEED, DEFAULT_TEST_YEAR, DEFAULT_VALIDATION_FRACTION, PREPARED_DIRECTORY
from frozen_solver.data.models import PreparationConfig
from frozen_solver.data.processor import DatasetProcessor


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


def main() -> None:
    """Resolve the requested source once; publish no data to Hugging Face implicitly."""
    load_dotenv()
    disable_progress_bars()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--outputs-directory", type=Path)
    source.add_argument("--outputs-repo")
    parser.add_argument("--revision", required=True, help="HF revision, or a descriptive local collection revision")
    parser.add_argument("--datasets-directory", type=Path, default=DATASETS_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=PREPARED_DIRECTORY)
    parser.add_argument("--solver-model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    directory, revision = args.outputs_directory, args.revision
    if args.outputs_repo:
        directory, revision = download_outputs(args.outputs_repo, revision, args.solver_model, args.horizon)
    config = PreparationConfig(outputs_directory=directory, datasets_directory=args.datasets_directory,
                               destination=args.destination, solver_model=args.solver_model, horizon=args.horizon,
                               test_year=args.test_year, validation_fraction=args.validation_fraction,
                               split_seed=args.seed, source_revision=f"{args.outputs_repo or 'local'}@{revision}")
    DatasetProcessor(config).run()


if __name__ == "__main__":
    main()
