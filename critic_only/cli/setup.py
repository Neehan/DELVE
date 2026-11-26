"""Prepare local training data from saved, audited solver trajectories.

Download and pack outputs once, or reuse the packed trajectories with --reprocess. Reconstruct analysis checkpoints, then save critic token inputs with matching references. Reprocessing replaces derived files only; no Qwen weights are loaded and nothing is uploaded.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
from datasets import Dataset
from datasets.utils import disable_progress_bars
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer

from discovery_execution.audit.constants import GRADING_RULE
from critic_only.constants import DEFAULT_HORIZON, DEFAULT_MAX_LENGTH, CRITIC_PROMPT_PATH, REFERENCE_PROMPT_PATH, CRITIC_TARGET, INPUT_PREFIX_BLOCKS, TRAJECTORY_SEEDS
from critic_only.data.models import PreparationConfig
from critic_only.data.critic_input_processor import CriticInputProcessor
from critic_only.data.tokenizer import PrefixTokenizer
from critic_only.data.storage import CriticInputStorage
from critic_only.data.prepared_input_publisher import PreparedInputPublisher
from discovery_execution.processing.checkpoint_reconstructor import CheckpointReconstructor
from discovery_execution.processing.storage import read_manifest, write_json_atomic, write_parquet_atomic, file_digest

from discovery_execution.harness.constants import (
    DATASETS_DIRECTORY,
    SUPPORTED_MODELS, MAX_BLOCKS,
)
from discovery_execution.processing.constants import (PROCESSED_DIRECTORY, TRAIN_PROBLEMS, CHECKPOINT_INTERVAL_TOKENS,
    CHECKPOINT_RECONSTRUCTION, MANIFEST_FILENAME)
from discovery_execution.processing.trajectory_processor import TrajectoryProcessor
from discovery_execution.processing.problem_splits import ProblemSplits


def download_outputs(repository: str, revision: str, solver: str) -> tuple[Path, str]:
    """Download one solver's published Parquet banks at a fixed Hugging Face commit.

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
    patterns = [f"*/*/{solver}/**/seed_*.parquet"]
    root = Path(snapshot_download(repository, repo_type="dataset", revision=resolved, allow_patterns=patterns))
    return root, f"{repository}@{resolved}"


def parse_arguments() -> argparse.Namespace:
    """Read setup options and reject invalid paths, budgets, or source selections.

    Returns:
        argparse.Namespace: Validated source, dataset directory, destination, and solver selection.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--outputs-directory", type=Path)
    source.add_argument("--outputs-repo")
    parser.add_argument("--reprocess", action="store_true", help="reuse packed trajectories and splits; replace derived checkpoints and token caches after validation")
    parser.add_argument("--revision", help="required HF outputs revision; omit for local files")
    parser.add_argument("--datasets-directory", type=Path, default=DATASETS_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=PROCESSED_DIRECTORY)
    parser.add_argument("--solver-model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="final training checkpoint in analysis blocks; later source checkpoints remain saved")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()
    if args.reprocess == bool(args.outputs_directory or args.outputs_repo):
        parser.error("choose --reprocess OR one outputs source")
    if args.reprocess and args.revision is not None:
        parser.error("--reprocess preserves the saved source/model revisions")
    if not 2 <= args.horizon <= MAX_BLOCKS or args.max_length <= 0:
        parser.error("invalid horizon or maximum input length")
    if args.outputs_repo and not args.revision:
        parser.error("--outputs-repo requires --revision")
    if args.outputs_directory is not None:
        if args.revision is not None:
            parser.error("--revision applies only to --outputs-repo")
        if not args.outputs_directory.is_dir():
            parser.error("outputs directory does not exist")
    if not args.reprocess and not args.datasets_directory.is_dir():
        parser.error("datasets directory does not exist")
    exists = (args.destination / args.solver_model / MANIFEST_FILENAME).is_file()
    if args.reprocess and not exists:
        parser.error("--reprocess requires an existing processed model manifest")
    if not args.reprocess and (args.destination / args.solver_model).exists():
        parser.error("processed model directory already exists; use --reprocess")
    return args



def tokenization_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Record the exact model revision, prompt text, and input limits for this dataset.

    Args:
        args: Setup CLI settings specifying solver, final horizon, and maximum input length.

    Returns:
        dict[str, Any]: Pinned model revision, input dimension, prompts, and tokenization limits.
    """
    model = SUPPORTED_MODELS[args.solver_model]
    if args.reprocess:
        saved = read_manifest(args.destination / args.solver_model)["tokenization"]["settings"]
        if saved["model"] != model:
            raise ValueError("saved tokenizer model differs from the requested solver")
        revision = saved["revision"]
    else:
        revision = HfApi().model_info(model).sha
    if revision is None:
        raise ValueError("model revision could not be resolved")
    config = AutoConfig.from_pretrained(model, revision=revision).get_text_config()
    if args.max_length > config.max_position_embeddings:
        raise ValueError("maximum input length exceeds the model context limit")
    return {"model": model, "revision": revision, "hidden_size": config.hidden_size,
            "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS,
            "checkpoint_reconstruction": CHECKPOINT_RECONSTRUCTION,
            "horizon": args.horizon, "max_length": args.max_length,
            "target": CRITIC_TARGET, "grading_rule": GRADING_RULE,
            "prefix_blocks": INPUT_PREFIX_BLOCKS, "trajectory_seeds": list(TRAJECTORY_SEEDS),
            "critic_prompt": CRITIC_PROMPT_PATH.read_text().strip(),
            "reference_prompt": REFERENCE_PROMPT_PATH.read_text().strip()}


def prepare_critic_inputs(directory: Path, settings: dict[str, Any]) -> None:
    """Pair individual 1× histories with exact future-checkpoint outcomes, tokenize both views and save them locally.

    Args:
        directory: Completed shared trajectory collection.
        settings: Pinned model/tokenizer identity, prompts, and horizon from tokenization_settings.

    Returns:
        None. Saves split-specific token chunks, the tokenizer, and a completed tokenization manifest section.
    """
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["revision"], padding_side="right")
    checkpoints = CheckpointReconstructor(tokenizer, settings["checkpoint_interval_tokens"])
    processor = CriticInputProcessor(PreparationConfig(source_directory=directory, horizon=settings["horizon"]), checkpoints)
    formatter = PrefixTokenizer(tokenizer)
    with TemporaryDirectory(dir=directory) as temporary:
        examples = Dataset.from_generator(processor.iter_inputs, cache_dir=temporary)
        if not len(examples):
            raise ValueError("no eligible nonterminal critic inputs")
        print(f"Prepared {len(examples)} individual 1× trajectories", flush=True)
        encoded = examples.map(formatter.encode, fn_kwargs={"settings": settings},
                               remove_columns=examples.column_names, load_from_cache_file=False)
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        manifest = read_manifest(directory)
        manifest.pop("tokenization", None)
        manifest.pop("analysis_checkpoints", None)
        save_analysis_checkpoints(staging, processor.checkpoint_rows, settings, manifest)
        CriticInputStorage(staging).save_inputs(encoded, tokenizer, settings)
        PreparedInputPublisher(directory).publish(staging)
        # Close Arrow memory maps before removing their temporary files on shared filesystems.
        del encoded, examples


def save_analysis_checkpoints(staging: Path, rows: list[dict[str, Any]], settings: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Save every derived boundary once; training selects only its requested horizon.

    Args:
        staging: Temporary model directory for publication after tokenization succeeds.
        rows: Trajectory identities and all graded boundaries, including those after 5×.
        settings: Pinned interval, tokenizer and reconstruction settings.
        manifest: Raw trajectory metadata to extend without rewriting raw shards.

    Returns:
        None. Writes a content-addressed checkpoint table and the staging manifest.
    """
    path = staging / "checkpoints.parquet"
    write_parquet_atomic(path, pa.Table.from_pylist(rows))
    digest = file_digest(path)
    named = path.with_name(f"checkpoints-{digest}.parquet")
    path.rename(named)
    manifest["analysis_checkpoints"] = {"path": named.name, "sha256": digest,
        "interval_tokens": settings["checkpoint_interval_tokens"], "reconstruction": settings["checkpoint_reconstruction"],
        "tokenizer_model": settings["model"], "tokenizer_revision": settings["revision"], "trajectory_seeds": list(TRAJECTORY_SEEDS)}
    write_json_atomic(staging / MANIFEST_FILENAME, manifest)


def main() -> None:
    """Create problem splits, pack raw trajectories, and save tokenized critic examples.

    Returns:
        None. Writes fixed problem splits, full trajectory chunks, tokenized critic inputs, and their metadata locally.
    """
    load_dotenv()
    disable_progress_bars()
    args = parse_arguments()
    settings = tokenization_settings(args)
    directory = args.destination / args.solver_model
    if not args.reprocess:
        splits = ProblemSplits(args.datasets_directory, args.destination, TRAIN_PROBLEMS)
        source, revision = args.outputs_directory, None
        if args.outputs_repo:
            source, revision = download_outputs(args.outputs_repo, args.revision, args.solver_model)
        processor = TrajectoryProcessor(source, directory, args.solver_model, splits)
        processor.compile_trajectories(revision, TRAJECTORY_SEEDS)
        # Pin the tokenizer before reconstruction so a failed first preparation can be retried with --reprocess.
        manifest = read_manifest(directory)
        manifest["tokenization"] = {"settings": settings, "complete": False, "shards": {}}
        write_json_atomic(directory / MANIFEST_FILENAME, manifest)
    prepare_critic_inputs(directory, settings)
    print(f"Saved trajectories and critic inputs: {directory}", flush=True)


if __name__ == "__main__":
    main()
