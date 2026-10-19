"""Create a tokenizer-specific Arrow cache once, before any training epochs."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from frozen_solver.cli.logging import disable_progress_bars
from frozen_solver.constants import (
    DEFAULT_BRANCH_LIMIT, DEFAULT_MAX_LENGTH, DEFAULT_ORACLE_FRACTION, DEFAULT_REFERENCE_MODE, DEFAULT_SEED,
    PREPARED_DIRECTORY, TOKENIZED_DIRECTORY,
)
from frozen_solver.data.tokenizer import PrefixTokenizer


def main() -> None:
    """Pin tokenizer/model revision and reject incompatible or silently truncated critic inputs."""
    load_dotenv()
    disable_progress_bars()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PREPARED_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=TOKENIZED_DIRECTORY)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--reference-mode", choices=("none", "solution"), default=DEFAULT_REFERENCE_MODE)
    parser.add_argument("--oracle-fraction", type=float, default=DEFAULT_ORACLE_FRACTION)
    parser.add_argument("--branch-limit", type=int, default=DEFAULT_BRANCH_LIMIT, help="training branches per prefix and condition")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.max_length < 1 or not 0 <= args.oracle_fraction <= 1:
        parser.error("max-length must be positive and oracle-fraction must be in [0, 1]")
    if args.branch_limit is not None and args.branch_limit < 1:
        parser.error("branch-limit must be positive")
    revision = HfApi().model_info(args.backbone, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, revision=revision, padding_side="right")
    if tokenizer.pad_token_id is None:
        parser.error("backbone tokenizer must define a pad token")
    settings = {"backbone": args.backbone, "revision": revision, "max_length": args.max_length,
                "reference_mode": args.reference_mode, "oracle_fraction": args.oracle_fraction,
                "branch_limit": args.branch_limit, "seed": args.seed}
    PrefixTokenizer(tokenizer).run(args.source, args.destination, settings)


if __name__ == "__main__":
    main()
