"""Common dataset and model selection arguments for collection and auditing."""

from argparse import ArgumentParser

from discovery_execution.harness.constants import (
    COMPETITIONS, DEFAULT_INJECTION, SUPPORTED_MODELS,
    DEFAULT_BLOCKS, EXPERIMENTS, MAX_BLOCKS, NO_SOURCE, SOURCE_ARMS,
)


def add_selection_arguments(parser: ArgumentParser, *, collecting: bool) -> None:
    """Define shared selectors, requiring experiment and seeds only for collection."""
    parser.add_argument("--competition", required=True, choices=COMPETITIONS)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--experiment", required=collecting,
                        choices=EXPERIMENTS)
    parser.add_argument("--compute-multiplier-k", type=int, choices=range(1, MAX_BLOCKS + 1),
                        default=DEFAULT_BLOCKS if collecting else None,
                        help=f"Output budget multiplier (collection default: {DEFAULT_BLOCKS}); optional audit filter")
    parser.add_argument("--seeds", required=collecting, type=int, nargs="+")
    parser.add_argument("--inject-sketch-at", type=int,
                        default=DEFAULT_INJECTION if collecting else None)
    parser.add_argument("--source-arm", choices=SOURCE_ARMS, default=NO_SOURCE)
    parser.add_argument("--source-seed", type=int, default=NO_SOURCE,
                        help="Original unaided trajectory seed; --seeds selects fresh branch seeds")
    parser.add_argument("--continue-at", type=int, choices=range(1, MAX_BLOCKS), default=NO_SOURCE,
                        help="Saved block boundary to branch from; budgets include this prefix")
