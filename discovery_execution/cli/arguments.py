"""Common dataset and model selection arguments for collection and auditing."""

from argparse import ArgumentParser

from discovery_execution.harness.constants import (
    COMPETITIONS, DEFAULT_INJECTION, ORACLE_EXECUTION_EXPERIMENT, SUPPORTED_MODELS,
    TOTAL_BLOCKS, UNAIDED_EXPERIMENT,
)


def add_selection_arguments(parser: ArgumentParser, *, collecting: bool) -> None:
    """Define shared selectors, requiring experiment and seeds only for collection."""
    parser.add_argument("--competition", required=True, choices=COMPETITIONS)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--experiment", required=collecting,
                        choices=(UNAIDED_EXPERIMENT, ORACLE_EXECUTION_EXPERIMENT))
    parser.add_argument("--compute-multiplier-k", type=int, choices=range(1, TOTAL_BLOCKS + 1),
                        help="Output budget multiplier; required for unaided collection, optional for auditing")
    parser.add_argument("--seeds", required=collecting, type=int, nargs="+")
    parser.add_argument("--inject-sketch-at", type=int, choices=range(TOTAL_BLOCKS),
                        default=DEFAULT_INJECTION if collecting else None)
