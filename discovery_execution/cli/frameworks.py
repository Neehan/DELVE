"""Fit SG, DE, R-SG, and R-DE from compiled binary audits."""

import logging
import sys
from argparse import ArgumentParser

from discovery_execution.frameworks.models import InputSelection
from discovery_execution.frameworks.runner import FrameworkRunner
from discovery_execution.frameworks.utils.constants import FRAMEWORK_NAMES
from discovery_execution.frameworks.utils.dataloader import DataLoader
from discovery_execution.harness.constants import COMPETITIONS, FAILURE_EXIT_CODE, LOG_FORMAT, OUTPUT_DIRECTORY, SUPPORTED_MODELS


def parse_arguments(arguments: list[str]) -> InputSelection:
    """Select competition/year partitions and framework names without inference settings."""
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--competition", dest="competitions", required=True, nargs="+", choices=COMPETITIONS)
    parser.add_argument("--year", dest="years", required=True, nargs="+", type=int)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORK_NAMES, default=list(FRAMEWORK_NAMES))
    try:
        return InputSelection(**vars(parser.parse_args(arguments)))
    except ValueError as error:
        parser.error(str(error))


def main() -> None:
    """Run local fitting and report errors without contacting an inference server."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    selection = parse_arguments(sys.argv[1:])
    try:
        runner = FrameworkRunner(DataLoader(OUTPUT_DIRECTORY), OUTPUT_DIRECTORY)
        for path in runner.run(selection):
            logging.info("Saved %s", path)
    except (ValueError, OSError, RuntimeError, ArithmeticError) as error:
        logging.error("Framework fitting failed: %s", error)
        raise SystemExit(FAILURE_EXIT_CODE) from error


if __name__ == "__main__":
    main()
