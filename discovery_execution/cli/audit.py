"""Parse saved-run selection and launch offline auditing."""

import sys
from argparse import ArgumentParser, Namespace

from discovery_execution.audit.runner import AuditRunner
from discovery_execution.cli.arguments import add_selection_arguments
from discovery_execution.harness.constants import CONTINUATION_EXPERIMENTS, ORACLE_EXECUTION_EXPERIMENT, OUTPUT_DIRECTORY
from discovery_execution.harness.storage import model_output_directory


def parse_arguments(arguments: list[str]) -> Namespace:
    """Select a model's saved trajectories with optional experiment, problem, and seed filters."""
    parser = ArgumentParser(allow_abbrev=False)
    add_selection_arguments(parser, collecting=False)
    parser.add_argument("--problems", nargs="+")
    args = parser.parse_args(arguments)
    if args.year < 1:
        parser.error("year must be positive")
    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    if args.inject_sketch_at is not None and args.experiment != ORACLE_EXECUTION_EXPERIMENT:
        parser.error("--inject-sketch-at requires --experiment oracle-execution")
    if args.inject_sketch_at is not None:
        if args.inject_sketch_at < 0 or (args.compute_multiplier_k is not None
                                       and args.inject_sketch_at >= args.compute_multiplier_k):
            parser.error("injection must be nonnegative and before the output budget")
    if any(value is not None for value in (args.source_arm, args.source_seed, args.continue_at)):
        if args.experiment is not None and args.experiment not in CONTINUATION_EXPERIMENTS:
            parser.error("source options require a continuation experiment")
        if args.source_seed is not None and args.source_seed < 0:
            parser.error("source-seed must be nonnegative")
        if args.continue_at is not None and args.compute_multiplier_k is not None:
            if args.continue_at >= args.compute_multiplier_k:
                parser.error("continue-at must precede the output budget")
    return args


def main() -> None:
    """Delegate saved-run discovery and grading to the auditor."""
    args = parse_arguments(sys.argv[1:])
    directory = model_output_directory(OUTPUT_DIRECTORY, args.competition, args.year, args.model)
    filters = {name: value for name, value in vars(args).items()
               if name in {"experiment", "compute_multiplier_k", "inject_sketch_at", "source_arm",
                           "source_seed", "continue_at"} and value is not None}
    AuditRunner().run(directory, args.problems, args.seeds, filters)


if __name__ == "__main__":
    main()
