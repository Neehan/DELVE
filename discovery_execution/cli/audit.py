"""Parse saved-run selection and launch offline auditing."""

import sys
from argparse import ArgumentParser
from pathlib import Path

from discovery_execution.audit.runner import AuditRunner


def parse_arguments(arguments: list[str]) -> tuple[Path, list[str] | None, list[int] | None]:
    """Select an output subtree and optional saved problem and seed identifiers."""
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Saved run, arm, problem, or seed directory to audit")
    parser.add_argument("--problems", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(arguments)
    return args.run_dir, args.problems, args.seeds


def main() -> None:
    """Delegate saved-run discovery and grading to the auditor."""
    directory, problems, seeds = parse_arguments(sys.argv[1:])
    AuditRunner().run(directory, problems, seeds)


if __name__ == "__main__":
    main()
