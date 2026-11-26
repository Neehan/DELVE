"""Validate and pack local generation outputs for Parquet readers or publication."""

from argparse import ArgumentParser
from pathlib import Path

from discovery_execution.harness.constants import OUTPUT_DIRECTORY
from discovery_execution.outputs.constants import PUBLISHED_DIRECTORY
from discovery_execution.outputs.packer import OutputPacker


def main() -> None:
    """Keep native generation state intact and create independently verified seed banks."""
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=PUBLISHED_DIRECTORY)
    args = parser.parse_args()
    paths = OutputPacker(args.source, args.destination).pack(None)
    print(f"Verified {len(paths)} Parquet banks in {args.destination}")


if __name__ == "__main__":
    main()
