"""Export native solver runs to Parquet for critic setup, DE fitting or HF publication."""

from argparse import ArgumentParser
from pathlib import Path

from discovery_execution.harness.constants import OUTPUT_DIRECTORY
from discovery_execution.outputs.constants import EXPORT_SEEDS, PUBLISHED_DIRECTORY
from discovery_execution.outputs.parquet_exporter import ParquetTrajectoryExporter


def main() -> None:
    """Read source/destination CLI paths and export verified files without changing native runs."""
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--destination", type=Path, default=PUBLISHED_DIRECTORY)
    parser.add_argument("--seeds", nargs="+", type=int, default=EXPORT_SEEDS, help="export these seeds; omit to select all available seeds")
    args = parser.parse_args()
    if args.seeds is not None and (min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds))):
        parser.error("seeds must be unique nonnegative integers")
    seeds = None if args.seeds is None else tuple(args.seeds)
    paths = ParquetTrajectoryExporter(args.source, args.destination).export_trajectories(seeds)
    print(f"Exported and verified {len(paths)} trajectory Parquet files in {args.destination}")


if __name__ == "__main__":
    main()
