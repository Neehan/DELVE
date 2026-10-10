"""Prepare local dataset partitions through the same loader used by the runner."""

import os
from argparse import ArgumentParser

from dotenv import load_dotenv

from discovery_execution.dataset import Dataset
from discovery_execution.harness.constants import (
    COMPETITIONS, DATASETS_DIRECTORY, ENV_FILE, LOGS_DIRECTORY, OUTPUT_DIRECTORY,
)

from discovery_execution.harness.models import DatasetConfig


def main() -> None:
    """Validate supplied partitions, or download missing ones from an immutable HF snapshot."""
    load_dotenv(ENV_FILE, override=False)
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--competition", required=True, choices=COMPETITIONS, nargs="+")
    parser.add_argument("--year", required=True, type=int, nargs="+")
    args = parser.parse_args()
    for directory in (DATASETS_DIRECTORY, OUTPUT_DIRECTORY, LOGS_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)
    for competition in args.competition:
        for year in args.year:
            config = DatasetConfig(
                competition=competition, year=year, datasets_directory=DATASETS_DIRECTORY,
                hf_repo=os.environ.get("HF_DATASETS_REPO"),
            )
            dataset = Dataset(config)
            print(f"Validated {competition}/{year}: {len(dataset)} problems")


if __name__ == "__main__":
    main()
