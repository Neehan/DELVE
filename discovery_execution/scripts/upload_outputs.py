"""Upload already exported trajectory Parquet files to the configured HF outputs repository."""

import os
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from discovery_execution.harness.constants import ENV_FILE
from discovery_execution.outputs.constants import PUBLISHED_DIRECTORY
from discovery_execution.outputs.parquet_uploader import ParquetTrajectoryUploader


def main() -> None:
    """Read HF credentials and the exported directory, then publish its existing Parquet files."""
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--directory", type=Path, default=PUBLISHED_DIRECTORY, help="directory produced by export_outputs")
    args = parser.parse_args()
    load_dotenv(ENV_FILE, override=False)
    repository = os.environ.get("HF_OUTPUTS_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repository or not repository.strip() or not token or not token.strip():
        raise ValueError("Set HF_OUTPUTS_REPO and HF_TOKEN before uploading outputs")
    repository, token = repository.strip(), token.strip()
    datasets_repository = os.environ.get("HF_DATASETS_REPO")
    if datasets_repository and repository == datasets_repository.strip():
        raise ValueError("HF_OUTPUTS_REPO must differ from HF_DATASETS_REPO")
    ParquetTrajectoryUploader(HfApi(token=token), repository, args.directory).upload()
    print(f"Uploaded outputs to https://huggingface.co/datasets/{repository}")


if __name__ == "__main__":
    main()
