"""Upload saved trajectories and audits to a separate Hugging Face dataset repository."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils._validators import validate_repo_id

from discovery_execution.harness.constants import ENV_FILE, OUTPUT_DIRECTORY
from discovery_execution.outputs.packer import OutputPacker


def upload_outputs(api: HfApi, repository: str, directory: Path) -> None:
    """Pack complete native outputs and publish only verified per-seed Parquet banks."""
    validate_repo_id(repository)
    if not directory.is_dir():
        raise FileNotFoundError(f"Output directory not found: {directory}")
    with TemporaryDirectory() as temporary:
        published = Path(temporary)
        OutputPacker(directory, published).pack(None)
        api.create_repo(repo_id=repository, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repository, repo_type="dataset", folder_path=published,
                          allow_patterns=["**/seed_*.parquet"], commit_message="Upload per-seed trajectory Parquet banks")


def main() -> None:
    """Read HF credentials and upload output/ after generation and auditing have stopped."""
    load_dotenv(ENV_FILE, override=False)
    repository = os.environ.get("HF_OUTPUTS_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repository or not repository.strip() or not token or not token.strip():
        raise ValueError("Set HF_OUTPUTS_REPO and HF_TOKEN before uploading outputs")
    repository, token = repository.strip(), token.strip()
    datasets_repository = os.environ.get("HF_DATASETS_REPO")
    if datasets_repository and repository == datasets_repository.strip():
        raise ValueError("HF_OUTPUTS_REPO must differ from HF_DATASETS_REPO")
    upload_outputs(HfApi(token=token), repository, OUTPUT_DIRECTORY)
    print(f"Uploaded outputs to https://huggingface.co/datasets/{repository}")


if __name__ == "__main__":
    main()
