"""Upload saved trajectories and audits to a separate Hugging Face dataset repository."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils._validators import validate_repo_id

from discovery_execution.harness.constants import ENV_FILE, OUTPUT_DIRECTORY, OUTPUT_UPLOAD_PATTERNS


def upload_outputs(api: HfApi, repository: str, directory: Path) -> None:
    """Publish recognized output files, preserving paths and remote-only files."""
    validate_repo_id(repository)
    if not directory.is_dir():
        raise FileNotFoundError(f"Output directory not found: {directory}")
    if not any(path.is_file() for pattern in OUTPUT_UPLOAD_PATTERNS for path in directory.glob(pattern)):
        raise ValueError(f"No saved trajectories or audits found in {directory}")
    api.create_repo(repo_id=repository, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repository, repo_type="dataset", folder_path=directory,
        allow_patterns=OUTPUT_UPLOAD_PATTERNS, commit_message="Upload trajectories and audits",
    )


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
