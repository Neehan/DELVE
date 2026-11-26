"""Upload exported Parquet files while preserving already published trajectories."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.hf_api import RepoSibling
from huggingface_hub.utils._validators import validate_repo_id

from discovery_execution.outputs.constants import SEED_PATTERN
from discovery_execution.outputs.models import require_preserved_trajectories
from discovery_execution.outputs.parquet_reader import ParquetTrajectoryReader


class ParquetTrajectoryUploader:
    """Use upload to publish existing exports through the injected HF API.

    Identical files are skipped. Updates may add trajectories but cannot change or remove existing ones.
    """

    def __init__(self, api: HfApi, repository: str, directory: Path) -> None:
        """Publish directory's exported files to repository through the injected HF api."""
        self.api = api
        self.repository = repository
        self.directory = directory
        self.reader = ParquetTrajectoryReader(directory)

    def upload(self) -> None:
        """Validate exports and commit additions, failing if the checked HF revision changes."""
        validate_repo_id(self.repository)
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        paths = sorted(self.directory.rglob(SEED_PATTERN))
        if not paths:
            raise ValueError(f"no exported trajectory Parquet files: {self.directory}")
        self.api.create_repo(repo_id=self.repository, repo_type="dataset", private=True, exist_ok=True)
        info = self.api.dataset_info(self.repository, files_metadata=True)
        if info.sha is None or info.siblings is None:
            raise ValueError("HF repository revision or file metadata is missing")
        remote_files = {file.rfilename: file for file in info.siblings}
        operations = []
        with TemporaryDirectory() as temporary:
            for path in paths:
                relative = path.relative_to(self.directory).as_posix()
                remote_file = remote_files.get(relative)
                if not self._needs_upload(path, remote_file, info.sha, Path(temporary)):
                    continue
                operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=path))
        if operations:
            # Guard the checked revision so a concurrent publisher cannot be silently overwritten.
            self.api.create_commit(repo_id=self.repository, repo_type="dataset", operations=operations,
                parent_commit=info.sha, commit_message="Upload trajectory Parquet files")

    def _needs_upload(self, path: Path, remote_file: RepoSibling | None, revision: str, download_directory: Path) -> bool:
        """Validate path and compare it with remote_file at revision, using download_directory for changed files.

        Return whether an upload adds trajectories; reject removed or changed existing rows.
        """
        fingerprints = self.reader.trajectory_fingerprints(path)
        if remote_file is None:
            return True
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if remote_file.lfs is not None and remote_file.lfs.sha256 == digest:
            return False
        self.api.hf_hub_download(self.repository, filename=remote_file.rfilename, repo_type="dataset",
            revision=revision, local_dir=download_directory)
        previous = ParquetTrajectoryReader(download_directory).trajectory_fingerprints(download_directory / remote_file.rfilename)
        require_preserved_trajectories(previous, fingerprints)
        return previous != fingerprints
