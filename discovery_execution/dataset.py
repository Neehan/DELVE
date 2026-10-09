"""Download and iterate a validated competition/year problem partition."""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

from huggingface_hub import HfApi, hf_hub_download

from discovery_execution.harness.models import DatasetConfig, Problem


class Dataset:
    """Load one configured partition and expose its problems in source order."""

    def __init__(self, config: DatasetConfig) -> None:
        """Download a missing partition and validate every row before exposing it."""
        self._config = config
        self._path = config.datasets_directory / config.competition / f"{config.year}.jsonl"
        if not self._path.exists():
            self._download()
        data = self._path.read_bytes()
        self._problems = self._parse(data)

    def __iter__(self) -> Iterator[Problem]:
        """Iterate the complete partition in its original order."""
        return iter(self._problems)

    def __len__(self) -> int:
        """Return the number of validated problems."""
        return len(self._problems)

    def _parse(self, data: bytes) -> list[Problem]:
        """Reject malformed rows, duplicate identities, and partition mismatches."""
        problems: dict[str, Problem] = {}
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            problem = Problem.model_validate_json(line)
            if problem.competition != self._config.competition or problem.year != self._config.year:
                raise ValueError(f"problem partition mismatch: {problem.problem_id}")
            if problem.problem_id in problems:
                raise ValueError(f"duplicate problem ID: {problem.problem_id}")
            problems[problem.problem_id] = problem
        if not problems:
            raise ValueError(f"empty dataset partition: {self._path}")
        return list(problems.values())

    def _download(self) -> None:
        """Validate a pinned HF download before installing it without replacing local data."""
        url = self._config.dataset_repo_url
        if url is None or not url.strip():
            raise FileNotFoundError(
                f"Dataset partition not found: {self._path}. "
                "Provide this file or set HF_DATASET_REPO_URL to download it."
            )
        repository = repository_id(url)
        revision = HfApi().dataset_info(repository).sha
        if revision is None:
            raise ValueError("Hugging Face did not return a dataset commit")
        filename = f"{self._config.competition}/{self._config.year}.jsonl"
        source = Path(hf_hub_download(repo_id=repository, filename=filename,
                                     repo_type="dataset", revision=revision))
        data = source.read_bytes()
        self._parse(data)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._install_file(self._path, data)
        metadata = {"repository": repository, "revision": revision, "filename": filename}
        self._install_file(self._path.with_suffix(".source.json"), json.dumps(metadata).encode("utf-8"))

    def _install_file(self, path: Path, content: bytes) -> None:
        """Publish flushed contents atomically, refusing to replace an existing local file."""
        with NamedTemporaryFile(mode="wb", dir=path.parent) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            # A hard link publishes the complete file and fails if another writer created the destination.
            os.link(handle.name, path)


def repository_id(url: str) -> str:
    """Require a dataset repository URL without file paths or embedded credentials."""
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.query or parsed.fragment:
        raise ValueError("HF_DATASET_REPO_URL must be https://huggingface.co/datasets/owner/repo")
    if len(parts) != 3 or parts[0] != "datasets" or not all(parts):
        raise ValueError("HF_DATASET_REPO_URL must identify a dataset repository")
    return "/".join(parts[1:])
