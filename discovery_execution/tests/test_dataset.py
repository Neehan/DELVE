"""Verify missing-partition setup and selection without contacting Hugging Face."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from discovery_execution.dataset import Dataset
from discovery_execution.harness.models import DatasetConfig, Problem


@pytest.fixture
def download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: Problem) -> Mock:
    """Inject a fixed HF revision and a local download result."""
    source = tmp_path / "download.jsonl"
    source.write_text(problem.model_dump_json() + "\n", encoding="utf-8")
    api = Mock()
    api.dataset_info.return_value.sha = "pinned-commit"
    downloader = Mock(return_value=str(source))
    monkeypatch.setattr("discovery_execution.dataset.HfApi", Mock(return_value=api))
    monkeypatch.setattr("discovery_execution.dataset.hf_hub_download", downloader)
    return downloader


def test_download_once_then_iterate_locally(dataset_config: DatasetConfig, download: Mock) -> None:
    """Download a pinned partition, then iterate it repeatedly without downloading again."""
    dataset = Dataset(dataset_config)
    assert [problem.problem_id for problem in dataset] == ["example"]
    assert list(dataset) == list(dataset)
    reloaded = Dataset(dataset_config.model_copy(update={"hf_repo": None}))
    assert list(reloaded) == list(dataset)
    download.assert_called_once_with(repo_id="owner/repo", filename="aime/2024.jsonl",
                                     repo_type="dataset", revision="pinned-commit")


def test_invalid_download_is_not_installed(dataset_config: DatasetConfig, download: Mock) -> None:
    """Reject downloaded partitions before publishing invalid data into the dataset root."""
    Path(download.return_value).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Dataset(dataset_config)
    assert not (dataset_config.datasets_directory / "aime" / "2024.jsonl").exists()


def test_install_failure_leaves_no_partial_partition(dataset_config: DatasetConfig, download: Mock,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after writing temporary contents must leave the final path absent and allow retry."""
    path = dataset_config.datasets_directory / "aime" / "2024.jsonl"
    with monkeypatch.context() as patch:
        patch.setattr("discovery_execution.dataset.os.fsync", Mock(side_effect=OSError("interrupted write")))
        with pytest.raises(OSError, match="interrupted write"):
            Dataset(dataset_config)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []
    assert len(Dataset(dataset_config)) == 1
    assert path.read_bytes() == Path(download.return_value).read_bytes()


def test_install_publishes_complete_files(dataset_config: DatasetConfig, download: Mock,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the dataset and source record become visible only after their full contents are written."""
    link = os.link
    published: list[Path] = []

    def publish(source: str, destination: Path) -> None:
        """Inspect staged contents at the atomic publication boundary."""
        assert not destination.exists()
        content = Path(source).read_bytes()
        if destination.suffix == ".jsonl":
            assert content == Path(download.return_value).read_bytes()
        else:
            assert json.loads(content) == {"repository": "owner/repo", "revision": "pinned-commit",
                                          "filename": "aime/2024.jsonl"}
        link(source, destination)
        published.append(destination)

    monkeypatch.setattr("discovery_execution.dataset.os.link", publish)
    Dataset(dataset_config)
    assert {path.name for path in published} == {"2024.jsonl", "2024.source.json"}
    assert set(published[0].parent.iterdir()) == set(published)


def test_concurrent_install_preserves_existing_partition(dataset_config: DatasetConfig, download: Mock,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """An independently installed partition must never be replaced by a competing download."""
    link = os.link
    path = dataset_config.datasets_directory / "aime" / "2024.jsonl"
    existing = b"another writer's dataset\n"

    def publish(source: str, destination: Path) -> None:
        """Simulate another process installing the destination immediately before publication."""
        destination.write_bytes(existing)
        link(source, destination)

    monkeypatch.setattr("discovery_execution.dataset.os.link", publish)
    with pytest.raises(FileExistsError):
        Dataset(dataset_config)
    assert path.read_bytes() == existing
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("repository", [None, "", "   "])
def test_missing_dataset_requires_repository(dataset_config: DatasetConfig, download: Mock,
                                             monkeypatch: pytest.MonkeyPatch, repository: str | None) -> None:
    """Reject missing data with no repository before any Hugging Face call."""
    api = Mock()
    monkeypatch.setattr("discovery_execution.dataset.HfApi", api)
    with pytest.raises(FileNotFoundError, match="HF_REPO") as error:
        Dataset(dataset_config.model_copy(update={"hf_repo": repository}))
    assert str(dataset_config.datasets_directory / "aime" / "2024.jsonl") in str(error.value)
    api.assert_not_called()
    download.assert_not_called()


@pytest.mark.parametrize("repository", ["https://huggingface.co/datasets/owner/repo", "owner/repo/file", "owner/bad..name"])
def test_invalid_repository_fails_before_download(dataset_config: DatasetConfig, download: Mock,
                                                 monkeypatch: pytest.MonkeyPatch, repository: str) -> None:
    """Accept HF repository IDs rather than URLs or paths, validating before network access."""
    api = Mock()
    monkeypatch.setattr("discovery_execution.dataset.HfApi", api)
    with pytest.raises(ValueError):
        Dataset(dataset_config.model_copy(update={"hf_repo": repository}))
    api.assert_not_called()
    download.assert_not_called()


def test_invalid_local_data_does_not_download(dataset_config: DatasetConfig, download: Mock) -> None:
    """Fail on invalid local data instead of silently replacing it from the repository."""
    path = dataset_config.datasets_directory / "aime" / "2024.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Dataset(dataset_config)
    download.assert_not_called()
    assert path.read_text(encoding="utf-8") == "{}\n"


def test_partition_year_is_not_hardcoded(dataset_config: DatasetConfig, problem: Problem) -> None:
    """Validate and load a partition beyond the originally listed years."""
    year = 2030
    config = DatasetConfig(**{**dataset_config.model_dump(), "year": year})
    row = Problem(**{**problem.model_dump(), "year": year})
    path = config.datasets_directory / row.competition / f"{year}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    assert list(Dataset(config)) == [row]
