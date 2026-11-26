"""Check upload selection, credentials, and failures without publishing data."""

import pyarrow.parquet as pq
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from discovery_execution.scripts.upload_outputs import main, upload_outputs


def test_upload_preserves_paths_and_excludes_non_artifacts(hf_upload_api: Mock, output_artifacts: Path) -> None:
    """Upload only recognized artifacts at their existing relative paths, without remote deletion."""
    captured = []
    def inspect_upload(**arguments: Any) -> None:
        """Inspect real staged Parquet while the upload call still owns its temporary directory."""
        root = Path(arguments["folder_path"])
        paths = list(root.rglob("*"))
        captured.extend(path.relative_to(root).as_posix() for path in paths if path.is_file())
        bank, = root.rglob("*.parquet")
        row, = pq.ParquetFile(bank).read().to_pylist()
        assert row["problem_id"] == "example" and row["grades"] == [{"block": 1, "score": 1}]
        assert "delete_patterns" not in arguments
    hf_upload_api.upload_folder.side_effect = inspect_upload
    upload_outputs(hf_upload_api, "owner/outputs", output_artifacts)
    hf_upload_api.create_repo.assert_called_once_with(repo_id="owner/outputs", repo_type="dataset", private=True, exist_ok=True)
    hf_upload_api.upload_folder.assert_called_once()
    assert captured == ["aime/2024/model/unaided-1x/seed_0.parquet"]



@pytest.mark.parametrize("existing", [False, True])
def test_no_outputs_fails_before_network(hf_upload_api: Mock, tmp_path: Path, existing: bool) -> None:
    """Reject missing or artifact-free directories before creating a remote repository."""
    root = tmp_path / "output"
    if existing:
        root.mkdir()
        (root / ".lock").touch()
    with pytest.raises((FileNotFoundError, ValueError)):
        upload_outputs(hf_upload_api, "owner/outputs", root)
    assert not hf_upload_api.mock_calls


def test_upload_failure_propagates(hf_upload_api: Mock, output_artifacts: Path) -> None:
    """Surface a failed upload immediately instead of reporting success."""
    hf_upload_api.upload_folder.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        upload_outputs(hf_upload_api, "owner/outputs", output_artifacts)


def test_main_reads_upload_environment(upload_environment: None, hf_upload_api: Mock,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the explicit token to HF and select the outputs repository."""
    constructor = Mock(return_value=hf_upload_api)
    monkeypatch.setattr("discovery_execution.scripts.upload_outputs.HfApi", constructor)
    main()
    constructor.assert_called_once_with(token="test-token")
    assert hf_upload_api.upload_folder.call_args.kwargs["repo_id"] == "owner/outputs"


@pytest.mark.parametrize(("name", "value"), [
    ("HF_TOKEN", ""), ("HF_TOKEN", "   "), ("HF_OUTPUTS_REPO", ""),
    ("HF_OUTPUTS_REPO", "owner/problems"), ("HF_OUTPUTS_REPO", "https://huggingface.co/datasets/owner/outputs"),
])
def test_invalid_upload_environment(upload_environment: None, hf_upload_api: Mock,
                                    monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    """Reject missing credentials, invalid IDs, and accidentally targeting the problems repository."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        main()
    assert not hf_upload_api.mock_calls
