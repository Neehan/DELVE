"""Verify incremental HF publication preserves existing rows and pins the checked revision."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import CommitOperationAdd
from huggingface_hub.hf_api import BlobLfsInfo, RepoSibling

from discovery_execution.outputs.constants import SCHEMA
from discovery_execution.outputs.models import Trajectory
from discovery_execution.outputs.parquet_uploader import ParquetTrajectoryUploader
from discovery_execution.scripts.upload_outputs import main


def add_problem_row(path: Path, problem_id: str) -> None:
    """Extend a fixture Parquet file with a second problem while retaining its first row."""
    rows = pq.read_table(path).to_pylist()
    original = Trajectory.from_row(rows[0])
    metadata = json.loads(original.metadata)
    metadata['problem_id'] = problem_id
    identity = Path(original.trajectory_id)
    identity = identity.parent.with_name(problem_id) / identity.name
    added = Trajectory(identity.as_posix(), json.dumps(metadata), original.state, original.audit)
    pq.write_table(pa.Table.from_pylist([*rows, added.to_row()], schema=SCHEMA), path)


def mock_remote_file(api: Mock, relative: str, remote_path: Path) -> None:
    """Serve a pinned remote fixture through the injected API without network access."""
    api.dataset_info.return_value.siblings = [RepoSibling(relative)]
    def download(repository: str, **arguments: Any) -> str:
        """Copy the remote fixture into the requested local download directory."""
        assert repository == 'owner/outputs' and arguments['revision'] == 'checked-revision'
        destination = arguments['local_dir'] / arguments['filename']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote_path, destination)
        return str(destination)
    api.hf_hub_download.side_effect = download


def test_upload_uses_existing_exports(hf_upload_api: Mock, published_outputs: Path) -> None:
    """Publish only exported Parquet, keeping bytes and paths intact and adding no deletion operations."""
    (published_outputs / '.env').write_text('not an upload')
    path, = published_outputs.rglob('*.parquet')
    original = path.read_bytes()
    ParquetTrajectoryUploader(hf_upload_api, 'owner/outputs', published_outputs).upload()
    hf_upload_api.create_repo.assert_called_once_with(repo_id='owner/outputs', repo_type='dataset', private=True, exist_ok=True)
    arguments = hf_upload_api.create_commit.call_args.kwargs
    operation, = arguments['operations']
    assert isinstance(operation, CommitOperationAdd)
    assert operation.path_in_repo == 'aime/2024/model/unaided-1x/seed_0.parquet'
    assert operation.path_or_fileobj in (path, str(path))
    assert arguments['parent_commit'] == 'checked-revision'
    assert path.read_bytes() == original
    hf_upload_api.hf_hub_download.assert_not_called()


@pytest.mark.parametrize('existing', [False, True])
def test_no_outputs_fails_before_network(hf_upload_api: Mock, tmp_path: Path, existing: bool) -> None:
    """Missing or artifact-free directories fail before any HF operation."""
    root = tmp_path / 'output'
    if existing:
        root.mkdir()
        (root / '.lock').touch()
    with pytest.raises((FileNotFoundError, ValueError)):
        ParquetTrajectoryUploader(hf_upload_api, 'owner/outputs', root).upload()
    assert not hf_upload_api.mock_calls


def test_identical_remote_file_skips_upload(hf_upload_api: Mock, published_outputs: Path) -> None:
    """Matching HF LFS hashes avoid downloading or recommitting unchanged files."""
    path, = published_outputs.rglob('*.parquet')
    data = path.read_bytes()
    hf_upload_api.dataset_info.return_value.siblings = [RepoSibling(path.relative_to(published_outputs).as_posix(),
        lfs=BlobLfsInfo(size=len(data), sha256=hashlib.sha256(data).hexdigest(), pointer_size=0))]
    ParquetTrajectoryUploader(hf_upload_api, 'owner/outputs', published_outputs).upload()
    hf_upload_api.hf_hub_download.assert_not_called()
    hf_upload_api.create_commit.assert_not_called()


@pytest.mark.parametrize('change', ['add', 'remove', 'modify', 'reencode'])
def test_existing_trajectories_are_preserved(hf_upload_api: Mock, published_outputs: Path, tmp_path: Path, change: str) -> None:
    """Allow new rows and ignore encoding changes; reject missing rows or changed existing grades."""
    local, = published_outputs.rglob('*.parquet')
    remote = tmp_path / 'remote.parquet'
    shutil.copyfile(local, remote)
    if change == 'add':
        add_problem_row(local, 'new-problem')
    elif change == 'remove':
        add_problem_row(remote, 'remote-only-problem')
    elif change == 'modify':
        row, = pq.read_table(local).to_pylist()
        audit = json.loads(row['audit'])
        audit['checkpoints']['1x']['correctness']['score'] = 0
        row['audit'] = json.dumps(audit)
        row['grades'][0]['score'] = 0
        pq.write_table(pa.Table.from_pylist([row], schema=SCHEMA), local)
    else:
        pq.write_table(pq.read_table(local), local, compression='NONE')
    mock_remote_file(hf_upload_api, local.relative_to(published_outputs).as_posix(), remote)
    uploader = ParquetTrajectoryUploader(hf_upload_api, 'owner/outputs', published_outputs)
    if change in ('remove', 'modify'):
        with pytest.raises(ValueError, match='remove or change existing trajectory'):
            uploader.upload()
        hf_upload_api.create_commit.assert_not_called()
    else:
        uploader.upload()
        assert hf_upload_api.create_commit.call_count == int(change == 'add')


def test_concurrent_update_fails_without_retry(hf_upload_api: Mock, published_outputs: Path) -> None:
    """A failed pinned commit is surfaced instead of retrying against unchecked remote data."""
    hf_upload_api.create_commit.side_effect = RuntimeError('parent changed')
    with pytest.raises(RuntimeError, match='parent changed'):
        ParquetTrajectoryUploader(hf_upload_api, 'owner/outputs', published_outputs).upload()
    assert hf_upload_api.create_commit.call_count == 1


def test_main_reads_upload_environment(upload_environment: None, hf_upload_api: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read credentials and upload existing exports to the outputs repository."""
    constructor = Mock(return_value=hf_upload_api)
    monkeypatch.setattr('discovery_execution.scripts.upload_outputs.HfApi', constructor)
    main()
    constructor.assert_called_once_with(token='test-token')
    assert hf_upload_api.create_commit.call_args.kwargs['repo_id'] == 'owner/outputs'


@pytest.mark.parametrize(('name', 'value'), [
    ('HF_TOKEN', ''), ('HF_TOKEN', '   '), ('HF_OUTPUTS_REPO', ''),
    ('HF_OUTPUTS_REPO', 'owner/problems'), ('HF_OUTPUTS_REPO', 'https://huggingface.co/datasets/owner/outputs'),
])
def test_invalid_upload_environment(upload_environment: None, hf_upload_api: Mock,
    monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    """Reject invalid credentials or accidentally targeting the problems repository."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        main()
    assert not hf_upload_api.mock_calls
