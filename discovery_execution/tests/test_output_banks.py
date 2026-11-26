"""Exercise publication round trips, checkpoint reconstruction, and seed-bank isolation."""

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.harness.models import RunState
from discovery_execution.audit.constants import GRADING_RULE
from discovery_execution.audit.compiler import AuditCompiler
from discovery_execution.outputs.parquet_reader import ParquetTrajectoryReader
from discovery_execution.outputs.constants import SCHEMA
from discovery_execution.outputs.native_reader import NativeTrajectoryReader
from discovery_execution.outputs.parquet_exporter import ParquetTrajectoryExporter
from discovery_execution.processing.storage import checkpoint_snapshot
from discovery_execution.scripts import export_outputs


def write_trajectory(root: Path, source_seed: int | None) -> Path:
    """Write a regressing curve with exact character-boundary checkpoints and request provenance."""
    metadata: dict[str, Any] = dict(competition="aime", year=2024, model="qwen3-4b-instruct-2507", problem_id="p1", seed=8,
                    arm="unaided-5x", experiment="unaided", compute_multiplier_k=5,
                    inject_sketch_at=0, solution_index=0, backend=dict(temperature=0.8, top_p=0.95, top_k=20))
    if source_seed is not None:
        metadata.update(arm="continue-at-1x", experiment="continue", source_arm="unaided-5x", source_seed=source_seed, continue_at=1)
    path = root / "aime/2024/qwen3-4b-instruct-2507" / metadata['arm'] / 'p1'
    path = path / f'seed_{source_seed}' / 'branch_8' if source_seed is not None else path / 'seed_8'
    path.mkdir(parents=True)
    start = 2 if source_seed is not None else 1
    state = RunState.model_validate(dict(messages=[dict(role='user', content='P'), dict(role='assistant', content='ABCDE')],
        output_tokens=5 * OUTPUT_TOKENS_PER_BLOCK, phase='revise', injected=False, response_finished=False,
        requests=[dict(seed=123, usage=dict(completion_tokens=17, prompt_tokens=4), finish_reason='length')],
        checkpoints=[dict(output_tokens=k * OUTPUT_TOKENS_PER_BLOCK, message_count=2, last_message_chars=k) for k in range(start, 6)]))
    (path / 'meta.json').write_text(json.dumps(metadata))
    (path / 'conversation.json.zst').write_bytes(zstandard.compress(state.model_dump_json().encode()))
    audit = dict(grading_rule=GRADING_RULE, complete=True, output_tokens=state.output_tokens,
                 checkpoints={f'{k}x': dict(correctness=dict(score=int(k in (1, 3)))) for k in range(start, 6)})
    (path / 'audit.json').write_text(json.dumps(audit))
    for boundary in state.checkpoints:
        snapshot = checkpoint_snapshot(state, boundary)
        (path / f'checkpoint_{boundary.output_tokens // OUTPUT_TOKENS_PER_BLOCK}x.json.zst').write_bytes(zstandard.compress(json.dumps(snapshot).encode()))
    return path


def test_round_trip_preserves_raw_curves_and_snapshots(tmp_path: Path) -> None:
    """Native grades can regress; packing preserves them and reconstructs every exact checkpoint."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    native = write_trajectory(source, None)
    original = NativeTrajectoryReader(source).read_trajectory(native)
    files = ParquetTrajectoryExporter(source, destination).export_trajectories(None)
    restored, = ParquetTrajectoryReader(destination).iter_trajectories('qwen3-4b-instruct-2507', None, None)
    assert original == restored
    assert json.loads(restored.audit)['checkpoints']['2x']['correctness']['score'] == 0
    assert len(files) == 1 and files[0].name == 'seed_8.parquet'
    assert json.loads((native / 'audit.json').read_text()) == json.loads(restored.audit)
    for boundary in restored.state.checkpoints:
        expected = json.loads(zstandard.decompress((native / f'checkpoint_{boundary.output_tokens // OUTPUT_TOKENS_PER_BLOCK}x.json.zst').read_bytes()))
        assert checkpoint_snapshot(restored.state, boundary) == expected


def test_parent_seeds_never_collide(tmp_path: Path) -> None:
    """Equal child seeds from distinct parents have separate banks and explicit branch identities."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    for parent in (0, 1):
        write_trajectory(source, parent)
    paths = ParquetTrajectoryExporter(source, destination).export_trajectories(None)
    rows = [pq.ParquetFile(path).read().to_pylist()[0] for path in paths]
    assert {row['source_seed'] for row in rows} == {0, 1}
    assert {row['branch_seed'] for row in rows} == {8}
    assert len(list(ParquetTrajectoryReader(destination).iter_trajectories('qwen3-4b-instruct-2507', None, None))) == 2


@pytest.mark.parametrize('source_seed', [None, 0, 1])
def test_projected_audits_match_native_compiler(tmp_path: Path, source_seed: int | None) -> None:
    """DE receives identical checkpoint records, including branch identities and regressing grades."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    native = write_trajectory(source, source_seed)
    arm = native.parents[2] if source_seed is not None else native.parents[1]
    compiled = AuditCompiler(arm).compile()
    expected = [json.loads(line) for line in compiled.read_text().splitlines()]
    exported, = ParquetTrajectoryExporter(source, destination).export_trajectories(None)
    actual = [record.model_dump(exclude_none=True) for record in ParquetTrajectoryReader(destination).iter_audits(exported.parent)]
    assert actual == expected


@pytest.mark.parametrize('grading_rule', [None, 'legacy-last-box'])
def test_projected_audits_reject_stale_grading(tmp_path: Path, grading_rule: str | None) -> None:
    """Reading only grades must preserve the native compiler's rejection of outdated answer-selection rules."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    native = write_trajectory(source, None)
    audit_path = native / 'audit.json'
    audit = json.loads(audit_path.read_text())
    audit['grading_rule'] = grading_rule
    audit_path.write_text(json.dumps(audit))
    exported, = ParquetTrajectoryExporter(source, destination).export_trajectories(None)
    with pytest.raises(ValueError, match='regrade with'):
        list(ParquetTrajectoryReader(destination).iter_audits(exported.parent))


def test_mismatched_snapshot_never_publishes(tmp_path: Path) -> None:
    """A changed checkpoint is rejected before any bank is marked complete."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    native = write_trajectory(source, None)
    (native / 'checkpoint_2x.json.zst').write_bytes(zstandard.compress(b'{}'))
    with pytest.raises(ValueError, match='snapshot disagrees'):
        ParquetTrajectoryExporter(source, destination).export_trajectories(None)
    assert not list(destination.rglob('*.parquet'))


@pytest.mark.parametrize('damage', ['version', 'duplicate', 'seed'])
def test_reject_corrupt_or_duplicate_banks(tmp_path: Path, damage: str) -> None:
    """Version, identity and duplicate-row faults cannot enter downstream analysis."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    write_trajectory(source, None)
    path, = ParquetTrajectoryExporter(source, destination).export_trajectories(None)
    table = pq.ParquetFile(path).read()
    if damage == 'version':
        table = table.replace_schema_metadata({b'delve.outputs.version': b'999'})
    elif damage == 'duplicate':
        table = pa.concat_tables([table, table])
    else:
        row, = table.to_pylist()
        row['seed'] = 99
        table = pa.Table.from_pylist([row], schema=SCHEMA)
    pq.write_table(table, path)
    with pytest.raises(ValueError):
        list(ParquetTrajectoryReader(destination).iter_trajectories('qwen3-4b-instruct-2507', None, None))


@pytest.mark.parametrize('addition', ['seed', 'model', 'problem'])
def test_incremental_export_preserves_existing_rows(tmp_path: Path, addition: str) -> None:
    """Reruns reuse identical bytes and can add a seed, model or problem without losing prior rows."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    original = write_trajectory(source, None)
    exporter = ParquetTrajectoryExporter(source, destination)
    first, = exporter.export_trajectories(None)
    stat = first.stat()
    first_bytes = first.read_bytes()
    assert exporter.export_trajectories(None) == [first]
    assert first.stat().st_mtime_ns == stat.st_mtime_ns
    metadata = json.loads((original / 'meta.json').read_text())
    if addition == 'seed':
        metadata['seed'] = 9
    elif addition == 'model':
        metadata['model'] = 'another-model'
    else:
        metadata['problem_id'] = 'another-problem'
    new_path = source / metadata['competition'] / str(metadata['year']) / metadata['model'] / metadata['arm'] / metadata['problem_id'] / f"seed_{metadata['seed']}"
    shutil.copytree(original, new_path)
    (new_path / 'meta.json').write_text(json.dumps(metadata))
    exported = exporter.export_trajectories(None)
    assert sum(pq.read_metadata(path).num_rows for path in exported) == 2
    if addition != 'problem':
        assert first.read_bytes() == first_bytes and first.stat().st_mtime_ns == stat.st_mtime_ns


@pytest.mark.parametrize('change', ['subset', 'grade'])
def test_export_rejects_existing_data_loss(tmp_path: Path, change: str) -> None:
    """A partial source or changed grade cannot replace an existing exported trajectory."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    original = write_trajectory(source, None)
    exporter = ParquetTrajectoryExporter(source, destination)
    path, = exporter.export_trajectories(None)
    if change == 'subset':
        rows = pq.read_table(path).to_pylist()
        added = json.loads(json.dumps(rows[0]))
        metadata = json.loads(added['metadata'])
        metadata['problem_id'] = 'remote-only'
        added['metadata'] = json.dumps(metadata)
        added['problem_id'] = 'remote-only'
        added['trajectory_id'] = added['trajectory_id'].replace('/p1/', '/remote-only/')
        pq.write_table(pa.Table.from_pylist([*rows, added], schema=SCHEMA), path)
    else:
        audit_path = original / 'audit.json'
        audit = json.loads(audit_path.read_text())
        audit['checkpoints']['1x']['correctness']['score'] = 0
        audit_path.write_text(json.dumps(audit))
    saved = path.read_bytes()
    with pytest.raises(ValueError, match='remove or change existing trajectory'):
        exporter.export_trajectories(None)
    assert path.read_bytes() == saved


def test_reader_filters_before_decoding(tmp_path: Path) -> None:
    """Unselected seeds and branch files need not be readable for root-only processing."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    write_trajectory(source, None)
    path, = ParquetTrajectoryExporter(source, destination).export_trajectories(None)
    path.with_name('seed_9.parquet').write_text('unfinished seed')
    branch = path.parent.with_name('continue-at-1x') / 'source_seed_0' / 'seed_8.parquet'
    branch.parent.mkdir(parents=True)
    branch.write_text('unselected arm')
    rows = list(ParquetTrajectoryReader(destination).iter_trajectories('qwen3-4b-instruct-2507', (8,), ('unaided-5x',)))
    assert len(rows) == 1


def test_export_cli_selects_finished_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exporting chosen seeds succeeds even while another seed is still missing its artifacts."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    original = write_trajectory(source, None)
    metadata = json.loads((original / 'meta.json').read_text())
    metadata['seed'] = 9
    unfinished = original.with_name('seed_9')
    unfinished.mkdir()
    (unfinished / 'meta.json').write_text(json.dumps(metadata))
    monkeypatch.setattr('sys.argv', ['export_outputs', '--source', str(source), '--destination', str(destination), '--seeds', '8'])
    export_outputs.main()
    paths = list(destination.rglob('*.parquet'))
    assert len(paths) == 1 and paths[0].name == 'seed_8.parquet'
