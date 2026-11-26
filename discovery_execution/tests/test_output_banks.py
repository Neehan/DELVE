"""Exercise publication round trips, checkpoint reconstruction, and seed-bank isolation."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.harness.models import RunState
from discovery_execution.outputs.bank import OutputBank
from discovery_execution.outputs.constants import SCHEMA
from discovery_execution.outputs.native import NativeReader
from discovery_execution.outputs.packer import OutputPacker
from discovery_execution.processing.storage import checkpoint_snapshot


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
    audit = dict(complete=True, output_tokens=state.output_tokens,
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
    original = NativeReader(source).read(native)
    files = OutputPacker(source, destination).pack(None)
    restored, = OutputBank(destination).iter_trajectories('qwen3-4b-instruct-2507', None)
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
    paths = OutputPacker(source, destination).pack(None)
    rows = [pq.ParquetFile(path).read().to_pylist()[0] for path in paths]
    assert {row['source_seed'] for row in rows} == {0, 1}
    assert {row['branch_seed'] for row in rows} == {8}
    assert len(list(OutputBank(destination).iter_trajectories('qwen3-4b-instruct-2507', None))) == 2


def test_mismatched_snapshot_never_publishes(tmp_path: Path) -> None:
    """A changed checkpoint is rejected before any bank is marked complete."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    native = write_trajectory(source, None)
    (native / 'checkpoint_2x.json.zst').write_bytes(zstandard.compress(b'{}'))
    with pytest.raises(ValueError, match='snapshot disagrees'):
        OutputPacker(source, destination).pack(None)
    assert not list(destination.rglob('*.parquet'))


@pytest.mark.parametrize('damage', ['version', 'duplicate', 'seed'])
def test_reject_corrupt_or_duplicate_banks(tmp_path: Path, damage: str) -> None:
    """Version, identity and duplicate-row faults cannot enter downstream analysis."""
    source, destination = tmp_path / 'native', tmp_path / 'published'
    write_trajectory(source, None)
    path, = OutputPacker(source, destination).pack(None)
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
        list(OutputBank(destination).iter_trajectories('qwen3-4b-instruct-2507', None))
