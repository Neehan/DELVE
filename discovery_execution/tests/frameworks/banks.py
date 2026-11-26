"""Small Parquet banks retaining the original framework observation fixtures."""

import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from discovery_execution.audit.models import AuditRecord
from discovery_execution.harness.models import Checkpoint, RunState
from discovery_execution.harness.constants import OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.outputs.bank import OutputBank
from discovery_execution.outputs.constants import SCHEMA
from discovery_execution.outputs.models import Trajectory


def read_records(directory: Path) -> list[dict]:
    """Expose checkpoint records so tests can alter observations independently of Parquet encoding."""
    return [row.model_dump() for row in OutputBank(directory.parents[3]).iter_audits(directory)]


def write_records(directory: Path, records: list[dict]) -> None:
    """Store complete per-trajectory curves in one fixture bank per seed."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row['problem_id'], row['seed']].append(row)
    banks = defaultdict(list)
    for (problem, seed), curve in grouped.items():
        first = curve[0]
        competition, year = first['dataset'].split('/')
        metadata = {key: first[key] for key in ('experiment', 'compute_multiplier_k', 'inject_sketch_at', 'solution_index', 'seed')}
        metadata.update(competition=competition, year=int(year), model=first['solver_model'], problem_id=problem, arm=directory.name)
        boundaries = [Checkpoint(output_tokens=row['checkpoint_multiplier_k'] * OUTPUT_TOKENS_PER_BLOCK,
                                 message_count=2, last_message_chars=1) for row in curve]
        state = RunState(messages=[dict(role='user', content='problem'), dict(role='assistant', content='x')],
                         output_tokens=first['compute_multiplier_k'] * OUTPUT_TOKENS_PER_BLOCK,
                         phase='solve', injected=first['experiment']=='oracle-execution', response_finished=True,
                         requests=[], checkpoints=boundaries)
        audit = dict(complete=True, output_tokens=state.output_tokens,
                     checkpoints={f"{row['checkpoint_multiplier_k']}x": dict(correctness=row['correctness']) for row in curve})
        identity = (directory / problem / f'seed_{seed}').relative_to(directory.parents[3]).as_posix()
        valid = {**metadata, 'competition': directory.parents[2].name,
                 'year': int(directory.parents[1].name), 'model': directory.parent.name}
        row = Trajectory(identity, json.dumps(valid), state, json.dumps(audit)).to_row()
        row['metadata'] = json.dumps(metadata)
        banks[seed].append(row)
    for path in directory.glob('seed_*.parquet'):
        path.unlink()
    for seed, rows in banks.items():
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), directory / f'seed_{seed}.parquet')
