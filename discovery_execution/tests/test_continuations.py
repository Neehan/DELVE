"""Exercise exact-prefix branching, oracle interventions, restart, and offline audits."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import zstandard

from discovery_execution.audit.models import AuditRecord
from discovery_execution.audit.runner import AuditRunner
from discovery_execution.cli.audit import parse_arguments as parse_audit
from discovery_execution.cli.run import parse_arguments
from discovery_execution.harness.constants import CHECKPOINT_FILENAME, CONVERSATION_FILENAME, META_FILENAME, OUTPUT_TOKENS_PER_BLOCK
from discovery_execution.harness.models import Problem, RunConfig, RunState
from discovery_execution.harness.openai_client import OpenAIClient
from discovery_execution.harness.runner import Runner
from discovery_execution.harness.self_refine import SelfRefine
from discovery_execution.harness.storage import Storage, atomic_write, model_output_directory, write_json


@pytest.mark.parametrize("block", [1, 2, 3, 4])
@pytest.mark.parametrize("experiment", ["continue", "continue-oracle"])
async def test_branch_from_exact_prefix(block: int, experiment: str, branch_source: RunConfig,
                                       make_branch_config: Callable[[str, int], RunConfig], problem: Problem,
                                       client: OpenAIClient, records: list[dict[str, Any]]) -> None:
    """Fork fresh seeds from identical text, keep source files intact, and spend only the remaining budget."""
    config = make_branch_config(experiment, block)
    root = model_output_directory(config.output_directory, config.competition, config.year, config.model)
    source = root / str(config.source_arm) / problem.problem_id / f"seed_{config.source_seed}"
    before = {path: path.read_bytes() for path in source.glob("*.json*")}
    snapshot = json.loads(zstandard.decompress((source / CHECKPOINT_FILENAME.format(k=block)).read_bytes()))
    first_seeds = []
    for seed in config.seeds:
        begin = len(records)
        refinement = SelfRefine(config, problem, seed, client)
        await refinement.run()
        payload = records[begin]["payload"]
        first_seeds.append(payload["seed"])
        assert payload["messages"][:len(snapshot["messages"])] == snapshot["messages"]
        if experiment == "continue-oracle":
            assert problem.solutions[0].sketch in payload["messages"][-1]["content"]
            assert not payload["continue_final_message"]
        else:
            assert payload["messages"] == snapshot["messages"]
            assert payload["continue_final_message"]
        assert sum(record["count"] for record in records[begin:]) == (config.compute_multiplier_k - block) * OUTPUT_TOKENS_PER_BLOCK
        assert refinement.state.output_tokens == config.compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK
        assert [item.output_tokens for item in refinement.state.checkpoints] == [
            k * OUTPUT_TOKENS_PER_BLOCK for k in range(block + 1, config.compute_multiplier_k + 1)]
        assert not (refinement.storage.directory / CHECKPOINT_FILENAME.format(k=block)).exists()
    assert len(set(first_seeds)) == len(config.seeds)
    count = len(records)
    assert await Runner(config, client).run()
    assert len(records) == count
    assert before == {path: path.read_bytes() for path in source.glob("*.json*")}


@pytest.mark.parametrize("experiment", ["continue", "continue-oracle"])
async def test_branch_restart_after_initialization(experiment: str, make_branch_config: Callable[[str, int], RunConfig],
                                                    problem: Problem, client: OpenAIClient,
                                                    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]) -> None:
    """Persist the parent state before generating; restart needs no source files and injects only once."""
    config = make_branch_config(experiment, 2).model_copy(update={"seeds": [2]})
    original_save = Storage.save

    def interrupt(storage: Storage, state: RunState) -> None:
        """Stop just after the fork has been durably initialized."""
        original_save(storage, state)
        raise RuntimeError("interrupted fork")

    with monkeypatch.context() as patch:
        patch.setattr(Storage, "save", interrupt)
        assert not await Runner(config, client).run()
    root = model_output_directory(config.output_directory, config.competition, config.year, config.model)
    source = root / str(config.source_arm)
    source.rename(root / "source-unavailable")
    begin = len(records)
    assert await Runner(config, client).run()
    assert sum(record["count"] for record in records[begin:]) == 3 * OUTPUT_TOKENS_PER_BLOCK
    path, = (root / f"{experiment}-at-2x").rglob(CONVERSATION_FILENAME)
    state = RunState.model_validate_json(zstandard.decompress(path.read_bytes()))
    sketch_count = sum(message["content"].count(problem.solutions[0].sketch) for message in state.messages)
    assert sketch_count == int(experiment == "continue-oracle")


async def test_source_boundary_natural_end(make_branch_config: Callable[[str, int], RunConfig], problem: Problem,
                                           client: OpenAIClient, records: list[dict[str, Any]]) -> None:
    """Restore the boundary's phase and natural-ending flag rather than the parent's final phase."""
    config = make_branch_config("continue", 1)
    root = model_output_directory(config.output_directory, config.competition, config.year, config.model)
    path = root / str(config.source_arm) / problem.problem_id / f"seed_{config.source_seed}" / CONVERSATION_FILENAME
    state = RunState.model_validate_json(zstandard.decompress(path.read_bytes()))
    request = next(item for item in state.requests if item["output_tokens"] == OUTPUT_TOKENS_PER_BLOCK)
    request.update(finish_reason="stop", phase="critique")
    atomic_write(path, zstandard.compress(state.model_dump_json().encode("utf-8")))
    begin = len(records)
    assert await Runner(config, client).run()
    first = records[begin]["payload"]
    assert first["messages"][-1]["role"] == "user"
    assert not first["continue_final_message"]
    assert "using the critique above" in first["messages"][-1]["content"].lower()


async def test_oracle_uses_parent_solution(make_branch_config: Callable[[str, int], RunConfig],
                                          multiple_solution_problem: Problem, client: OpenAIClient) -> None:
    """Branch seeds cannot change the source seed's round-robin sketch selection."""
    config = make_branch_config("continue-oracle", 1)
    assert config.source_seed == 1
    for seed in config.seeds:
        refinement = SelfRefine(config, multiple_solution_problem, seed, client)
        await refinement.run()
        assert refinement.storage.read_metadata()["solution_index"] == 1
        assert refinement.sketch == multiple_solution_problem.solutions[1].sketch
    explicit = make_branch_config("continue-oracle", 2).model_copy(update={"solution_index": 2})
    refinement = SelfRefine(explicit, multiple_solution_problem, 2, client)
    await refinement.run()
    assert refinement.storage.read_metadata()["solution_index"] == 2


@pytest.mark.parametrize("damage", ["missing", "snapshot", "backend"])
async def test_invalid_source_fails_before_generation(damage: str, make_branch_config: Callable[[str, int], RunConfig],
                                                       problem: Problem, client: OpenAIClient,
                                                       records: list[dict[str, Any]]) -> None:
    """Reject unavailable forks, mismatched text, and changed sampling settings without issuing requests."""
    config = make_branch_config("continue", 2)
    root = model_output_directory(config.output_directory, config.competition, config.year, config.model)
    source = root / str(config.source_arm) / problem.problem_id / f"seed_{config.source_seed}"
    if damage == "missing":
        config = config.model_copy(update={"source_seed": 99})
    elif damage == "snapshot":
        path = source / CHECKPOINT_FILENAME.format(k=2)
        snapshot = json.loads(zstandard.decompress(path.read_bytes()))
        snapshot["messages"][-1]["content"] += "CORRUPTION"
        atomic_write(path, zstandard.compress(json.dumps(snapshot).encode("utf-8")))
    else:
        path = source / META_FILENAME
        metadata = json.loads(path.read_text())
        metadata["backend"]["temperature"] += 1
        write_json(path, metadata)
    begin = len(records)
    assert not await Runner(config, client).run()
    assert len(records) == begin


async def test_audit_branch_identity_and_absolute_checkpoints(make_branch_config: Callable[[str, int], RunConfig],
                                                              branch_source: RunConfig, client: OpenAIClient) -> None:
    """Audit future checkpoints only and distinguish branch seeds under multiple parents in one export."""
    config = make_branch_config("continue", 2)
    assert await Runner(config, client).run()
    oracle = make_branch_config("continue-oracle", 2)
    assert await Runner(oracle, client).run()
    root = model_output_directory(config.output_directory, config.competition, config.year, config.model)
    AuditRunner().run(root, None, [2], {"experiment": "continue", "source_seed": 1, "continue_at": 2})
    compiled = root / "continue-at-2x" / "audit.jsonl"
    rows = [AuditRecord.model_validate_json(line) for line in compiled.read_text().splitlines()]
    assert [row.checkpoint_multiplier_k for row in rows] == [3, 4, 5]
    assert all(row.source_seed == 1 and row.seed == 2 and row.source_arm == config.source_arm for row in rows)
    assert not list((root / "continue-oracle-at-2x").rglob("audit.json"))
    AuditRunner().run(root, None, [3], {"experiment": "continue"})
    assert len(compiled.read_text().splitlines()) == 6
    AuditRunner().run(root, None, None, {"experiment": "continue-oracle"})
    oracle_rows = [AuditRecord.model_validate_json(line) for line in (root / "continue-oracle-at-2x/audit.jsonl").read_text().splitlines()]
    assert len(oracle_rows) == 6
    assert all(row.inject_sketch_at == 2 for row in oracle_rows)
    assert await Runner(branch_source.model_copy(update={"seeds": [0]}), client).run()
    assert await Runner(config.model_copy(update={"source_seed": 0}), client).run()
    AuditRunner().run(root, None, None, {"experiment": "continue", "source_seed": 0})
    rows = [AuditRecord.model_validate_json(line) for line in compiled.read_text().splitlines()]
    assert len(rows) == 12
    assert {(row.source_seed, row.seed) for row in rows} == {(0, 2), (0, 3), (1, 2), (1, 3)}


@pytest.mark.parametrize("options", [
    [], ["--source-arm", "unaided-5x"],
    ["--source-arm", "unaided-5x", "--source-seed", "0", "--continue-at", "5"],
    ["--source-arm", "unaided-1x", "--source-seed", "0", "--continue-at", "2"],
    ["--source-arm", "unaided-5x", "--source-seed", "-1", "--continue-at", "1"],
    ["--source-arm", "../unaided-5x", "--source-seed", "0", "--continue-at", "1"],
    ["--source-arm", "unaided-5x", "--source-seed", "0", "--continue-at", "1", "--inject-sketch-at", "1"],
])
def test_invalid_branch_selection(run_arguments: list[str], options: list[str]) -> None:
    """Reject incomplete or ambiguous source selections before creating output directories."""
    with pytest.raises(SystemExit):
        parse_arguments([*run_arguments, "--experiment", "continue-oracle", *options])


@pytest.mark.parametrize("experiment", ["unaided", "oracle-execution"])
def test_root_run_rejects_source_options(run_arguments: list[str], experiment: str) -> None:
    """Do not silently ignore source options on independent root runs."""
    with pytest.raises(SystemExit):
        parse_arguments([*run_arguments, "--experiment", experiment, "--source-seed", "0"])


def test_audit_accepts_source_filters(run_arguments: list[str]) -> None:
    """Audit filters use the same flag names and keep source and branch seeds distinct."""
    args = parse_audit([*run_arguments, "--experiment", "continue-oracle", "--source-seed", "1", "--continue-at", "2"])
    assert args.source_seed == 1 and args.seeds == [0] and args.continue_at == 2


@pytest.mark.parametrize("experiment", ["continue", "continue-oracle"])
async def test_resume_partial_branch(experiment: str, make_branch_config: Callable[[str, int], RunConfig],
                                     client: OpenAIClient, records: list[dict[str, Any]],
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue from a saved branch response without regenerating its parent or completed branch output."""
    config = make_branch_config(experiment, 2).model_copy(update={"seeds": [2]})
    original_save = Storage.save
    begin = len(records)

    def interrupt(storage: Storage, state: RunState) -> None:
        """Crash after saving the first branch checkpoint."""
        original_save(storage, state)
        if state.output_tokens == 3 * OUTPUT_TOKENS_PER_BLOCK:
            raise RuntimeError("interrupted branch")

    with monkeypatch.context() as patch:
        patch.setattr(Storage, "save", interrupt)
        assert not await Runner(config, client).run()
    assert sum(record["count"] for record in records[begin:]) == OUTPUT_TOKENS_PER_BLOCK
    assert await Runner(config, client).run()
    assert sum(record["count"] for record in records[begin:]) == 3 * OUTPUT_TOKENS_PER_BLOCK
    root = model_output_directory(config.output_directory, config.competition, config.year, config.model)
    AuditRunner().run(root, None, None, {"experiment": experiment})
    changed = config.model_copy(update={"compute_multiplier_k": 4})
    count = len(records)
    assert not await Runner(changed, client).run()
    assert len(records) == count


async def test_source_seed_changes_branch_sampling(make_branch_config: Callable[[str, int], RunConfig],
                                                    branch_source: RunConfig, client: OpenAIClient,
                                                    problem: Problem, records: list[dict[str, Any]]) -> None:
    """Identically numbered branches from different parents must not reuse the same request seed."""
    assert await Runner(branch_source.model_copy(update={"seeds": [0]}), client).run()
    config = make_branch_config("continue", 1)
    seeds = []
    for source_seed in (0, 1):
        refinement = SelfRefine(config.model_copy(update={"source_seed": source_seed}), problem, 2, client)
        begin = len(records)
        await refinement.run()
        seeds.append(records[begin]["payload"]["seed"])
    assert seeds[0] != seeds[1]
