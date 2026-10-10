"""Exercise run selection, intervention boundaries, and full-budget collection."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import zstandard

from discovery_execution.harness.constants import (
    AUDIT_FILENAME, CHECKPOINT_FILENAME, CONVERSATION_FILENAME, GENERATION_RETRY_DELAYS, MIN_CONTEXT_TOKENS, ORACLE_EXECUTION_EXPERIMENT, OUTPUT_TOKENS_PER_BLOCK, SUPPORTED_MODELS, TOTAL_BLOCKS, TOTAL_OUTPUT_TOKENS, UNAIDED_EXPERIMENT,
)
from discovery_execution.harness.models import GenerationRequest, GenerationResponse, Problem, RunConfig, RunState
from discovery_execution.harness.openai_client import OpenAIClient
from discovery_execution.harness.storage import Storage
from discovery_execution.harness.self_refine import SelfRefine
from discovery_execution.audit.runner import AuditRunner
from discovery_execution.cli.run import backend_configuration, parse_arguments
from discovery_execution.harness.runner import Runner


def test_missing_endpoint_fails(backend_environment: dict[str, str]) -> None:
    """A missing endpoint fails instead of silently selecting a server."""
    del backend_environment["VLLM_BASE_URL"]
    with pytest.raises(KeyError, match="VLLM_BASE_URL"):
        backend_configuration(backend_environment)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
@pytest.mark.parametrize("experiment,injection,compute_multiplier_k", [
    *[(UNAIDED_EXPERIMENT, 0, block) for block in range(1, TOTAL_BLOCKS + 1)],
    *[(ORACLE_EXECUTION_EXPERIMENT, block, TOTAL_BLOCKS) for block in range(TOTAL_BLOCKS)],
])
async def test_full_trajectory(experiment: str, injection: int, compute_multiplier_k: int, model_name: str, problem: Problem,
                               make_run_config: Callable[[str, int], RunConfig],
                               client: OpenAIClient, records: list[dict[str, Any]]) -> None:
    """Every arm exhausts its own budget despite early success and skips completed runs on resume."""
    config = make_run_config(experiment, injection)
    batch = Runner(config, client)
    assert await batch.run()
    request_count = len(records)
    assert await batch.run()
    assert len(records) == request_count
    root = config.output_directory / "aime" / "2024" / model_name
    arm = f"unaided-{compute_multiplier_k}x" if experiment == UNAIDED_EXPERIMENT else f"oracle-at-{injection}x-execution"
    directory = root / arm / problem.problem_id / "seed_0"
    state = RunState.model_validate_json(zstandard.decompress((directory / CONVERSATION_FILENAME).read_bytes()))
    assert state.output_tokens == compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK
    snapshots = json.loads(zstandard.decompress((directory / CHECKPOINT_FILENAME).read_bytes()))
    assert len(snapshots) == compute_multiplier_k
    assert [item.output_tokens for item in state.checkpoints] == [
        block * OUTPUT_TOKENS_PER_BLOCK for block in range(1, compute_multiplier_k + 1)
    ]
    assert state.requests[0]["phase"] == "solve"
    assert "critique" in {item["phase"] for item in state.requests}
    if compute_multiplier_k > 1:
        assert "revise" in {item["phase"] for item in state.requests}
    assert not (directory / AUDIT_FILENAME).exists()
    AuditRunner().run(root, None, None, {})
    audit = json.loads((directory / AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert audit["complete"] and all(item["correctness"]["score"] == 0 for item in audit["checkpoints"].values())
    assert_intervention(records, experiment, injection, compute_multiplier_k)


def test_unsupported_model_fails(make_run_config: Callable[[str, int], RunConfig]) -> None:
    """Reject arbitrary model names even when configuration bypasses the CLI."""
    config = make_run_config(UNAIDED_EXPERIMENT, 0)
    with pytest.raises(ValueError, match="unsupported model"):
        RunConfig.model_validate({**config.model_dump(), "model": "unknown-model"})


async def test_wrong_served_name_fails(client: OpenAIClient) -> None:
    """Fail during preflight when the endpoint does not serve the selected alias."""
    with pytest.raises(ValueError, match="--served-model-name qwen3-1.7b"):
        await client.inspect("qwen3-1.7b")


@pytest.mark.parametrize("experiment,multiplier,expected", [
    (UNAIDED_EXPERIMENT, None, None), (UNAIDED_EXPERIMENT, 0, None),
    (UNAIDED_EXPERIMENT, TOTAL_BLOCKS + 1, None), (UNAIDED_EXPERIMENT, 2, 2),
    (ORACLE_EXECUTION_EXPERIMENT, None, TOTAL_BLOCKS),
    (ORACLE_EXECUTION_EXPERIMENT, TOTAL_BLOCKS, TOTAL_BLOCKS),
    (ORACLE_EXECUTION_EXPERIMENT, 2, None),
])
def test_cli_budget_selection(run_arguments: list[str], experiment: str,
                              multiplier: int | None, expected: int | None) -> None:
    """Require an explicit unaided horizon and reject attempts to shorten oracle runs."""
    arguments = [*run_arguments, "--experiment", experiment]
    if multiplier is not None:
        arguments.extend(["--compute-multiplier-k", str(multiplier)])
    if expected is None:
        with pytest.raises(SystemExit):
            parse_arguments(arguments)
    else:
        assert parse_arguments(arguments).compute_multiplier_k == expected


@pytest.mark.parametrize("context_tokens", [MIN_CONTEXT_TOKENS - 1])
async def test_small_context_fails_before_generation(make_run_config: Callable[[str, int], RunConfig],
                                                     client: OpenAIClient, records: list[dict[str, Any]]) -> None:
    """Reject an undersized server before creating any trajectory output."""
    config = make_run_config(UNAIDED_EXPERIMENT, 0)
    with pytest.raises(ValueError, match="max_model_len must be at least"):
        await Runner(config, client).run()
    assert not records
    assert not config.output_directory.exists()


@pytest.mark.parametrize("compute_multiplier_k", [1])
async def test_unaided_arms_share_prompts_but_not_sampling_seeds(make_run_config: Callable[[str, int], RunConfig],
                                                               client: OpenAIClient, records: list[dict[str, Any]]) -> None:
    """Separate budgets use identical instructions but distinct reproducible sampling streams."""
    config = make_run_config(UNAIDED_EXPERIMENT, 0)
    assert await Runner(config, client).run()
    first_request = records[0]["payload"]
    records.clear()
    longer = RunConfig.model_validate({**config.model_dump(), "compute_multiplier_k": 2})
    assert await Runner(longer, client).run()
    assert records[0]["payload"]["messages"] == first_request["messages"]
    assert records[0]["payload"]["seed"] != first_request["seed"]


def assert_intervention(records: list[dict[str, Any]], experiment: str, injection: int, compute_multiplier_k: int) -> None:
    """Inspect actual HTTP payloads for sketch timing, continuation, and reference isolation."""
    consumed = 0
    for record in records:
        payload = record["payload"]
        messages = payload["messages"]
        text = "\n".join(message["content"] for message in messages)
        expected = experiment == ORACLE_EXECUTION_EXPERIMENT and consumed >= injection * OUTPUT_TOKENS_PER_BLOCK
        assert text.count("SELECTED_SKETCH") == int(expected)
        assert "PRIVATE_REFERENCE" not in text
        assert payload["max_tokens"] == OUTPUT_TOKENS_PER_BLOCK - consumed % OUTPUT_TOKENS_PER_BLOCK
        if consumed and consumed % OUTPUT_TOKENS_PER_BLOCK == 0:
            injecting_now = experiment == ORACLE_EXECUTION_EXPERIMENT and consumed == injection * OUTPUT_TOKENS_PER_BLOCK
            assert payload["continue_final_message"] is (not injecting_now)
        consumed += record["count"]
    assert consumed == compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK


async def test_missing_sketch_fails_before_requests(make_run_config: Callable[[str, int], RunConfig],
                                                   client: OpenAIClient, records: list[dict[str, Any]]) -> None:
    """An invalid solution selection fails before creating trajectories or contacting the server."""
    config = make_run_config(ORACLE_EXECUTION_EXPERIMENT, 0)
    config = config.model_copy(update={"solution_index": 1})
    batch = Runner(config, client)
    with pytest.raises(ValueError, match="missing solution index"):
        await batch.run()
    assert not records
    assert not config.output_directory.exists()


@pytest.mark.parametrize("experiment,injection,compute_multiplier_k", [
    (UNAIDED_EXPERIMENT, 0, 2), (ORACLE_EXECUTION_EXPERIMENT, 1, TOTAL_BLOCKS),
])
async def test_resume_after_saved_boundary(experiment: str, injection: int, compute_multiplier_k: int,
                                           make_run_config: Callable[[str, int], RunConfig],
                                           client: OpenAIClient, records: list[dict[str, Any]],
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume a short unaided or delayed oracle run without repeating saved output."""
    config = make_run_config(experiment, injection)
    original_save = Storage.save

    def interrupt_after_save(storage: Storage, state: RunState) -> None:
        """Simulate process failure immediately after a durable boundary checkpoint."""
        original_save(storage, state)
        if state.output_tokens == OUTPUT_TOKENS_PER_BLOCK:
            raise RuntimeError("simulated process failure")

    batch = Runner(config, client)
    with monkeypatch.context() as patch:
        patch.setattr(Storage, "save", interrupt_after_save)
        assert not await batch.run()
    assert sum(record["count"] for record in records) == OUTPUT_TOKENS_PER_BLOCK
    # A restarted Slurm server receives a new local port.
    monkeypatch.setattr(client, "_config", client.configuration.model_copy(update={"base_url": "http://localhost:23456/v1"}))
    assert await batch.run()
    assert_intervention(records, experiment, injection, compute_multiplier_k)


@pytest.mark.parametrize("completed_requests", range(5))
async def test_resume_after_interrupted_response(completed_requests: int,
                                                make_run_config: Callable[[str, int], RunConfig],
                                                client: OpenAIClient, records: list[dict[str, Any]],
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry interrupted solve, critique, revision, continuation, and sketch injection without partial text."""
    config = make_run_config(ORACLE_EXECUTION_EXPERIMENT, 1)
    original_generate = client._generate_once
    interrupted: list[GenerationRequest] = []

    async def disconnect(request: GenerationRequest) -> GenerationResponse:
        """Exhaust automatic retries after the selected number of saved responses."""
        if len(records) == completed_requests:
            interrupted.append(request)
            raise ConnectionError("simulated disconnect")
        return await original_generate(request)

    with monkeypatch.context() as patch:
        patch.setattr(client, "_generate_once", disconnect)
        assert not await Runner(config, client).run()
    checkpoints = list(config.output_directory.rglob(CONVERSATION_FILENAME))
    if completed_requests:
        state = RunState.model_validate_json(zstandard.decompress(checkpoints[0].read_bytes()))
        assert len(state.requests) == completed_requests
        assert state.output_tokens == sum(record["count"] for record in records)
    else:
        assert not checkpoints
    assert await Runner(config, client).run()
    retried = records[completed_requests]["payload"]
    assert retried["messages"] == interrupted[0].messages
    assert retried["seed"] == interrupted[0].seed
    assert retried["max_tokens"] == interrupted[0].max_tokens
    assert retried["continue_final_message"] == interrupted[0].continue_response
    assert_intervention(records, ORACLE_EXECUTION_EXPERIMENT, 1, TOTAL_BLOCKS)
    assert not list(config.output_directory.rglob("requests"))
    for path in config.output_directory.rglob("*.json*"):
        content = path.read_bytes()
        if path.suffix == ".zst":
            content = zstandard.decompress(content)
        assert b"UNFINISHED_RESPONSE" not in content


@pytest.mark.parametrize("compute_multiplier_k", [2])
async def test_automatic_retry_keeps_saved_progress(make_run_config: Callable[[str, int], RunConfig],
                                                   client: OpenAIClient, records: list[dict[str, Any]],
                                                   retry_delays: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover a temporary failure without restarting the trajectory or duplicating completed output."""
    config = make_run_config(UNAIDED_EXPERIMENT, 0)
    original_generate = client._generate_once
    failed: list[GenerationRequest] = []

    async def disconnect_once(request: GenerationRequest) -> GenerationResponse:
        """Fail the critique once, after the solve response has been saved."""
        if len(records) == 1 and not failed:
            failed.append(request)
            raise ConnectionError("temporary disconnect")
        return await original_generate(request)

    monkeypatch.setattr(client, "_generate_once", disconnect_once)
    assert await Runner(config, client).run()
    path, = config.output_directory.rglob(CONVERSATION_FILENAME)
    state = RunState.model_validate_json(zstandard.decompress(path.read_bytes()))
    assert state.messages[1]["content"] == r"\boxed{42}"
    assert len(state.requests) == len(records)
    assert state.output_tokens == config.compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK
    assert records[1]["payload"]["messages"] == failed[0].messages
    assert records[1]["payload"]["seed"] == failed[0].seed
    assert retry_delays == [GENERATION_RETRY_DELAYS[0]]


async def test_resume_after_checkpoint_save_failure(make_run_config: Callable[[str, int], RunConfig],
                                                    client: OpenAIClient, records: list[dict[str, Any]],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Regenerate a received but unsaved response from the last durable checkpoint."""
    config = make_run_config(ORACLE_EXECUTION_EXPERIMENT, 1)
    original_save = Storage.save

    def fail_before_save(storage: Storage, state: RunState) -> None:
        """Leave the previous checkpoint intact when the next save fails."""
        if len(state.requests) == 2:
            raise OSError("simulated checkpoint failure")
        original_save(storage, state)

    with monkeypatch.context() as patch:
        patch.setattr(Storage, "save", fail_before_save)
        assert not await Runner(config, client).run()
    path, = config.output_directory.rglob(CONVERSATION_FILENAME)
    state = RunState.model_validate_json(zstandard.decompress(path.read_bytes()))
    assert len(state.requests) == 1
    assert state.output_tokens == records[0]["count"]
    discarded = records.pop()
    assert await Runner(config, client).run()
    assert records[1]["payload"] == discarded["payload"]
    assert_intervention(records, ORACLE_EXECUTION_EXPERIMENT, 1, TOTAL_BLOCKS)


async def test_each_trajectory_owns_its_state_and_storage(problem: Problem, local_partition: Path,
                                                         make_run_config: Callable[[str, int], RunConfig],
                                                         client: OpenAIClient, activity: dict[str, int],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise overlapping runs through the shared client while each owns its prompts, state, and files."""
    config = make_run_config(UNAIDED_EXPERIMENT, 0)
    config = config.model_copy(update={"seeds": [0, 1], "batch_size": 2})
    second = problem.model_copy(update={"problem_id": "second", "statement": "A different problem."})
    with local_partition.open("a", encoding="utf-8") as handle:
        handle.write(second.model_dump_json() + "\n")
    instances: list[SelfRefine] = []

    def create(config: RunConfig, problem: Problem, seed: int, client: OpenAIClient) -> SelfRefine:
        """Capture actual per-trajectory instances without replacing their execution logic."""
        refinement = SelfRefine(config, problem, seed, client)
        instances.append(refinement)
        return refinement

    monkeypatch.setattr("discovery_execution.harness.runner.SelfRefine", create)
    batch = Runner(config, client)
    assert await batch.run()
    count = len(config.seeds) * len(batch.dataset)
    assert len(instances) == count
    assert activity["peak"] == config.batch_size
    assert len({id(item.state) for item in instances}) == count
    assert len({id(item.prompts) for item in instances}) == count
    assert len({item.storage.directory for item in instances}) == count
    for item in instances:
        assert item.state.output_tokens == TOTAL_OUTPUT_TOKENS
        assert len(item.state.checkpoints) == TOTAL_BLOCKS
        assert item.problem.statement in item.state.messages[0]["content"]
        settings = item.storage.read_run()
        assert settings["problem_id"] == item.problem.problem_id
        assert not {"problem", "solutions", "prompts", "sketch", "final_answer"} & settings.keys()
