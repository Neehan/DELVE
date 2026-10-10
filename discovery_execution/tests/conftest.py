"""Shared records, temporary datasets, and an isolated streaming model client."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from discovery_execution.harness.constants import (
    MIN_CONTEXT_TOKENS, OUTPUT_TOKENS_PER_BLOCK, TOTAL_BLOCKS,
)
from discovery_execution.harness.models import BackendConfig, Checkpoint, DatasetConfig, GenerationRequest, Problem, RunConfig, RunState, Solution
from discovery_execution.harness.openai_client import OpenAIClient
from discovery_execution.cli.run import backend_configuration, parse_arguments


@pytest.fixture
def problem() -> Problem:
    """Use distinct reference and sketch markers to detect leakage into model requests."""
    return Problem(problem_id="example", competition="aime", year=2024,
                   statement="Compute six times seven.", final_answer="42",
                   solutions=[Solution(source="reference", solution="PRIVATE_REFERENCE", sketch="SELECTED_SKETCH")])


@pytest.fixture
def backend_environment() -> dict[str, str]:
    """Supply isolated test settings without reading application environment files."""
    return {"VLLM_BASE_URL": "http://localhost:8000/v1"}


@pytest.fixture
def backend(backend_environment: dict[str, str]) -> BackendConfig:
    """Validate the model settings through the cli."""
    return backend_configuration(backend_environment)


@pytest.fixture
def dataset_config(tmp_path: Path, problem: Problem) -> DatasetConfig:
    """Point a single partition at a fresh temporary directory."""
    return DatasetConfig(competition=problem.competition, year=problem.year,
                         datasets_directory=tmp_path / "datasets",
                         dataset_repo_url="https://huggingface.co/datasets/owner/repo")


@pytest.fixture
def local_partition(dataset_config: DatasetConfig, problem: Problem) -> Path:
    """Write the shared problem into the partition expected by the dataset loader."""
    path = dataset_config.datasets_directory / problem.competition / f"{problem.year}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(problem.model_dump_json() + "\n", encoding="utf-8")
    return path


@pytest.fixture
def model_name() -> str:
    """Use the same supported name for CLI selection and the mock model server."""
    return "qwen3-4b-instruct-2507"


@pytest.fixture
def compute_multiplier_k() -> int:
    """Use the full budget unless a test selects a shorter unaided run."""
    return TOTAL_BLOCKS


@pytest.fixture
def context_tokens() -> int:
    """Serve the minimum accepted context capacity unless overridden by a test."""
    return MIN_CONTEXT_TOKENS


@pytest.fixture
def run_arguments(dataset_config: DatasetConfig, model_name: str) -> list[str]:
    """Supply shared CLI identity arguments without selecting an arm or budget."""
    return ["--competition", dataset_config.competition, "--year", str(dataset_config.year),
            "--model", model_name, "--seeds", "0"]


@pytest.fixture
def make_run_config(dataset_config: DatasetConfig, local_partition: Path,
                    tmp_path: Path, run_arguments: list[str], compute_multiplier_k: int) -> Callable[[str, int], RunConfig]:
    """Build CLI-validated arms against a prepared temporary partition."""
    def create(experiment: str, injection: int) -> RunConfig:
        """Select an arm and replace repository paths with test-owned directories."""
        config = parse_arguments([
            *run_arguments, "--experiment", experiment,
            "--compute-multiplier-k", str(compute_multiplier_k),
            "--inject-sketch-at", str(injection),
        ])
        return config.model_copy(update={**dataset_config.model_dump(), "output_directory": tmp_path / "output"})
    return create


@pytest.fixture
def conversation() -> RunState:
    """Represent a completed response at the first token boundary."""
    state = RunState.empty()
    text = r"\boxed{42}"
    state.messages = [{"role": "assistant", "content": text}]
    state.output_tokens = OUTPUT_TOKENS_PER_BLOCK
    state.requests = [{"output_tokens": OUTPUT_TOKENS_PER_BLOCK}]
    state.checkpoints = [Checkpoint(output_tokens=OUTPUT_TOKENS_PER_BLOCK,
                                    message_count=1, last_message_chars=len(text))]
    return state


@pytest.fixture
def records() -> list[dict[str, Any]]:
    """Capture generation payloads and server-reported token counts for one test."""
    return []


@pytest.fixture
def activity() -> dict[str, int]:
    """Track overlapping requests through the shared mock client."""
    return {"active": 0, "peak": 0}


@pytest.fixture
def retry_delays() -> list[float]:
    """Record retry waits without delaying tests."""
    return []


@pytest.fixture
def make_client(backend: BackendConfig, retry_delays: list[float]) -> Callable[
    [Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]], AbstractAsyncContextManager[OpenAIClient]
]:
    """Inject isolated HTTP handlers and a nonblocking retry clock."""
    async def sleep(delay: float) -> None:
        """Capture requested backoff while letting concurrent tasks run."""
        retry_delays.append(delay)
        await asyncio.sleep(0)

    @asynccontextmanager
    async def create(handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]) -> AsyncGenerator[OpenAIClient, None]:
        """Close the mock transport after each client test."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            yield OpenAIClient(transport, backend, sleep)
    return create


@pytest.fixture
def generation_request(model_name: str) -> GenerationRequest:
    """Provide a fixed request for validating identical retries."""
    return GenerationRequest(model_name, [{"role": "user", "content": "Solve the problem."}],
                             OUTPUT_TOKENS_PER_BLOCK, 123, False)


@pytest.fixture
def completion_event() -> dict[str, Any]:
    """Describe a completed stream with one output token and explicit usage."""
    return {"choices": [{"index": 0, "delta": {"content": "COMPLETE_RESPONSE"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1}}


@pytest.fixture
async def client(records: list[dict[str, Any]],
                 activity: dict[str, int], model_name: str, context_tokens: int,
                 make_client: Callable[[Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]],
                                       AbstractAsyncContextManager[OpenAIClient]]) -> AsyncGenerator[OpenAIClient, None]:
    """Serve metadata and streamed completions entirely in memory, closing transport after each test."""
    async def respond(request: httpx.Request) -> httpx.Response:
        """Alternate natural endings and truncation while permitting requests to overlap."""
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model_name, "max_model_len": context_tokens}]})
        if request.url.path.endswith("/version"):
            return httpx.Response(200, json={"version": "test"})
        assert request.url.path.endswith("/chat/completions")
        activity["active"] += 1
        activity["peak"] = max(activity["peak"], activity["active"])
        await asyncio.sleep(0)
        activity["active"] -= 1
        payload = json.loads(request.content)
        assert payload["model"] == model_name
        natural_end = len(records) % 2 == 0
        count = 1 if natural_end else payload["max_tokens"]
        text = r"\boxed{42}" if not records else r"\boxed{0}"
        records.append({"payload": payload, "count": count})
        event = {"choices": [{"index": 0, "delta": {"content": text},
                               "finish_reason": "stop" if natural_end else "length"}],
                 "usage": {"completion_tokens": count}}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    async with make_client(respond) as model_client:
        yield model_client
