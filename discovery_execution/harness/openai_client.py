"""Streaming OpenAI-compatible chat requests with server-reported token accounting."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from discovery_execution.harness.constants import (
    DEFAULT_MIN_P, DEFAULT_PENALTY, DEFAULT_REPETITION_PENALTY, MIN_CONTEXT_TOKENS,
    GENERATION_RETRY_DELAYS, RETRYABLE_HTTP_STATUS_CODES, SSE_DONE, SSE_PREFIX,
)
from discovery_execution.harness.models import BackendConfig, GenerationRequest, GenerationResponse


class OpenAIClient:
    """Send standard chat messages and require complete, budget-valid usage reports."""

    def __init__(self, client: httpx.AsyncClient, config: BackendConfig,
                 sleep: Callable[[float], Awaitable[None]]) -> None:
        """Inject transport, sampling settings, and the asynchronous retry wait."""
        self._client = client
        self._config = config
        self._sleep = sleep

    @property
    def configuration(self) -> BackendConfig:
        """Expose immutable inference settings for saved run identity."""
        return self._config

    async def inspect(self, model: str) -> dict[str, Any]:
        """Confirm the served model and minimum context capacity before generation."""
        response = await self._client.get(f"{self._config.base_url}/models")
        response.raise_for_status()
        matches = [item for item in response.json()["data"] if item["id"] == model]
        if len(matches) != 1:
            raise ValueError(f"vLLM must serve exactly one model named {model}; use --served-model-name {model}")
        entry = matches[0]
        if entry["max_model_len"] < MIN_CONTEXT_TOKENS:
            raise ValueError(f"vLLM max_model_len must be at least {MIN_CONTEXT_TOKENS} tokens")
        endpoint = self._config.base_url.removesuffix("/v1")
        version = await self._client.get(f"{endpoint}/version")
        version.raise_for_status()
        return {"model": entry["id"], "max_model_len": entry["max_model_len"],
                "version": version.json()}

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Retry transient failures with the same request, returning only a complete response."""
        retries = 0
        while True:
            try:
                return await self._generate_once(request)
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in RETRYABLE_HTTP_STATUS_CODES:
                    raise
                failure = error
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, ConnectionError) as error:
                failure = error
            if retries == len(GENERATION_RETRY_DELAYS):
                raise failure
            delay = GENERATION_RETRY_DELAYS[retries]
            retries += 1
            logging.warning("Generation failed for %s seed=%s; retry %s/%s in %ss: %s",
                            request.model, request.seed, retries, len(GENERATION_RETRY_DELAYS), delay, failure)
            await self._sleep(delay)

    async def _generate_once(self, request: GenerationRequest) -> GenerationResponse:
        """Buffer one stream; failed attempts discard all partial text and usage."""
        fragments: list[str] = []
        finish: str | None = None
        usage: dict[str, Any] | None = None
        async with self._client.stream("POST", f"{self._config.base_url}/chat/completions",
                                       json=self._payload(request)) as response:
            response.raise_for_status()
            async for data in self._events(response):
                if data == SSE_DONE:
                    self._validate_completion(request, finish, usage)
                    assert finish is not None and usage is not None
                    return GenerationResponse("".join(fragments), finish, usage)
                event = json.loads(data)
                if "error" in event:
                    code = event["error"].get("code")
                    if code in RETRYABLE_HTTP_STATUS_CODES:
                        raise httpx.HTTPStatusError(f"vLLM stream error: {event['error']}",
                                                    request=response.request,
                                                    response=httpx.Response(code, request=response.request))
                    raise RuntimeError(f"vLLM stream error: {event['error']}")
                for choice in event["choices"]:
                    if finish is not None or choice["index"] != 0:
                        raise ValueError("unexpected choice or output after termination")
                    delta = choice["delta"]
                    if delta.get("reasoning") or delta.get("reasoning_content"):
                        raise ValueError("serve raw thinking text: disable the vLLM reasoning parser")
                    if delta.get("tool_calls"):
                        raise ValueError("tool calls are not supported")
                    finish = choice["finish_reason"]
                    fragments.append(delta.get("content") or "")
                if event.get("usage") is not None:
                    usage = event["usage"]
        raise ConnectionError("vLLM disconnected without [DONE]; response discarded")

    async def _events(self, response: httpx.Response) -> AsyncIterator[str]:
        """Read complete SSE events while ignoring keepalive comments."""
        lines: list[str] = []
        async for line in response.aiter_lines():
            if not line:
                if lines:
                    yield "\n".join(lines)
                    lines = []
            elif line.startswith(SSE_PREFIX):
                lines.append(line.removeprefix(SSE_PREFIX))

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Continue an unfinished assistant message without inserting a user instruction."""
        return {
            "model": request.model, "messages": request.messages, "max_tokens": request.max_tokens,
            "seed": request.seed, "temperature": self._config.temperature,
            "top_p": self._config.top_p, "top_k": self._config.top_k,
            "min_p": DEFAULT_MIN_P, "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "presence_penalty": DEFAULT_PENALTY, "frequency_penalty": DEFAULT_PENALTY,
            "n": 1, "stream": True, "stream_options": {"include_usage": True},
            "continue_final_message": request.continue_response,
            "add_generation_prompt": not request.continue_response,
        }

    def _validate_completion(self, request: GenerationRequest, finish: str | None,
                             usage: dict[str, Any] | None) -> None:
        """Use server token counts instead of tokenizing text or counting stream chunks."""
        if finish not in {"stop", "length"} or usage is None:
            raise ValueError("vLLM must return a finish reason and final usage")
        count = usage["completion_tokens"]
        if not 0 < count <= request.max_tokens:
            raise ValueError("invalid completion-token count")
        if finish == "length" and count != request.max_tokens:
            raise ValueError("server context or output cap was reached before the requested boundary")
