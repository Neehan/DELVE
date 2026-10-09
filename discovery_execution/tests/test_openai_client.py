"""Check bounded retries, partial-stream isolation, and permanent failure handling."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import pytest

from discovery_execution.harness.constants import GENERATION_RETRY_DELAYS, RETRYABLE_HTTP_STATUS_CODES
from discovery_execution.harness.models import GenerationRequest
from discovery_execution.harness.openai_client import OpenAIClient

ClientFactory = Callable[[Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]],
                         AbstractAsyncContextManager[OpenAIClient]]


class InterruptedStream(httpx.AsyncByteStream):
    """Yield partial text and usage, then disconnect or raise the injected transport error."""

    def __init__(self, error: type[httpx.RequestError] | None) -> None:
        """Select the failure occurring after streamed text has reached the client."""
        self.error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Emit an uncommitted response before terminating the stream."""
        event = {"choices": [{"index": 0, "delta": {"content": "UNFINISHED_RESPONSE"}, "finish_reason": None}],
                 "usage": {"completion_tokens": 7}}
        yield f"data: {json.dumps(event)}\n\n".encode()
        if self.error is not None:
            raise self.error("interrupted stream")


@pytest.mark.parametrize("status", sorted(RETRYABLE_HTTP_STATUS_CODES))
async def test_transient_http_retry(status: int, make_client: ClientFactory, generation_request: GenerationRequest,
                                    completion_event: dict[str, Any], retry_delays: list[float]) -> None:
    """Retry temporary HTTP failures with identical messages, seed, and token allowance."""
    payloads: list[bytes] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        """Fail once, then complete the repeated request."""
        payloads.append(request.content)
        if len(payloads) == 1:
            return httpx.Response(status)
        return httpx.Response(200, text=f"data: {json.dumps(completion_event)}\n\ndata: [DONE]\n\n")

    async with make_client(respond) as client:
        response = await client.generate(generation_request)
    assert response.text == "COMPLETE_RESPONSE"
    assert payloads == [payloads[0], payloads[0]]
    assert retry_delays == [GENERATION_RETRY_DELAYS[0]]


@pytest.mark.parametrize("error", [None, httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError])
async def test_partial_stream_is_discarded(error: type[httpx.RequestError] | None, make_client: ClientFactory,
                                          generation_request: GenerationRequest, completion_event: dict[str, Any],
                                          retry_delays: list[float]) -> None:
    """Failed text and usage cannot leak into the successful retry's response."""
    payloads: list[bytes] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        """Break the first stream after partial output, then return a complete response."""
        payloads.append(request.content)
        if len(payloads) == 1:
            return httpx.Response(200, stream=InterruptedStream(error))
        return httpx.Response(200, text=f"data: {json.dumps(completion_event)}\n\ndata: [DONE]\n\n")

    async with make_client(respond) as client:
        response = await client.generate(generation_request)
    assert response.text == "COMPLETE_RESPONSE"
    assert response.usage == {"completion_tokens": 1}
    assert payloads == [payloads[0], payloads[0]]
    assert retry_delays == [GENERATION_RETRY_DELAYS[0]]


async def test_retry_limit(make_client: ClientFactory, generation_request: GenerationRequest,
                           retry_delays: list[float]) -> None:
    """Stop after the initial attempt and three retries, with no extra sleep after exhaustion."""
    attempts: list[bytes] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        """Keep the server unavailable throughout the bounded retry window."""
        attempts.append(request.content)
        return httpx.Response(503)

    async with make_client(respond) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate(generation_request)
    assert len(attempts) == len(GENERATION_RETRY_DELAYS) + 1
    assert len(set(attempts)) == 1
    assert retry_delays == list(GENERATION_RETRY_DELAYS)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_permanent_http_failure(status: int, make_client: ClientFactory,
                                      generation_request: GenerationRequest, retry_delays: list[float]) -> None:
    """Invalid requests and authentication failures fail immediately."""
    attempts: list[bytes] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        """Reject the request without a transient status."""
        attempts.append(request.content)
        return httpx.Response(status)

    async with make_client(respond) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate(generation_request)
    assert len(attempts) == 1
    assert not retry_delays


async def test_streamed_server_error(make_client: ClientFactory, generation_request: GenerationRequest,
                                     completion_event: dict[str, Any], retry_delays: list[float]) -> None:
    """Retry a transient server error delivered inside an HTTP 200 event stream."""
    attempts: list[bytes] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        """Return an in-band error once before completing the same request."""
        attempts.append(request.content)
        if len(attempts) == 1:
            return httpx.Response(200, text='data: {"error": {"code": 503, "message": "unavailable"}}\n\n')
        return httpx.Response(200, text=f"data: {json.dumps(completion_event)}\n\ndata: [DONE]\n\n")

    async with make_client(respond) as client:
        assert (await client.generate(generation_request)).text == "COMPLETE_RESPONSE"
    assert len(attempts) == 2
    assert retry_delays == [GENERATION_RETRY_DELAYS[0]]


async def test_invalid_usage_is_not_retried(make_client: ClientFactory, generation_request: GenerationRequest,
                                           completion_event: dict[str, Any], retry_delays: list[float]) -> None:
    """A response that violates token accounting is a protocol error, not a transient failure."""
    completion_event["usage"]["completion_tokens"] = generation_request.max_tokens + 1

    async def respond(request: httpx.Request) -> httpx.Response:
        """Return an invalid usage report with normal stream termination."""
        return httpx.Response(200, text=f"data: {json.dumps(completion_event)}\n\ndata: [DONE]\n\n")

    async with make_client(respond) as client:
        with pytest.raises(ValueError, match="invalid completion-token count"):
            await client.generate(generation_request)
    assert not retry_delays


async def test_cancellation_is_not_retried(make_client: ClientFactory, generation_request: GenerationRequest,
                                          retry_delays: list[float]) -> None:
    """Task cancellation propagates immediately instead of starting another request."""
    async def respond(request: httpx.Request) -> httpx.Response:
        """Cancel while opening the generation stream."""
        raise asyncio.CancelledError()

    async with make_client(respond) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.generate(generation_request)
    assert not retry_delays
