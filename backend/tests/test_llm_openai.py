"""Tests for the OpenAI-compatible client, using a fake HTTP transport (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from app.llm.base import ChatMessage, LLMProviderError, LLMRequest, Role
from app.llm.openai_compatible import OpenAICompatibleClient

BASE_URL = "https://llm.example.test/v1"


def _completion(content: str | None = '{"ok": true}', *, choices: bool = True) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini-3.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
        if choices
        else [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _client(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        model="gemini-3.5-flash",
        base_url=BASE_URL,
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _request(json_mode: bool = True) -> LLMRequest:
    return LLMRequest(
        messages=[
            ChatMessage(role=Role.SYSTEM, content="be precise"),
            ChatMessage(role=Role.USER, content="hello"),
            ChatMessage(role=Role.ASSISTANT, content="earlier answer"),
        ],
        temperature=0.2,
        json_mode=json_mode,
    )


def test_sends_openai_chat_request_and_maps_response() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_completion())

    client = _client(handler)
    response = client.complete(_request())

    assert seen["url"] == f"{BASE_URL}/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "gemini-3.5-flash"
    assert seen["body"]["temperature"] == 0.2
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user", "assistant"]
    assert response.content == '{"ok": true}'
    assert response.model == "gemini-3.5-flash" == client.model_name
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.latency_ms >= 0


def test_json_mode_off_omits_response_format() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_completion())

    _client(handler).complete(_request(json_mode=False))

    assert "response_format" not in seen["body"]


def test_null_content_becomes_empty_string() -> None:
    client = _client(lambda _: httpx2.Response(200, json=_completion(content=None)))

    assert client.complete(_request()).content == ""


def test_no_choices_raises_provider_error() -> None:
    client = _client(lambda _: httpx2.Response(200, json=_completion(choices=False)))

    with pytest.raises(LLMProviderError, match="no choices"):
        client.complete(_request())


@pytest.mark.parametrize(
    ("status", "error_name"),
    [(401, "AuthenticationError"), (429, "RateLimitError"), (500, "InternalServerError")],
)
def test_http_errors_become_provider_errors(status: int, error_name: str) -> None:
    client = _client(lambda _: httpx2.Response(status, json={"error": {"message": "nope"}}))

    with pytest.raises(LLMProviderError, match=error_name):
        client.complete(_request())
