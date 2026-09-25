"""LLM client for any OpenAI-compatible chat API (OpenAI, Google Gemini, Groq, Ollama...)."""

from __future__ import annotations

import time

import httpx2
import openai
from openai.types.chat import ChatCompletionMessageParam

from app.llm.base import (
    ChatMessage,
    LLMClient,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    Role,
    TokenUsage,
)


def _to_openai_message(message: ChatMessage) -> ChatCompletionMessageParam:
    match message.role:
        case Role.SYSTEM:
            return {"role": "system", "content": message.content}
        case Role.USER:
            return {"role": "user", "content": message.content}
        case Role.ASSISTANT:
            return {"role": "assistant", "content": message.content}


class OpenAICompatibleClient(LLMClient):
    """Calls ``/chat/completions`` via the official ``openai`` SDK with a configurable base URL.

    The SDK retries rate-limit and transient errors with exponential backoff (``max_retries``).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        http_client: httpx2.Client | None = None,
    ) -> None:
        self._model = model
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
            http_client=http_client,
        )

    @property
    def model_name(self) -> str:
        """The configured model identifier."""
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one chat completion request. Wraps every SDK error in ``LLMProviderError``."""
        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[_to_openai_message(m) for m in request.messages],
                temperature=request.temperature,
                response_format={"type": "json_object"} if request.json_mode else openai.omit,
            )
        except openai.OpenAIError as exc:
            raise LLMProviderError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if not completion.choices:
            raise LLMProviderError("provider returned no choices")
        usage = completion.usage
        return LLMResponse(
            content=completion.choices[0].message.content or "",
            model=completion.model or self._model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            ),
            latency_ms=latency_ms,
        )
