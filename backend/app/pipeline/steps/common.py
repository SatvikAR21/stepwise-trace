"""Shared helper for LLM-backed steps: send a rendered prompt, parse the structured answer."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from app.llm.base import (
    META_DOC_ID,
    META_STEP,
    ChatMessage,
    LLMClient,
    LLMError,
    LLMOutputError,
    LLMRequest,
    LLMResponse,
)
from app.llm.prompts import PromptTemplate
from app.llm.structured import parse_json_output
from app.pipeline.models import StepName
from app.tracing.models import LLMCallRecord
from app.tracing.tracer import SpanRecorder, current_span


def call_structured[ModelT: BaseModel](
    llm: LLMClient,
    prompt: PromptTemplate,
    variables: Mapping[str, str],
    output_model: type[ModelT],
    *,
    doc_id: str,
    step: StepName,
    temperature: float = 0.0,
) -> ModelT:
    """Render ``prompt``, make one JSON-mode LLM call and validate the answer.

    Inside a trace, the prompt version and every call (request, raw answer, tokens, latency, error)
    are recorded on the current span. Raises ``LLMProviderError`` if the call fails and
    ``LLMOutputError`` if the answer is invalid.
    """
    messages = prompt.render(**variables)
    span = current_span()
    if span is not None:
        span.set_prompt(prompt.name, prompt.version)
    request = LLMRequest(
        messages=messages,
        temperature=temperature,
        json_mode=True,
        metadata={META_DOC_ID: doc_id, META_STEP: step.value},
    )
    try:
        response = llm.complete(request)
    except LLMError as exc:
        _record_call(span, 1, messages, None, f"{type(exc).__name__}: {exc}")
        raise
    try:
        result = parse_json_output(response.content, output_model)
    except LLMOutputError as exc:
        _record_call(span, 1, messages, response, str(exc))
        raise
    _record_call(span, 1, messages, response, None)
    return result


def _record_call(
    span: SpanRecorder | None,
    attempt: int,
    messages: list[ChatMessage],
    response: LLMResponse | None,
    error: str | None,
) -> None:
    if span is None:
        return
    span.record_llm_call(
        LLMCallRecord(
            attempt=attempt,
            messages=messages,
            raw_response=response.content if response else None,
            model=response.model if response else None,
            prompt_tokens=response.usage.prompt_tokens if response else 0,
            completion_tokens=response.usage.completion_tokens if response else 0,
            latency_ms=response.latency_ms if response else 0.0,
            error=error,
        )
    )


def to_prompt_json(model: BaseModel) -> str:
    """Serialize a model compactly for inclusion in a prompt."""
    return model.model_dump_json(exclude_none=True)
