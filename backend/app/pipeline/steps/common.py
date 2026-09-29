"""Shared helper for LLM-backed steps: send a rendered prompt, parse the structured answer."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from app.llm.base import (
    META_ATTEMPT,
    META_DOC_ID,
    META_STEP,
    ChatMessage,
    LLMClient,
    LLMError,
    LLMOutputError,
    LLMRequest,
    LLMResponse,
    Role,
)
from app.llm.prompts import REPAIR_PROMPT, PromptTemplate
from app.llm.structured import parse_scored_output
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
    max_repair_attempts: int = 1,
) -> ModelT:
    """Render ``prompt``, call the LLM in JSON mode and validate the answer.

    If the answer is invalid, the LLM is shown its answer and the problems and asked again, up to
    ``max_repair_attempts`` times. The answer's self-reported ``confidence`` (1-5) is split off.
    Inside a trace, the prompt versions, every attempt and the confidence are recorded on the
    current span. Raises ``LLMProviderError`` if a call fails and ``LLMOutputError`` if the last
    answer is still invalid.
    """
    messages = prompt.render(**variables)
    span = current_span()
    if span is not None:
        span.set_prompt(prompt.name, prompt.version)
    attempt = 1
    while True:
        request = LLMRequest(
            messages=messages,
            temperature=temperature,
            json_mode=True,
            metadata={META_DOC_ID: doc_id, META_STEP: step.value, META_ATTEMPT: str(attempt)},
        )
        try:
            response = llm.complete(request)
        except LLMError as exc:
            _record_call(span, attempt, messages, None, f"{type(exc).__name__}: {exc}")
            raise
        try:
            scored = parse_scored_output(response.content, output_model)
        except LLMOutputError as exc:
            _record_call(span, attempt, messages, response, str(exc))
            if attempt > max_repair_attempts:
                raise
            messages = [
                *messages,
                ChatMessage(role=Role.ASSISTANT, content=response.content),
                REPAIR_PROMPT.render_user(problems=str(exc)),
            ]
            if span is not None:
                span.set_repair_prompt(REPAIR_PROMPT.version)
            attempt += 1
            continue
        _record_call(span, attempt, messages, response, None)
        if span is not None:
            span.set_confidence(scored.confidence, scored.confidence_note)
        return scored.value


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
            wait_ms=response.wait_ms if response else 0.0,
            error=error,
        )
    )


def to_prompt_json(model: BaseModel) -> str:
    """Serialize a model compactly for inclusion in a prompt."""
    return model.model_dump_json(exclude_none=True)
