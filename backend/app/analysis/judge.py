"""Ask an LLM judge to grade every step of one recorded run, in a single call.

The judge receives a "case file" built from the trace: the original document once, then what each
step was given and what it produced (or, for a step that crashed, its raw answers and the error).
It never sees the steps' self-reported confidence, so its verdict is independent of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.analysis.models import JudgeAnswer
from app.analysis.taxonomy import TAXONOMY
from app.llm.base import (
    META_ATTEMPT,
    META_DOC_ID,
    META_STEP,
    ChatMessage,
    LLMClient,
    LLMError,
    LLMOutputError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    Role,
)
from app.llm.prompts import JUDGE_REPAIR_PROMPT, TRACE_JUDGE_PROMPT
from app.llm.structured import parse_json_output
from app.pipeline.models import PipelineConfig, StepName
from app.tracing.models import LLMCallRecord, Span, SpanStatus, Trace

JUDGE_STEP = "judge"  # the step name in request metadata; mock judge scripts are keyed by it
_TAIL_CHARS = 160

_RECEIVES = {
    StepName.INTAKE: "the original document above.",
    StepName.EXTRACTION: "the text kept by intake.",
    StepName.CLASSIFICATION: "the text kept by intake and the extraction output above.",
    StepName.SUMMARIZATION: (
        "the text kept by intake, the extraction output and the classification output above."
    ),
}


class JudgeFailedError(LLMError):
    """The judge gave no usable verdict.

    ``calls`` records every attempt. ``provider_error`` is set when the provider itself failed
    (network, rate limit, used-up quota), as opposed to answering with something unusable.
    """

    def __init__(
        self,
        message: str,
        *,
        calls: list[LLMCallRecord],
        provider_error: LLMProviderError | None = None,
    ) -> None:
        super().__init__(message)
        self.calls = calls
        self.provider_error = provider_error


@dataclass(frozen=True)
class JudgeResult:
    """A validated verdict plus the record of every call it took."""

    answer: JudgeAnswer
    calls: list[LLMCallRecord]


def render_categories() -> str:
    """The taxonomy as prompt text: one line per category with the steps where it can start."""
    lines = []
    for definition in TAXONOMY:
        steps = ", ".join(step.value for step in StepName if step in definition.steps)
        lines.append(
            f"- {definition.category.value} ({definition.label}), can start at: {steps}. "
            f"{definition.definition}"
        )
    return "\n".join(lines)


def steps_that_ran(trace: Trace) -> list[StepName]:
    """The steps recorded in ``trace``, in pipeline order."""
    return [StepName(span.name) for span in trace.spans]


def build_case_file(trace: Trace) -> str:
    """Everything the judge needs about one run, as plain text."""
    spans = {StepName(span.name): span for span in trace.spans}
    raw: dict[str, Any] = spans[StepName.INTAKE].input
    content = str(raw["content"])
    parts = [
        f"Document ID: {trace.doc_id}",
        "",
        f"=== ORIGINAL DOCUMENT (format: {raw['format']}, {len(content)} characters) ===",
        "<<<",
        content,
        ">>>",
    ]
    for number, step in enumerate(StepName, start=1):
        parts += ["", f"=== STEP {number}: {step.value} ==="]
        span = spans.get(step)
        if span is None:
            parts.append("Did not run (the pipeline stopped earlier). Do not grade it.")
            continue
        parts.append(f"Received: {_RECEIVES[step]}")
        if span.status is SpanStatus.ERROR:
            parts += _failed_step(span)
        elif step is StepName.INTAKE:
            parts.append(_intake_output(span.output))
        else:
            parts += ["Output:", json.dumps(span.output, ensure_ascii=False)]
    return "\n".join(parts)


def judge_trace(
    trace: Trace, llm: LLMClient, *, temperature: float = 0.0, max_repair_attempts: int = 1
) -> JudgeResult:
    """Grade every step of ``trace`` in one LLM call (plus repair attempts if the answer is bad).

    Raises ``JudgeFailedError`` if the provider fails or the last answer is still unusable.
    """
    config = trace.config or PipelineConfig()
    messages = TRACE_JUDGE_PROMPT.render(
        max_chars=str(config.intake_max_chars),
        categories=render_categories(),
        case_file=build_case_file(trace),
    )
    expected = steps_that_ran(trace)
    calls: list[LLMCallRecord] = []
    attempt = 1
    while True:
        request = LLMRequest(
            messages=messages,
            temperature=temperature,
            json_mode=True,
            metadata={META_DOC_ID: trace.doc_id, META_STEP: JUDGE_STEP, META_ATTEMPT: str(attempt)},
        )
        try:
            response = llm.complete(request)
        except LLMError as exc:
            calls.append(_record(attempt, messages, None, f"{type(exc).__name__}: {exc}"))
            provider_error = exc if isinstance(exc, LLMProviderError) else None
            raise JudgeFailedError(str(exc), calls=calls, provider_error=provider_error) from exc
        try:
            answer = parse_json_output(response.content, JudgeAnswer)
            problems = answer.problems_for(expected)
            if problems:
                raise LLMOutputError("; ".join(problems), raw_output=response.content)
        except LLMOutputError as exc:
            calls.append(_record(attempt, messages, response, str(exc)))
            if attempt > max_repair_attempts:
                raise JudgeFailedError(str(exc), calls=calls) from exc
            messages = [
                *messages,
                ChatMessage(role=Role.ASSISTANT, content=response.content),
                JUDGE_REPAIR_PROMPT.render_user(problems=str(exc)),
            ]
            attempt += 1
            continue
        calls.append(_record(attempt, messages, response, None))
        return JudgeResult(answer=answer, calls=calls)


def _intake_output(output: dict[str, Any]) -> str:
    notes = "; ".join(output["cleanup_notes"]) or "none"
    if not output["truncated"]:
        return (
            f"Output: the cleaned text, {output['char_count']} characters, nothing cut off "
            f"(cleanup: {notes})."
        )
    tail = str(output["text"])[-_TAIL_CHARS:]
    return (
        f"Output: only the first {output['char_count']} characters were kept (TRUNCATED; "
        f'cleanup: {notes}). The kept text ends with:\n"...{tail}"\n'
        "Everything in the original document after that point was cut off and never reached the "
        "later steps."
    )


def _failed_step(span: Span) -> list[str]:
    error = span.error
    reason = f"{error.error_type}: {error.message}" if error else "unknown error"
    lines = [f"This step FAILED ({reason}) and produced no output."]
    for call in span.llm_calls:
        lines += [
            f"AI answer, attempt {call.attempt}:",
            "<<<",
            call.raw_response or "(no answer)",
            ">>>",
        ]
    return lines


def _record(
    attempt: int, messages: list[ChatMessage], response: LLMResponse | None, error: str | None
) -> LLMCallRecord:
    return LLMCallRecord(
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
