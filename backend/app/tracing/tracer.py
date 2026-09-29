"""Record pipeline runs: an active trace, a span per step, and a one-line step decorator.

Usage::

    @traced_step(StepName.EXTRACTION)
    def run_extraction(document, llm): ...

    with start_trace("doc_1", model="mock-llm") as recorder:
        result = run_pipeline(document, llm)
    trace = recorder.finish(result)

The active trace and span live in context variables, so code deep inside a step (such as the LLM
call helper) can add details to the current span without passing it around. Outside a trace the
decorator does nothing, so steps behave exactly the same in plain unit tests.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from typing import Any

import structlog
from pydantic import BaseModel

from app.core.logging import get_logger
from app.pipeline.models import PipelineConfig, PipelineResult
from app.tracing.models import (
    FeatureValue,
    LLMCallRecord,
    Span,
    SpanError,
    SpanStatus,
    Trace,
)
from app.tracing.status import assess

FeatureExtractor = Callable[[Any, Any], dict[str, FeatureValue]]

logger = get_logger(__name__)


def new_trace_id() -> str:
    """32 random hex characters (16 bytes), the OpenTelemetry trace-id format."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """16 random hex characters (8 bytes), the OpenTelemetry span-id format."""
    return secrets.token_hex(8)


def _now() -> datetime:
    return datetime.now(UTC)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if value is None or isinstance(value, str | int | float | bool | list | dict):
        return value
    return repr(value)


def _json_len(value: Any) -> int:
    return 0 if value is None else len(json.dumps(value, ensure_ascii=False))


class SpanRecorder:
    """Collects the details of one step while it runs."""

    def __init__(self, trace_id: str, name: str, parent_span_id: str | None = None) -> None:
        self.trace_id = trace_id
        self.span_id = new_span_id()
        self.parent_span_id = parent_span_id
        self.name = name
        self.input: Any = None
        self.output: Any = None
        self.prompt_name: str | None = None
        self.prompt_version: str | None = None
        self.repair_prompt_version: str | None = None
        self.llm_calls: list[LLMCallRecord] = []
        self.confidence: int | None = None
        self.confidence_note: str | None = None
        self.error: SpanError | None = None
        self.features: dict[str, FeatureValue] = {}

    def set_prompt(self, name: str, version: str) -> None:
        """Remember which prompt template (and version) the step used."""
        self.prompt_name, self.prompt_version = name, version

    def set_repair_prompt(self, version: str) -> None:
        """Remember which version of the repair prompt was sent after an invalid answer."""
        self.repair_prompt_version = version

    def record_llm_call(self, record: LLMCallRecord) -> None:
        """Add one LLM request/response (or failed attempt) to the span."""
        self.llm_calls.append(record)

    def set_confidence(self, value: int | None, note: str | None = None) -> None:
        """Store the step's self-reported confidence (1-5), or why it is missing."""
        self.confidence, self.confidence_note = value, note

    def to_span(
        self, status: SpanStatus, started_at: datetime, ended_at: datetime, duration_ms: float
    ) -> Span:
        """Freeze the collected details into an immutable ``Span`` record."""
        return Span(
            span_id=self.span_id,
            trace_id=self.trace_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            input=self.input,
            output=self.output,
            prompt_name=self.prompt_name,
            prompt_version=self.prompt_version,
            repair_prompt_version=self.repair_prompt_version,
            llm_calls=self.llm_calls,
            confidence=self.confidence,
            confidence_note=self.confidence_note,
            error=self.error,
            features={**self._common_features(duration_ms), **self.features},
        )

    def _common_features(self, duration_ms: float) -> dict[str, FeatureValue]:
        """Measurements every span gets; step-specific features are added by the decorator."""
        features: dict[str, FeatureValue] = {
            "duration_ms": round(duration_ms, 3),
            "input_chars": _json_len(self.input),
            "output_chars": _json_len(self.output),
        }
        calls = self.llm_calls
        if calls:
            features |= {
                "llm_attempts": len(calls),
                "repair_attempts": len(calls) - 1,
                "prompt_tokens": sum(c.prompt_tokens for c in calls),
                "completion_tokens": sum(c.completion_tokens for c in calls),
                "llm_latency_ms": round(sum(c.latency_ms for c in calls), 3),
                "wait_ms": round(sum(c.wait_ms for c in calls), 3),
                "model": calls[-1].model,
                "prompt_version": self.prompt_version,
                "confidence": self.confidence,
            }
        return features


class TraceRecorder:
    """Collects the spans of one run while it executes."""

    def __init__(self, doc_id: str, model: str, config: PipelineConfig | None = None) -> None:
        self.trace_id = new_trace_id()
        self.doc_id = doc_id
        self.model = model
        self.config = config
        self.started_at = _now()
        self._started = time.perf_counter()
        self.spans: list[Span] = []

    def add(self, span: Span) -> None:
        """Append a finished span."""
        self.spans.append(span)

    def finish(self, result: PipelineResult) -> Trace:
        """Build the final ``Trace``, including its status and score."""
        verdict = assess(self.spans, result.status)
        return Trace(
            trace_id=self.trace_id,
            doc_id=self.doc_id,
            model=self.model,
            config=self.config,
            started_at=self.started_at,
            ended_at=_now(),
            duration_ms=(time.perf_counter() - self._started) * 1000,
            status=verdict.status,
            status_reasons=verdict.reasons,
            final_score=verdict.final_score,
            pipeline_status=result.status,
            error_step=result.error.step if result.error else None,
            spans=self.spans,
            final_output=_to_jsonable(result.summary),
        )


_current_trace: ContextVar[TraceRecorder | None] = ContextVar("current_trace", default=None)
_current_span: ContextVar[SpanRecorder | None] = ContextVar("current_span", default=None)


def current_span() -> SpanRecorder | None:
    """The span being recorded right now, or ``None`` outside a trace."""
    return _current_span.get()


@contextmanager
def start_trace(
    doc_id: str, *, model: str, config: PipelineConfig | None = None
) -> Iterator[TraceRecorder]:
    """Record every traced step that runs inside this block into one trace.

    ``config`` is stored with the trace so the run can be repeated with the same settings.
    While the block runs, every log line also carries the ``trace_id``.
    """
    recorder = TraceRecorder(doc_id, model, config)
    token = _current_trace.set(recorder)
    try:
        with structlog.contextvars.bound_contextvars(trace_id=recorder.trace_id):
            yield recorder
    finally:
        _current_trace.reset(token)


@contextmanager
def span(name: str) -> Iterator[SpanRecorder | None]:
    """Record the enclosed block as one span of the active trace (no-op outside a trace).

    Timing and any exception are captured; the exception is re-raised unchanged.
    """
    trace = _current_trace.get()
    if trace is None:
        yield None
        return
    parent = _current_span.get()
    recorder = SpanRecorder(trace.trace_id, name, parent.span_id if parent else None)
    token = _current_span.set(recorder)
    started_at, started = _now(), time.perf_counter()
    status = SpanStatus.OK
    try:
        yield recorder
    except Exception as exc:
        status = SpanStatus.ERROR
        recorder.error = SpanError(error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        _current_span.reset(token)
        duration_ms = (time.perf_counter() - started) * 1000
        trace.add(recorder.to_span(status, started_at, _now(), duration_ms))


def traced_step[**P, R](
    name: str, *, features: FeatureExtractor | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that records a step as a span: its first argument is the input, its return value
    the output. ``features(input, output)`` may add step-specific measurements to the span."""

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with span(str(name)) as recorder:
                if recorder is None:
                    return func(*args, **kwargs)
                step_input = args[0] if args else None
                recorder.input = _to_jsonable(step_input)
                result = func(*args, **kwargs)
                recorder.output = _to_jsonable(result)
                if features is not None:
                    recorder.features |= _safe_features(features, name, step_input, result)
                return result

        return wrapper

    return decorate


def _safe_features(
    extractor: FeatureExtractor, name: str, step_input: Any, output: Any
) -> dict[str, FeatureValue]:
    """Feature logging is observability, so a bug in it must never break the pipeline."""
    try:
        return extractor(step_input, output)
    except Exception as exc:
        logger.warning("feature_extraction_failed", span=name, error_type=type(exc).__name__)
        return {"feature_error": type(exc).__name__}
