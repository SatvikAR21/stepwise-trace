"""Tests for the rules that give a finished run its status and score."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.llm.base import ChatMessage, Role
from app.pipeline.models import PipelineStatus
from app.tracing.models import FeatureValue, LLMCallRecord, Span, SpanError, SpanStatus, TraceStatus
from app.tracing.status import assess

NOW = datetime(2026, 1, 1, tzinfo=UTC)
CALL = LLMCallRecord(attempt=1, messages=[ChatMessage(role=Role.USER, content="x")])


def _span(
    name: str,
    *,
    confidence: int | None = None,
    calls: int = 1,
    status: SpanStatus = SpanStatus.OK,
    features: dict[str, FeatureValue] | None = None,
    **extra: Any,
) -> Span:
    return Span(
        span_id="0" * 16,
        trace_id="0" * 32,
        name=name,
        status=status,
        started_at=NOW,
        ended_at=NOW,
        duration_ms=1.0,
        llm_calls=[CALL] * calls,
        confidence=confidence,
        features=features or {},
        **extra,
    )


def test_confident_complete_run_is_success_with_lowest_score() -> None:
    spans = [
        _span("intake", calls=0),
        _span("extraction", confidence=5),
        _span("summarization", confidence=3),
    ]

    verdict = assess(spans, PipelineStatus.COMPLETED)

    assert verdict.status is TraceStatus.SUCCESS
    assert verdict.reasons == []
    assert verdict.final_score == 3


def test_low_confidence_makes_the_run_degraded() -> None:
    verdict = assess([_span("classification", confidence=2)], PipelineStatus.COMPLETED)

    assert verdict.status is TraceStatus.DEGRADED
    assert verdict.reasons == ["classification: low confidence (2/5)"]
    assert verdict.final_score == 2


def test_missing_confidence_makes_the_run_degraded() -> None:
    verdict = assess([_span("extraction")], PipelineStatus.COMPLETED)

    assert verdict.status is TraceStatus.DEGRADED
    assert verdict.reasons == ["extraction: confidence missing"]
    assert verdict.final_score is None


def test_invalid_confidence_is_reported_as_invalid() -> None:
    span = _span("extraction", confidence_note="invalid: 'high'")

    verdict = assess([span], PipelineStatus.COMPLETED)

    assert verdict.status is TraceStatus.DEGRADED
    assert verdict.reasons == ["extraction: confidence invalid: 'high'"]


def test_repair_and_truncation_make_the_run_degraded() -> None:
    spans = [
        _span("intake", calls=0, features={"truncated": True}),
        _span("extraction", confidence=5, calls=2),
    ]

    verdict = assess(spans, PipelineStatus.COMPLETED)

    assert verdict.status is TraceStatus.DEGRADED
    assert verdict.reasons == [
        "intake: input truncated",
        "extraction: needed 1 repair attempt(s)",
    ]


def test_failed_pipeline_is_a_failure_and_ignores_missing_confidence_of_failed_span() -> None:
    failed = _span(
        "extraction",
        status=SpanStatus.ERROR,
        error=SpanError(error_type="LLMOutputError", message="bad"),
    )

    verdict = assess([_span("intake", calls=0), failed], PipelineStatus.FAILED)

    assert verdict.status is TraceStatus.FAILURE
    assert verdict.reasons == ["extraction failed: LLMOutputError"]
    assert verdict.final_score is None
