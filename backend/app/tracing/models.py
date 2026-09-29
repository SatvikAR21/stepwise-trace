"""Trace and span records: what one pipeline run, and every step inside it, looked like.

IDs follow the OpenTelemetry format (32 hex characters for a trace, 16 for a span), so traces could
later be exported to standard observability tools without changing their identity.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.llm.base import ChatMessage
from app.pipeline.models import PipelineConfig, PipelineStatus, StepName

TRACE_ID_PATTERN = r"^[0-9a-f]{32}$"
SPAN_ID_PATTERN = r"^[0-9a-f]{16}$"

FeatureValue = str | int | float | bool | None


class TraceStatus(StrEnum):
    """Overall quality verdict for a run, before any deeper analysis."""

    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILURE = "failure"


class SpanStatus(StrEnum):
    """Whether the step finished or raised an error."""

    OK = "ok"
    ERROR = "error"


class LLMCallRecord(BaseModel):
    """One request to the LLM inside a step: the first try or a repair attempt."""

    attempt: int = Field(ge=1)
    messages: list[ChatMessage]
    raw_response: str | None = None
    model: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    wait_ms: float = Field(default=0.0, ge=0.0, description="Time spent waiting for a rate limit")
    error: str | None = Field(default=None, description="Why this attempt failed, if it did")


class SpanError(BaseModel):
    """The exception that stopped a step."""

    error_type: str
    message: str


class Span(BaseModel):
    """The record of one step: what it received, what it produced, and how the LLM behaved."""

    span_id: str = Field(pattern=SPAN_ID_PATTERN)
    trace_id: str = Field(pattern=TRACE_ID_PATTERN)
    parent_span_id: str | None = Field(default=None, pattern=SPAN_ID_PATTERN)
    name: str
    status: SpanStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: float = Field(ge=0.0)
    input: Any = None
    output: Any = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    repair_prompt_version: str | None = Field(
        default=None, description="Version of the repair prompt, if an answer had to be repaired"
    )
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    confidence: int | None = Field(default=None, ge=1, le=5)
    confidence_note: str | None = Field(
        default=None, description="Why the confidence is missing or was rejected"
    )
    error: SpanError | None = None
    features: dict[str, FeatureValue] = Field(default_factory=dict)


class Trace(BaseModel):
    """The record of one complete pipeline run."""

    trace_id: str = Field(pattern=TRACE_ID_PATTERN)
    doc_id: str
    model: str
    config: PipelineConfig | None = Field(
        default=None, description="The settings the run used, so it can be repeated exactly"
    )
    started_at: datetime
    ended_at: datetime
    duration_ms: float = Field(ge=0.0)
    status: TraceStatus
    status_reasons: list[str] = Field(default_factory=list)
    final_score: int | None = Field(
        default=None, ge=1, le=5, description="Lowest step confidence: the weakest link"
    )
    pipeline_status: PipelineStatus
    error_step: StepName | None = Field(
        default=None,
        description="The step that raised an error and stopped the run (not the root cause)",
    )
    spans: list[Span] = Field(default_factory=list)
    final_output: Any = None
