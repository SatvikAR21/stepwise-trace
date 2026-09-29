"""Rules that turn a finished run's spans into a status and a score.

- failure:  a step raised an error, so the pipeline stopped.
- degraded: the run finished but something is suspicious: a step's confidence is low (<= 2),
            missing or invalid, an LLM answer needed a repair attempt, or the input was truncated.
- success:  none of the above.

The final score is the lowest confidence of any step (the weakest link). A confidently wrong run
still counts as "success" here; catching those is the job of the root-cause analyzer.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.pipeline.models import PipelineStatus
from app.tracing.models import Span, SpanStatus, TraceStatus

LOW_CONFIDENCE = 2
TRUNCATED_FEATURE = "truncated"


class Assessment(BaseModel):
    """The verdict for one run."""

    status: TraceStatus
    reasons: list[str]
    final_score: int | None


def assess(spans: list[Span], pipeline_status: PipelineStatus) -> Assessment:
    """Apply the status rules to the spans of one run."""
    reasons: list[str] = []
    for span in spans:
        if span.status is SpanStatus.ERROR and span.error is not None:
            reasons.append(f"{span.name} failed: {span.error.error_type}")
        if span.features.get(TRUNCATED_FEATURE) is True:
            reasons.append(f"{span.name}: input truncated")
        if not span.llm_calls:
            continue
        if len(span.llm_calls) > 1:
            reasons.append(f"{span.name}: needed {len(span.llm_calls) - 1} repair attempt(s)")
        if span.status is SpanStatus.OK:
            if span.confidence is None:
                reasons.append(f"{span.name}: confidence {span.confidence_note or 'missing'}")
            elif span.confidence <= LOW_CONFIDENCE:
                reasons.append(f"{span.name}: low confidence ({span.confidence}/5)")

    scores = [s.confidence for s in spans if s.confidence is not None]
    if pipeline_status is PipelineStatus.FAILED:
        status = TraceStatus.FAILURE
    elif reasons:
        status = TraceStatus.DEGRADED
    else:
        status = TraceStatus.SUCCESS
    return Assessment(status=status, reasons=reasons, final_score=min(scores) if scores else None)
