"""Backward root-cause analysis of one trace.

1. The judge grades every step that ran, in one call (see ``judge.py``).
2. The walk goes backward from the final output towards the document. Every step that introduced
   a problem and scored at or below the drop line is a suspect; the earliest suspect is the root
   cause. Every step is visited: a summary that faithfully repeats an invented date looks fine on
   its own, so stopping at the first healthy step would let the real culprit go free.
3. Later steps that carried the problem forward form the evidence chain.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.analysis.judge import judge_messages, judge_trace, request_fingerprint
from app.analysis.models import Analysis, JudgeAnswer, StepFinding, StepGrade, StepRole
from app.analysis.taxonomy import definition_of
from app.llm.base import LLMClient
from app.llm.prompts import TRACE_JUDGE_PROMPT
from app.pipeline.models import StepName
from app.tracing.models import Span, SpanStatus, Trace

DEFAULT_DROP_SCORE = 2
EXCERPT_CHARS = 600
_ORDER = {step: index for index, step in enumerate(StepName)}

ReuseVerdict = Callable[[str], JudgeAnswer | None]


def analyze_trace(
    trace: Trace,
    judge: LLMClient,
    *,
    drop_score: int = DEFAULT_DROP_SCORE,
    temperature: float = 0.0,
    max_repair_attempts: int = 1,
    reuse: ReuseVerdict | None = None,
) -> Analysis:
    """Diagnose ``trace``. ``reuse`` can return an earlier verdict for an identical request (by
    fingerprint), in which case the judge is not called. Raises ``JudgeFailedError``."""
    started = time.perf_counter()
    messages = judge_messages(trace)
    fingerprint = request_fingerprint(messages, judge.model_name)
    verdict = reuse(fingerprint) if reuse else None
    calls = []
    if verdict is None:
        result = judge_trace(
            trace,
            judge,
            messages=messages,
            temperature=temperature,
            max_repair_attempts=max_repair_attempts,
        )
        verdict, calls = result.answer, result.calls
    root, findings = walk_backward(trace, verdict, drop_score)
    return Analysis(
        analysis_id=secrets.token_hex(16),
        trace_id=trace.trace_id,
        doc_id=trace.doc_id,
        created_at=datetime.now(UTC),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        judge_model=judge.model_name,
        judge_prompt_version=TRACE_JUDGE_PROMPT.version,
        judge_fingerprint=fingerprint,
        reused_verdict=not calls,
        drop_score=drop_score,
        root_cause_step=root.step if root else None,
        category=root.category if root else None,
        summary=summarize(root, findings, drop_score),
        steps=findings,
        verdict=verdict,
        judge_calls=calls,
    )


def walk_backward(
    trace: Trace, verdict: JudgeAnswer, drop_score: int
) -> tuple[StepGrade | None, list[StepFinding]]:
    """Find the root cause (or ``None``) and describe every step's part in the failure."""
    root: StepGrade | None = None
    for grade in reversed(verdict.steps):  # from the final output back towards the document
        if _significant(grade, drop_score):
            root = grade  # an earlier culprit replaces a later one
    grades = {grade.step: grade for grade in verdict.steps}
    spans = {StepName(span.name): span for span in trace.spans}
    findings = [
        _finding(step, grades.get(step), spans.get(step), root, drop_score) for step in StepName
    ]
    return root, findings


def summarize(root: StepGrade | None, findings: list[StepFinding], drop_score: int) -> str:
    """The evidence chain in words."""
    if root is None:
        text = (
            f"No step showed a significant quality drop (every step scored above {drop_score}/5)."
        )
        minor = [f for f in findings if f.role is StepRole.MINOR]
        if minor:
            text += " Minor issues: " + "; ".join(_issue(f, f.introduced) for f in minor) + "."
        return text
    category = definition_of(root.category).label if root.category else "uncategorized"
    text = f"Root cause: {step_label(root.step)}, {category}. {_sentence(root.explanation)}"
    propagated = [f for f in findings if f.role is StepRole.PROPAGATED]
    if propagated:
        text += " It propagated to " + "; ".join(_issue(f, f.inherited) for f in propagated) + "."
    secondary = [f for f in findings if f.role is StepRole.SECONDARY]
    if secondary:
        text += " Also failed: " + "; ".join(_issue(f, f.introduced) for f in secondary) + "."
    return text


def step_label(step: StepName) -> str:
    """'Step 2 (Extraction)'."""
    return f"Step {_ORDER[step] + 1} ({step.value.capitalize()})"


def _significant(grade: StepGrade, drop_score: int) -> bool:
    return bool(grade.introduced) and grade.score <= drop_score


def _role(grade: StepGrade, root: StepGrade | None, drop_score: int) -> StepRole:
    if root is not None and grade.step is root.step:
        return StepRole.ROOT_CAUSE
    after_root = root is not None and _ORDER[grade.step] > _ORDER[root.step]
    if after_root and _significant(grade, drop_score):
        return StepRole.SECONDARY
    if after_root and grade.inherited:
        return StepRole.PROPAGATED
    if grade.introduced:
        return StepRole.MINOR
    return StepRole.HEALTHY


def _finding(
    step: StepName,
    grade: StepGrade | None,
    span: Span | None,
    root: StepGrade | None,
    drop_score: int,
) -> StepFinding:
    if grade is None or span is None:
        return StepFinding(step=step, role=StepRole.NOT_RUN)
    return StepFinding(
        step=step,
        role=_role(grade, root, drop_score),
        score=grade.score,
        introduced=grade.introduced,
        inherited=grade.inherited,
        category=grade.category,
        explanation=grade.explanation,
        evidence=grade.evidence,
        received=_excerpt(_received(step, span)),
        produced=_excerpt(_produced(step, span)),
    )


def _received(step: StepName, span: Span) -> Any:
    data: dict[str, Any] = span.input or {}
    match step:
        case StepName.INTAKE:
            return data.get("content")
        case StepName.EXTRACTION:
            return data.get("text")
        case _:  # the document text is shown under intake; list only what earlier steps added
            return {k: v for k, v in data.items() if k != "document"}


def _produced(step: StepName, span: Span) -> Any:
    if span.status is SpanStatus.ERROR:
        reason = f"{span.error.error_type}: {span.error.message}" if span.error else "error"
        answers = [c.raw_response for c in span.llm_calls if c.raw_response]
        return f"FAILED ({reason})" + (f"; last answer: {answers[-1]}" if answers else "")
    if step is StepName.INTAKE:
        return (span.output or {}).get("text")
    return span.output


def _excerpt(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= EXCERPT_CHARS else text[: EXCERPT_CHARS - 1] + "…"


def _issue(finding: StepFinding, problems: list[str]) -> str:
    return f"{step_label(finding.step)}: {problems[0]}" if problems else step_label(finding.step)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."
