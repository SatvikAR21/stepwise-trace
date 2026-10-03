"""Grade the root-cause analyzer against the corpus answer key.

Each document runs through the scripted (mock) pipeline, where the planted failures live, and is
then diagnosed by the configured judge. A diagnosis is compared with the manifest's
``failing_step`` and ``intended_failure``; healthy documents must get no root cause.

Real judges are paid for in requests, often from a small daily quota, so: a hard ``max_calls``
limit covers every request including repairs, saved verdicts for identical requests are reused
for free, and a provider failure (such as a used-up daily quota) stops the run at once.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from app.analysis.analyzer import analyze_trace
from app.analysis.judge import JudgeFailedError, judge_messages, request_fingerprint
from app.analysis.models import Analysis
from app.analysis.store import AnalysisStore
from app.analysis.taxonomy import FailureCategory
from app.llm.base import LLMClient
from app.llm.mock import MOCK_MODEL_NAME
from app.llm.prompts import TRACE_JUDGE_PROMPT
from app.pipeline.documents import CorpusSplit, ManifestEntry, load_document
from app.pipeline.models import PipelineConfig, StepName
from app.tracing.models import LLMCallRecord
from app.tracing.service import trace_pipeline
from app.tracing.store import TraceStore


class Outcome(StrEnum):
    """How one diagnosis compares with the answer key."""

    CORRECT = "correct"  # right step and right category
    STEP_ONLY = "step-only"  # right step, wrong category
    WRONG = "wrong"  # blamed another step
    MISSED = "missed"  # a planted failure, but no root cause named
    HEALTHY_OK = "healthy-ok"  # a healthy document with no root cause named
    FALSE_ALARM = "false-alarm"  # a healthy document that was blamed anyway
    NOT_JUDGED = "not-judged"  # call limit, provider failure, unusable answer, or run stopped


class StopReason(StrEnum):
    """Why an evaluation ended before judging every document."""

    QUOTA_EXHAUSTED = "quota-exhausted"  # a daily quota is used up: try again after its reset
    PROVIDER_ERROR = "provider-error"  # any other provider failure, e.g. an overloaded model


class DocResult(BaseModel):
    """One document's diagnosis next to its answer key."""

    doc_id: str
    split: CorpusSplit
    planted_step: StepName | None
    planted_category: FailureCategory | None
    diagnosed_step: StepName | None = None
    diagnosed_category: FailureCategory | None = None
    outcome: Outcome
    analysis_id: str | None = None
    trace_id: str | None = None
    judge_calls: int = 0
    reused_verdict: bool = False
    flagged_checks: list[str] = Field(default_factory=list)
    checks_point_at_planted_step: bool = False
    note: str | None = None


class Totals(BaseModel):
    """The report card's numbers."""

    broken: int
    broken_judged: int
    right_step: int
    right_category: int
    correct: int
    missed: int
    healthy: int
    healthy_judged: int
    false_alarms: int
    not_judged: int
    checks_caught: int = Field(description="Broken documents flagged by at least one check")
    checks_right_step: int = Field(description="... flagged by a check pointing at the right step")
    checks_false_alarms: int = Field(description="Healthy documents flagged by at least one check")
    judge_calls: int
    reused_verdicts: int
    prompt_tokens: int
    completion_tokens: int
    judge_latency_ms: float


class EvaluationReport(BaseModel):
    """The full result of one evaluation run."""

    started_at: datetime
    finished_at: datetime
    splits: list[CorpusSplit]
    judge_model: str
    judge_prompt_version: str
    scripted_judge: bool = Field(description="True for the mock judge: not a measurement")
    drop_score: int
    max_calls: int | None
    stopped: StopReason | None = None
    stop_detail: str | None = None
    rows: list[DocResult]
    totals: Totals


def evaluate_corpus(
    entries: list[ManifestEntry],
    *,
    data_dir: Path,
    pipeline_llm: LLMClient,
    judge: LLMClient,
    trace_store: TraceStore,
    analysis_store: AnalysisStore,
    config: PipelineConfig,
    drop_score: int,
    temperature: float = 0.0,
    max_repair_attempts: int = 1,
    max_calls: int | None = None,
    on_result: Callable[[DocResult], None] | None = None,
) -> EvaluationReport:
    """Run, diagnose and grade ``entries``. ``max_calls`` (None = no limit) caps judge requests."""
    started = datetime.now(UTC)
    rows: list[DocResult] = []
    calls: list[LLMCallRecord] = []
    stopped: StopReason | None = None
    stop_detail: str | None = None
    for entry in entries:
        _, trace = trace_pipeline(load_document(entry, data_dir), pipeline_llm, config)
        trace_store.save(trace)
        reusable = analysis_store.verdict_for(
            request_fingerprint(judge_messages(trace), judge.model_name)
        )
        remaining = None if max_calls is None else max_calls - len(calls)
        if reusable is None and remaining is not None and remaining < 1:
            row = _not_judged(entry, trace.trace_id, "call limit reached")
        else:
            # Repairs may not push the run past the limit either.
            repairs = max_repair_attempts if remaining is None else remaining - 1
            try:
                analysis = analyze_trace(
                    trace,
                    judge,
                    drop_score=drop_score,
                    temperature=temperature,
                    max_repair_attempts=max(min(repairs, max_repair_attempts), 0),
                    reuse=analysis_store.verdict_for,
                )
            except JudgeFailedError as exc:
                calls.extend(exc.calls)
                if exc.provider_error is not None:
                    quota = exc.provider_error.quota_exhausted
                    stopped = StopReason.QUOTA_EXHAUSTED if quota else StopReason.PROVIDER_ERROR
                    stop_detail = "daily quota used up" if quota else str(exc)
                    note = stop_detail
                else:
                    note = f"unusable answer: {exc}"
                row = _not_judged(entry, trace.trace_id, note, calls=len(exc.calls))
            else:
                analysis_store.save(analysis)
                calls.extend(analysis.judge_calls)
                row = grade(entry, analysis)
        rows.append(row)
        if on_result is not None:
            on_result(row)
        if stopped is not None:
            break
    for entry in entries[len(rows) :]:
        row = _not_judged(entry, None, "not reached (run stopped)")
        rows.append(row)
        if on_result is not None:
            on_result(row)
    return EvaluationReport(
        started_at=started,
        finished_at=datetime.now(UTC),
        splits=[split for split in CorpusSplit if any(e.split is split for e in entries)],
        judge_model=judge.model_name,
        judge_prompt_version=TRACE_JUDGE_PROMPT.version,
        scripted_judge=judge.model_name == MOCK_MODEL_NAME,
        drop_score=drop_score,
        max_calls=max_calls,
        stopped=stopped,
        stop_detail=stop_detail,
        rows=rows,
        totals=totals(rows, calls),
    )


def grade(entry: ManifestEntry, analysis: Analysis) -> DocResult:
    """Compare one diagnosis with the document's answer key."""
    planted_step = StepName(entry.failing_step) if entry.failing_step else None
    planted_category = FailureCategory(entry.intended_failure) if entry.intended_failure else None
    found_step, found_category = analysis.root_cause_step, analysis.category
    if planted_step is None:
        outcome = Outcome.HEALTHY_OK if found_step is None else Outcome.FALSE_ALARM
    elif found_step is None:
        outcome = Outcome.MISSED
    elif found_step is not planted_step:
        outcome = Outcome.WRONG
    else:
        outcome = Outcome.CORRECT if found_category is planted_category else Outcome.STEP_ONLY
    flagged = [check for check in analysis.checks if check.flagged]
    return DocResult(
        doc_id=entry.doc_id,
        split=entry.split,
        planted_step=planted_step,
        planted_category=planted_category,
        diagnosed_step=found_step,
        diagnosed_category=found_category,
        outcome=outcome,
        analysis_id=analysis.analysis_id,
        trace_id=analysis.trace_id,
        judge_calls=len(analysis.judge_calls),
        reused_verdict=analysis.reused_verdict,
        flagged_checks=[check.name for check in flagged],
        checks_point_at_planted_step=planted_step is not None
        and any(check.step is planted_step for check in flagged),
    )


def totals(rows: list[DocResult], calls: list[LLMCallRecord]) -> Totals:
    """Count the outcomes and the cost of a list of results."""
    broken = [r for r in rows if r.planted_step is not None]
    healthy = [r for r in rows if r.planted_step is None]
    judged_broken = [r for r in broken if r.outcome is not Outcome.NOT_JUDGED]
    judged_healthy = [r for r in healthy if r.outcome is not Outcome.NOT_JUDGED]
    return Totals(
        broken=len(broken),
        broken_judged=len(judged_broken),
        right_step=sum(r.diagnosed_step is r.planted_step for r in judged_broken),
        right_category=sum(r.diagnosed_category is r.planted_category for r in judged_broken),
        correct=sum(r.outcome is Outcome.CORRECT for r in judged_broken),
        missed=sum(r.outcome is Outcome.MISSED for r in judged_broken),
        healthy=len(healthy),
        healthy_judged=len(judged_healthy),
        false_alarms=sum(r.outcome is Outcome.FALSE_ALARM for r in judged_healthy),
        not_judged=sum(r.outcome is Outcome.NOT_JUDGED for r in rows),
        checks_caught=sum(bool(r.flagged_checks) for r in judged_broken),
        checks_right_step=sum(r.checks_point_at_planted_step for r in judged_broken),
        checks_false_alarms=sum(bool(r.flagged_checks) for r in judged_healthy),
        judge_calls=len(calls),
        reused_verdicts=sum(r.reused_verdict for r in rows),
        prompt_tokens=sum(c.prompt_tokens for c in calls),
        completion_tokens=sum(c.completion_tokens for c in calls),
        judge_latency_ms=round(sum(c.latency_ms for c in calls), 1),
    )


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% range for a proportion measured on few cases (Wilson score interval)."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def save_report(report: EvaluationReport, directory: Path) -> Path:
    """Write ``report`` to ``<directory>/<timestamp>-<splits>.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{report.started_at:%Y%m%dT%H%M%SZ}-{'-'.join(s.value for s in report.splits)}.json"
    path = directory / name
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _not_judged(
    entry: ManifestEntry, trace_id: str | None, note: str, *, calls: int = 0
) -> DocResult:
    return DocResult(
        doc_id=entry.doc_id,
        split=entry.split,
        planted_step=StepName(entry.failing_step) if entry.failing_step else None,
        planted_category=(
            FailureCategory(entry.intended_failure) if entry.intended_failure else None
        ),
        outcome=Outcome.NOT_JUDGED,
        trace_id=trace_id,
        judge_calls=calls,
        note=note,
    )
