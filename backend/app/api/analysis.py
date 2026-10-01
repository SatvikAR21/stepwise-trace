"""Root-cause analysis endpoints: diagnose a recorded trace, and read its latest diagnosis."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.analysis.analyzer import analyze_trace
from app.analysis.judge import JudgeFailedError
from app.analysis.models import Analysis
from app.analysis.store import AnalysisStore
from app.api.traces import StoreDep as TraceStoreDep
from app.core.config import Settings, get_settings
from app.llm.base import LLMClient

router = APIRouter(prefix="/traces", tags=["analysis"])


def get_analysis_store(request: Request) -> AnalysisStore:
    """The analysis store created by the application factory."""
    store: AnalysisStore = request.app.state.analysis_store
    return store


def get_judge(request: Request) -> LLMClient:
    """The judge client created by the application factory."""
    judge: LLMClient = request.app.state.judge_llm
    return judge


AnalysisStoreDep = Annotated[AnalysisStore, Depends(get_analysis_store)]
JudgeDep = Annotated[LLMClient, Depends(get_judge)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("/{trace_id}/analysis", response_model=Analysis, status_code=status.HTTP_201_CREATED)
def analyze(
    trace_id: str,
    traces: TraceStoreDep,
    analyses: AnalysisStoreDep,
    judge: JudgeDep,
    settings: SettingsDep,
) -> Analysis:
    """Diagnose the trace's root cause and save the analysis.

    An identical earlier judge request is answered from the saved verdict without a new call.
    Returns 404 for an unknown trace, 503 if the judge's provider fails (for example a used-up
    daily quota) and 502 if the judge's answer stays unusable.
    """
    trace = traces.get(trace_id)
    if trace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"trace {trace_id!r} not found")
    try:
        analysis = analyze_trace(
            trace,
            judge,
            drop_score=settings.judge_drop_score,
            temperature=settings.judge_temperature,
            max_repair_attempts=settings.judge_max_repair_attempts,
            reuse=analyses.verdict_for,
        )
    except JudgeFailedError as exc:
        if exc.provider_error is not None:
            reason = "daily quota used up" if exc.provider_error.quota_exhausted else str(exc)
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"judge unavailable: {reason}"
            ) from exc
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail=f"judge gave no usable verdict: {exc}"
        ) from exc
    analyses.save(analysis)
    return analysis


@router.get("/{trace_id}/analysis", response_model=Analysis)
def latest_analysis(trace_id: str, analyses: AnalysisStoreDep) -> Analysis:
    """The most recent analysis of the trace, or 404."""
    analysis = analyses.latest_for_trace(trace_id)
    if analysis is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"no analysis of trace {trace_id!r} yet"
        )
    return analysis
