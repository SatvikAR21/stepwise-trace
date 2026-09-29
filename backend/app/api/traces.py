"""Read-only endpoints for recorded traces: list them and fetch one."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.tracing.models import Trace, TraceStatus
from app.tracing.store import TracePage, TraceStore

router = APIRouter(prefix="/traces", tags=["traces"])


def get_trace_store(request: Request) -> TraceStore:
    """The store created by the application factory."""
    store: TraceStore = request.app.state.trace_store
    return store


StoreDep = Annotated[TraceStore, Depends(get_trace_store)]


@router.get("", response_model=TracePage)
def list_traces(
    store: StoreDep,
    trace_status: Annotated[TraceStatus | None, Query(alias="status")] = None,
    doc_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TracePage:
    """Trace summaries, newest first. Filter with ``?status=degraded`` and/or ``?doc_id=...``."""
    return store.list_traces(status=trace_status, doc_id=doc_id, limit=limit, offset=offset)


@router.get("/{trace_id}", response_model=Trace)
def get_trace(trace_id: str, store: StoreDep) -> Trace:
    """One full trace with all its spans, or 404."""
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"trace {trace_id!r} not found")
    return trace
