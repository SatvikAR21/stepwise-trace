"""Save traces as JSON files and index them in SQLite so they can be searched.

Each trace is written in full to ``<traces_dir>/<trace_id>.json`` (human-readable, easy to diff),
and a small row with the fields we filter and sort by goes into a SQLite table. Nothing is created
on disk until the store is first used.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Engine, String, create_engine, func, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.pipeline.models import PipelineStatus, StepName
from app.tracing.models import TRACE_ID_PATTERN, Trace, TraceStatus

_TRACE_ID_RE = re.compile(TRACE_ID_PATTERN)


class _Base(DeclarativeBase):
    pass


class TraceRow(_Base):
    """The searchable index entry for one trace (the full trace lives in its JSON file)."""

    __tablename__ = "traces"

    trace_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(100), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)  # UTC, stored without tz
    status: Mapped[str] = mapped_column(String(16), index=True)
    final_score: Mapped[int | None]
    pipeline_status: Mapped[str] = mapped_column(String(16))
    failing_step: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(100))
    duration_ms: Mapped[float]


class TraceSummary(BaseModel):
    """One line of a trace listing."""

    trace_id: str
    doc_id: str
    started_at: datetime
    status: TraceStatus
    final_score: int | None
    pipeline_status: PipelineStatus
    failing_step: StepName | None
    model: str
    duration_ms: float


class TracePage(BaseModel):
    """A page of trace summaries, newest first."""

    items: list[TraceSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class TraceStore:
    """JSON files for the full traces, plus a SQLite index."""

    def __init__(self, traces_dir: Path, database_path: Path) -> None:
        self._traces_dir = traces_dir
        self._database_path = database_path
        self._engine: Engine | None = None

    def save(self, trace: Trace) -> Path:
        """Write the trace file and add (or replace) its index row. Returns the file path."""
        engine = self._db()
        path = self._file(trace.trace_id)
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        with Session(engine) as session, session.begin():
            session.merge(_to_row(trace))
        return path

    def get(self, trace_id: str) -> Trace | None:
        """The full trace, or ``None`` if the id is malformed or unknown."""
        if not _TRACE_ID_RE.fullmatch(trace_id):
            return None
        path = self._file(trace_id)
        if not path.is_file():
            return None
        return Trace.model_validate_json(path.read_text(encoding="utf-8"))

    def list_traces(
        self,
        *,
        status: TraceStatus | None = None,
        doc_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> TracePage:
        """Trace summaries, newest first, optionally filtered by status and/or document."""
        query = select(TraceRow)
        if status is not None:
            query = query.where(TraceRow.status == status.value)
        if doc_id is not None:
            query = query.where(TraceRow.doc_id == doc_id)
        with Session(self._db()) as session:
            total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
            rows = session.scalars(
                query.order_by(TraceRow.started_at.desc(), TraceRow.trace_id)
                .limit(limit)
                .offset(offset)
            ).all()
            items = [_to_summary(row) for row in rows]
        return TracePage(items=items, total=total, limit=limit, offset=offset)

    def close(self) -> None:
        """Release the database connections (the store reopens them if used again)."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def _db(self) -> Engine:
        if self._engine is None:
            self._traces_dir.mkdir(parents=True, exist_ok=True)
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            self._engine = create_engine(URL.create("sqlite", database=str(self._database_path)))
            _Base.metadata.create_all(self._engine)
        return self._engine

    def _file(self, trace_id: str) -> Path:
        return self._traces_dir / f"{trace_id}.json"


def _to_row(trace: Trace) -> TraceRow:
    return TraceRow(
        trace_id=trace.trace_id,
        doc_id=trace.doc_id,
        started_at=trace.started_at.astimezone(UTC).replace(tzinfo=None),
        status=trace.status.value,
        final_score=trace.final_score,
        pipeline_status=trace.pipeline_status.value,
        failing_step=trace.failing_step.value if trace.failing_step else None,
        model=trace.model,
        duration_ms=trace.duration_ms,
    )


def _to_summary(row: TraceRow) -> TraceSummary:
    return TraceSummary(
        trace_id=row.trace_id,
        doc_id=row.doc_id,
        started_at=row.started_at.replace(tzinfo=UTC),
        status=TraceStatus(row.status),
        final_score=row.final_score,
        pipeline_status=PipelineStatus(row.pipeline_status),
        failing_step=StepName(row.failing_step) if row.failing_step else None,
        model=row.model,
        duration_ms=row.duration_ms,
    )
