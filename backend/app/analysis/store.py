"""Save root-cause analyses as JSON files with a SQLite index, and find reusable verdicts.

Each analysis is written to ``<analyses_dir>/<analysis_id>.json``; a row in the ``analyses``
table (in the same database as the trace index) records what is filtered and counted on. A new
table is created on first use, so an existing database keeps working.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Engine, String, create_engine, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.analysis.models import ANALYSIS_ID_PATTERN, Analysis, JudgeAnswer

_ANALYSIS_ID_RE = re.compile(ANALYSIS_ID_PATTERN)


class _Base(DeclarativeBase):
    pass


class AnalysisRow(_Base):
    """The index entry for one analysis (the full analysis lives in its JSON file)."""

    __tablename__ = "analyses"

    analysis_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    doc_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)  # UTC, stored without tz
    root_cause_step: Mapped[str | None] = mapped_column(String(32))
    category: Mapped[str | None] = mapped_column(String(40))
    judge_model: Mapped[str] = mapped_column(String(100))
    judge_prompt_version: Mapped[str] = mapped_column(String(20))
    judge_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    reused_verdict: Mapped[bool]
    duration_ms: Mapped[float]


class AnalysisStore:
    """JSON files for the full analyses, plus a SQLite index."""

    def __init__(self, analyses_dir: Path, database_path: Path) -> None:
        self._analyses_dir = analyses_dir
        self._database_path = database_path
        self._engine: Engine | None = None

    def save(self, analysis: Analysis) -> Path:
        """Write the analysis file and its index row. Returns the file path."""
        engine = self._db()
        path = self._file(analysis.analysis_id)
        path.write_text(analysis.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        with Session(engine) as session, session.begin():
            session.merge(_to_row(analysis))
        return path

    def get(self, analysis_id: str) -> Analysis | None:
        """The full analysis, or ``None`` if the id is malformed or unknown."""
        if not _ANALYSIS_ID_RE.fullmatch(analysis_id):
            return None
        path = self._file(analysis_id)
        if not path.is_file():
            return None
        return Analysis.model_validate_json(path.read_text(encoding="utf-8"))

    def latest_for_trace(self, trace_id: str) -> Analysis | None:
        """The most recent analysis of ``trace_id``, or ``None``."""
        query = (
            select(AnalysisRow.analysis_id)
            .where(AnalysisRow.trace_id == trace_id)
            .order_by(AnalysisRow.created_at.desc())
            .limit(1)
        )
        with Session(self._db()) as session:
            analysis_id = session.scalar(query)
        return self.get(analysis_id) if analysis_id else None

    def verdict_for(self, fingerprint: str) -> JudgeAnswer | None:
        """A saved verdict for an identical judge request (same fingerprint), or ``None``.

        Pass this as ``reuse`` to ``analyze_trace`` so an identical question is never asked twice.
        """
        query = (
            select(AnalysisRow.analysis_id)
            .where(AnalysisRow.judge_fingerprint == fingerprint)
            .order_by(AnalysisRow.created_at.desc())
            .limit(1)
        )
        with Session(self._db()) as session:
            analysis_id = session.scalar(query)
        analysis = self.get(analysis_id) if analysis_id else None
        return analysis.verdict if analysis else None

    def close(self) -> None:
        """Release the database connections (the store reopens them if used again)."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def _db(self) -> Engine:
        if self._engine is None:
            self._analyses_dir.mkdir(parents=True, exist_ok=True)
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            self._engine = create_engine(URL.create("sqlite", database=str(self._database_path)))
            _Base.metadata.create_all(self._engine)
        return self._engine

    def _file(self, analysis_id: str) -> Path:
        return self._analyses_dir / f"{analysis_id}.json"


def _to_row(analysis: Analysis) -> AnalysisRow:
    return AnalysisRow(
        analysis_id=analysis.analysis_id,
        trace_id=analysis.trace_id,
        doc_id=analysis.doc_id,
        created_at=analysis.created_at.astimezone(UTC).replace(tzinfo=None),
        root_cause_step=analysis.root_cause_step.value if analysis.root_cause_step else None,
        category=analysis.category.value if analysis.category else None,
        judge_model=analysis.judge_model,
        judge_prompt_version=analysis.judge_prompt_version,
        judge_fingerprint=analysis.judge_fingerprint,
        reused_verdict=analysis.reused_verdict,
        duration_ms=analysis.duration_ms,
    )
