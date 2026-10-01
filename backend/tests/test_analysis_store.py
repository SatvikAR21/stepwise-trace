"""Tests for saving analyses and reusing verdicts."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from app.analysis.analyzer import analyze_trace
from app.analysis.models import Analysis
from app.analysis.store import AnalysisStore
from app.core.config import BACKEND_DIR, Settings
from app.llm.factory import build_judge_client
from app.llm.mock import MockLLMClient
from app.pipeline.documents import load_document, load_manifest
from app.pipeline.models import StepName
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline
from app.tracing.store import TraceStore

DATA_DIR = BACKEND_DIR / "data"


def _trace(doc_id: str) -> Trace:
    entry = load_manifest(DATA_DIR).get(doc_id)
    llm = MockLLMClient(scripts_dir=DATA_DIR / "mock_responses")
    return trace_pipeline(load_document(entry, DATA_DIR), llm)[1]


def _analyze(trace: Trace, store: AnalysisStore | None = None) -> Analysis:
    judge = build_judge_client(Settings(_env_file=None, data_dir=DATA_DIR))
    return analyze_trace(trace, judge, reuse=store.verdict_for if store else None)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[AnalysisStore]:
    analysis_store = AnalysisStore(tmp_path / "analyses", tmp_path / "test.db")
    yield analysis_store
    analysis_store.close()


def test_saved_analysis_reads_back_identically(store: AnalysisStore) -> None:
    analysis = _analyze(_trace("contract_no_dates_04"))

    path = store.save(analysis)

    assert path.name == f"{analysis.analysis_id}.json"
    assert path.read_bytes().endswith(b"}\n") and b"\r\n" not in path.read_bytes()
    assert store.get(analysis.analysis_id) == analysis


def test_latest_for_trace_returns_the_newest_analysis(store: AnalysisStore) -> None:
    trace = _trace("contract_no_dates_04")
    first = _analyze(trace)
    second = _analyze(trace).model_copy(update={"created_at": first.created_at.replace(year=2030)})
    store.save(second)
    store.save(first)

    latest = store.latest_for_trace(trace.trace_id)

    assert latest is not None and latest.analysis_id == second.analysis_id
    assert store.latest_for_trace("0" * 32) is None


def test_an_identical_request_reuses_the_saved_verdict(store: AnalysisStore) -> None:
    store.save(_analyze(_trace("invoice_multi_currency_09")))

    again = _analyze(_trace("invoice_multi_currency_09"), store)  # a new run, same content

    assert again.reused_verdict and again.judge_calls == []
    assert again.root_cause_step is StepName.SUMMARIZATION


def test_a_different_request_is_not_reused(store: AnalysisStore) -> None:
    store.save(_analyze(_trace("invoice_multi_currency_09")))

    other = _analyze(_trace("invoice_simple_06"), store)

    assert not other.reused_verdict and len(other.judge_calls) == 1


def test_unknown_or_malformed_ids_return_none(store: AnalysisStore) -> None:
    assert store.get("f" * 32) is None
    assert store.get("../../secrets") is None
    assert store.verdict_for("no-such-fingerprint") is None


def test_analyses_share_the_trace_database_without_disturbing_it(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    traces = TraceStore(tmp_path / "traces", db)
    analyses = AnalysisStore(tmp_path / "traces" / "analyses", db)
    trace = _trace("contract_no_dates_04")
    try:
        traces.save(trace)
        analyses.save(_analyze(trace))
        assert traces.list_traces().total == 1
    finally:
        traces.close()
        analyses.close()

    with closing(sqlite3.connect(db)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert {"traces", "analyses"} <= tables


def test_nothing_is_written_until_first_use(tmp_path: Path) -> None:
    AnalysisStore(tmp_path / "analyses", tmp_path / "x.db").close()

    assert list(tmp_path.iterdir()) == []
