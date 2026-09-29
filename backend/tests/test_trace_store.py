"""Tests for saving traces as JSON files with a SQLite index."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.llm.mock import MockLLMClient
from app.pipeline.models import RawDocument
from app.tracing.models import Trace, TraceStatus
from app.tracing.service import trace_pipeline
from app.tracing.store import TraceStore

DOC = RawDocument(doc_id="d1", content="Invoice 42 from Acme. Total due $10.00.")
SCRIPTS: dict[tuple[str, str], Any] = {
    ("d1", "extraction"): {"organizations": [{"name": "Acme"}], "confidence": 5},
    ("d1", "classification"): {"document_type": "invoice", "rationale": "bill", "confidence": 5},
    ("d1", "summarization"): {"headline": "Acme bill", "confidence": 4},
}
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[TraceStore]:
    trace_store = TraceStore(tmp_path / "traces", tmp_path / "db" / "index.db")
    yield trace_store
    trace_store.close()


@pytest.fixture(scope="module")
def base_trace() -> Trace:
    return trace_pipeline(DOC, MockLLMClient(SCRIPTS))[1]


def _variant(base: Trace, n: int, **changes: Any) -> Trace:
    return base.model_copy(
        update={"trace_id": f"{n:032x}", "started_at": T0 + timedelta(minutes=n), **changes}
    )


def test_nothing_is_created_until_first_use(tmp_path: Path) -> None:
    TraceStore(tmp_path / "traces", tmp_path / "index.db")

    assert list(tmp_path.iterdir()) == []


def test_save_writes_a_readable_json_file_and_get_reads_it_back(
    store: TraceStore, base_trace: Trace, tmp_path: Path
) -> None:
    path = store.save(base_trace)

    assert path == tmp_path / "traces" / f"{base_trace.trace_id}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["trace_id"] == base_trace.trace_id
    assert (tmp_path / "db" / "index.db").is_file()
    assert store.get(base_trace.trace_id) == base_trace


@pytest.mark.parametrize("trace_id", ["0" * 32, "not-a-trace-id", "../../etc/passwd"])
def test_get_unknown_or_malformed_id_returns_none(store: TraceStore, trace_id: str) -> None:
    assert store.get(trace_id) is None


def test_list_is_newest_first_with_summary_fields(store: TraceStore, base_trace: Trace) -> None:
    for n in (1, 3, 2):
        store.save(_variant(base_trace, n))

    page = store.list_traces()

    assert [item.trace_id for item in page.items] == [f"{n:032x}" for n in (3, 2, 1)]
    assert page.total == 3
    first = page.items[0]
    assert first.started_at == T0 + timedelta(minutes=3)
    assert first.doc_id == "d1"
    assert first.status is TraceStatus.SUCCESS
    assert first.final_score == 4
    assert first.model == "mock-llm"
    assert first.failing_step is None


def test_list_filters_by_status_and_document(store: TraceStore, base_trace: Trace) -> None:
    store.save(_variant(base_trace, 1))
    store.save(_variant(base_trace, 2, status=TraceStatus.DEGRADED))
    store.save(_variant(base_trace, 3, status=TraceStatus.DEGRADED, doc_id="d2"))

    degraded = store.list_traces(status=TraceStatus.DEGRADED)
    d1_degraded = store.list_traces(status=TraceStatus.DEGRADED, doc_id="d1")

    assert [i.trace_id for i in degraded.items] == [f"{n:032x}" for n in (3, 2)]
    assert [i.trace_id for i in d1_degraded.items] == [f"{2:032x}"]


def test_list_pages_through_results(store: TraceStore, base_trace: Trace) -> None:
    for n in range(1, 6):
        store.save(_variant(base_trace, n))

    page = store.list_traces(limit=2, offset=2)

    assert [i.trace_id for i in page.items] == [f"{n:032x}" for n in (3, 2)]
    assert (page.total, page.limit, page.offset) == (5, 2, 2)


def test_saving_the_same_trace_twice_keeps_one_index_row(
    store: TraceStore, base_trace: Trace
) -> None:
    store.save(base_trace)
    store.save(base_trace.model_copy(update={"status": TraceStatus.DEGRADED}))

    page = store.list_traces()

    assert page.total == 1
    assert page.items[0].status is TraceStatus.DEGRADED


def test_store_can_be_used_again_after_close(store: TraceStore, base_trace: Trace) -> None:
    store.save(base_trace)
    store.close()

    assert store.list_traces().total == 1
