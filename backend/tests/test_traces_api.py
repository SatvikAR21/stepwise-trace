"""Tests for the trace list and detail endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

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
def saved(client: TestClient) -> list[Trace]:
    """Three traces in the app's store: n=1 success, n=2 degraded, n=3 success for another doc."""
    base = trace_pipeline(DOC, MockLLMClient(SCRIPTS))[1]
    store: TraceStore = client.app.state.trace_store  # type: ignore[attr-defined]
    traces = [
        base.model_copy(
            update={"trace_id": f"{n:032x}", "started_at": T0 + timedelta(minutes=n), **changes}
        )
        for n, changes in (
            (1, {}),
            (2, {"status": TraceStatus.DEGRADED}),
            (3, {"doc_id": "d2"}),
        )
    ]
    for trace in traces:
        store.save(trace)
    return traces


def test_empty_list(client: TestClient) -> None:
    response = client.get("/traces")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_list_is_newest_first(client: TestClient, saved: list[Trace]) -> None:
    body = client.get("/traces").json()

    assert [item["trace_id"] for item in body["items"]] == [t.trace_id for t in reversed(saved)]
    assert body["total"] == 3
    assert body["items"][0]["status"] == "success"
    assert body["items"][0]["final_score"] == 4


def test_list_filters_and_pages(client: TestClient, saved: list[Trace]) -> None:
    degraded = client.get("/traces", params={"status": "degraded"}).json()
    d1 = client.get("/traces", params={"doc_id": "d1", "limit": 1, "offset": 1}).json()

    assert [i["trace_id"] for i in degraded["items"]] == [saved[1].trace_id]
    assert [i["trace_id"] for i in d1["items"]] == [saved[0].trace_id]
    assert (d1["total"], d1["limit"], d1["offset"]) == (2, 1, 1)


@pytest.mark.parametrize(
    "params", [{"status": "broken"}, {"limit": 0}, {"limit": 201}, {"offset": -1}]
)
def test_invalid_query_parameters_are_rejected(
    client: TestClient, params: dict[str, str | int]
) -> None:
    assert client.get("/traces", params=params).status_code == 422


def test_get_returns_the_full_trace(client: TestClient, saved: list[Trace]) -> None:
    response = client.get(f"/traces/{saved[0].trace_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"] == saved[0].trace_id
    assert [s["name"] for s in body["spans"]] == [
        "intake",
        "extraction",
        "classification",
        "summarization",
    ]
    assert Trace.model_validate(body) == saved[0]


@pytest.mark.parametrize("trace_id", ["0" * 32, "not-a-trace-id"])
def test_get_unknown_trace_is_404(client: TestClient, trace_id: str) -> None:
    response = client.get(f"/traces/{trace_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": f"trace {trace_id!r} not found"}


def test_endpoints_are_documented(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert {"/traces", "/traces/{trace_id}", "/health"} <= set(paths)
