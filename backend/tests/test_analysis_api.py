"""Tests for the analysis endpoints (pretend judge, temporary storage)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.analysis.judge import JUDGE_STEP
from app.core.config import BACKEND_DIR
from app.llm.base import LLMClient, LLMProviderError, LLMRequest, LLMResponse
from app.llm.mock import MockLLMClient
from app.pipeline.documents import load_document, load_manifest
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline
from app.tracing.store import TraceStore

DATA_DIR = BACKEND_DIR / "data"
DOC_ID = "correspondence_unnamed_ceo_19"


@pytest.fixture
def saved_trace(client: TestClient) -> Trace:
    entry = load_manifest(DATA_DIR).get(DOC_ID)
    llm = MockLLMClient(scripts_dir=DATA_DIR / "mock_responses")
    trace = trace_pipeline(load_document(entry, DATA_DIR), llm)[1]
    store: TraceStore = client.app.state.trace_store  # type: ignore[attr-defined]
    store.save(trace)
    return trace


def test_post_diagnoses_the_trace_and_saves_the_analysis(
    client: TestClient, saved_trace: Trace
) -> None:
    response = client.post(f"/traces/{saved_trace.trace_id}/analysis")

    assert response.status_code == 201
    body = response.json()
    assert body["trace_id"] == saved_trace.trace_id
    assert body["root_cause_step"] == "extraction"
    assert body["category"] == "extraction_hallucination"
    assert body["summary"].startswith("Root cause: Step 2 (Extraction), Extraction Hallucination.")
    assert [s["role"] for s in body["steps"]] == ["healthy", "root_cause", "healthy", "propagated"]
    assert {c["name"] for c in body["checks"] if c["flagged"]} == {"ungrounded_entities"}

    latest = client.get(f"/traces/{saved_trace.trace_id}/analysis")
    assert latest.status_code == 200
    assert latest.json()["analysis_id"] == body["analysis_id"]


def test_a_second_post_reuses_the_saved_verdict(client: TestClient, saved_trace: Trace) -> None:
    first = client.post(f"/traces/{saved_trace.trace_id}/analysis").json()
    second = client.post(f"/traces/{saved_trace.trace_id}/analysis").json()

    assert not first["reused_verdict"] and len(first["judge_calls"]) == 1
    assert second["reused_verdict"] and second["judge_calls"] == []
    assert second["analysis_id"] != first["analysis_id"]


def test_unknown_trace_gives_404(client: TestClient) -> None:
    assert client.post(f"/traces/{'0' * 32}/analysis").status_code == 404
    assert client.get(f"/traces/{'0' * 32}/analysis").status_code == 404


def test_trace_without_analysis_gives_404_on_get(client: TestClient, saved_trace: Trace) -> None:
    response = client.get(f"/traces/{saved_trace.trace_id}/analysis")

    assert response.status_code == 404
    assert "no analysis" in response.json()["detail"]


def test_an_unusable_judge_answer_gives_502(client: TestClient, saved_trace: Trace) -> None:
    client.app.state.judge_llm = MockLLMClient({(DOC_ID, JUDGE_STEP): "no json here"})  # type: ignore[attr-defined]

    response = client.post(f"/traces/{saved_trace.trace_id}/analysis")

    assert response.status_code == 502
    assert "judge gave no usable verdict" in response.json()["detail"]


class _QuotaUsedUp(LLMClient):
    @property
    def model_name(self) -> str:
        return "gemini-test"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError("RateLimitError: per day", status_code=429, quota_exhausted=True)


def test_a_used_up_daily_quota_gives_503(client: TestClient, saved_trace: Trace) -> None:
    client.app.state.judge_llm = _QuotaUsedUp()  # type: ignore[attr-defined]

    response = client.post(f"/traces/{saved_trace.trace_id}/analysis")

    assert response.status_code == 503
    assert response.json()["detail"] == "judge unavailable: daily quota used up"
    assert client.get(f"/traces/{saved_trace.trace_id}/analysis").status_code == 404
