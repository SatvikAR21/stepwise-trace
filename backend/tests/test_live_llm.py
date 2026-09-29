"""Optional smoke test against the real LLM configured in .env.

Skipped by default. Run with:  uv run pytest -m live --no-cov -s
Uses a few requests of your free-tier quota (paced by LLM_MAX_RPM if set).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import LLMProvider, Settings
from app.llm.factory import build_llm_client
from app.pipeline.documents import load_document, load_manifest
from app.pipeline.models import DocumentType, PipelineConfig, PipelineStatus
from app.tracing.service import trace_pipeline

pytestmark = pytest.mark.live


def test_real_llm_runs_simple_invoice(data_dir: Path) -> None:
    settings = Settings()  # reads the repo-root .env (API key, model, base URL)
    if settings.llm_api_key is None:
        pytest.skip("LLM_API_KEY is not set in .env")
    settings = settings.model_copy(update={"llm_provider": LLMProvider.OPENAI})
    entry = load_manifest(data_dir).get("invoice_simple_06")

    result, trace = trace_pipeline(
        load_document(entry, data_dir), build_llm_client(settings), PipelineConfig()
    )

    print(result.model_dump_json(indent=2, exclude_none=True))
    for span in trace.spans:
        print(
            f"{span.name}: confidence={span.confidence} ({span.confidence_note}), "
            f"attempts={len(span.llm_calls)}, features={span.features}"
        )
    print(f"trace status={trace.status.value} score={trace.final_score} {trace.status_reasons}")
    assert result.status is PipelineStatus.COMPLETED, result.error
    assert result.classification is not None
    assert result.classification.document_type is DocumentType.INVOICE
    assert all(s.confidence is not None for s in trace.spans if s.llm_calls), "no confidence"
