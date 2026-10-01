"""Optional smoke tests against the real LLM configured in .env.

Skipped by default. Run with:  uv run pytest -m live --no-cov -s
Uses requests of your free-tier quota: 3 for the pipeline test (paced by LLM_MAX_RPM if set) and
1 (2 if a repair is needed) for the judge test (paced by JUDGE_MAX_RPM).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.analyzer import analyze_trace
from app.core.config import LLMProvider, Settings
from app.llm.factory import MOCK_SCRIPTS_SUBDIR, build_judge_client, build_llm_client
from app.llm.mock import MockLLMClient
from app.pipeline.documents import load_document, load_manifest
from app.pipeline.models import DocumentType, PipelineConfig, PipelineStatus, StepName
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


def test_real_judge_grades_a_planted_failure(data_dir: Path) -> None:
    settings = Settings()  # reads the repo-root .env (key, base URL, JUDGE_* settings)
    if (settings.judge_api_key or settings.llm_api_key) is None:
        pytest.skip("neither JUDGE_API_KEY nor LLM_API_KEY is set in .env")
    settings = settings.model_copy(update={"judge_provider": LLMProvider.OPENAI})
    entry = load_manifest(data_dir).get("contract_no_dates_04")
    pipeline = MockLLMClient(scripts_dir=data_dir / MOCK_SCRIPTS_SUBDIR)
    _, trace = trace_pipeline(load_document(entry, data_dir), pipeline)

    analysis = analyze_trace(
        trace,
        build_judge_client(settings),
        drop_score=settings.judge_drop_score,
        max_repair_attempts=settings.judge_max_repair_attempts,
    )

    print(analysis.summary)
    for finding in analysis.steps:
        print(
            f"{finding.step.value}: {finding.role.value} score={finding.score} {finding.introduced}"
        )
    assert [grade.step for grade in analysis.verdict.steps] == list(StepName)
    assert analysis.judge_calls and analysis.judge_model == settings.judge_model
