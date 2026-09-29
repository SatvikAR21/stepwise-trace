"""Tests for chaining the steps and recording failures."""

from __future__ import annotations

import pytest

from app.llm.base import LLMClient, LLMRequest, LLMResponse
from app.llm.mock import MockLLMClient
from app.pipeline.models import (
    DocumentType,
    InvoiceSummary,
    PipelineConfig,
    PipelineStatus,
    RawDocument,
    StepName,
)
from app.pipeline.runner import run_pipeline

DOC = RawDocument(doc_id="d1", content="Invoice 42 from Acme. Total due $10.00.")
GOOD_SCRIPTS = {
    ("d1", "extraction"): {"organizations": [{"name": "Acme"}]},
    ("d1", "classification"): {"document_type": "invoice", "rationale": "bill"},
    ("d1", "summarization"): {"headline": "Acme bill", "invoice_number": "42"},
}


def test_happy_path_runs_all_four_steps() -> None:
    llm = MockLLMClient(GOOD_SCRIPTS)

    result = run_pipeline(DOC, llm)

    assert result.status is PipelineStatus.COMPLETED
    assert result.error is None
    assert result.normalized is not None
    assert result.entities is not None
    assert result.entities.organizations[0].name == "Acme"
    assert result.classification is not None
    assert result.classification.document_type is DocumentType.INVOICE
    assert isinstance(result.summary, InvoiceSummary)
    assert [c.metadata["step"] for c in llm.calls] == [
        "extraction",
        "classification",
        "summarization",
    ]


def test_config_is_applied() -> None:
    llm = MockLLMClient(GOOD_SCRIPTS)

    result = run_pipeline(DOC, llm, PipelineConfig(intake_max_chars=10, temperature=0.5))

    assert result.normalized is not None
    assert result.normalized.truncated is True
    assert {c.temperature for c in llm.calls} == {0.5}


def test_intake_failure_stops_before_any_llm_call() -> None:
    llm = MockLLMClient(GOOD_SCRIPTS)

    result = run_pipeline(RawDocument(doc_id="d1", content="   "), llm)

    assert result.status is PipelineStatus.FAILED
    assert result.error is not None
    assert result.error.step is StepName.INTAKE
    assert result.error.error_type == "IntakeError"
    assert llm.calls == []


def test_malformed_llm_output_is_recorded_with_raw_text() -> None:
    llm = MockLLMClient({**GOOD_SCRIPTS, ("d1", "extraction"): "Sorry, I can't do that."})

    result = run_pipeline(DOC, llm)

    assert result.status is PipelineStatus.FAILED
    assert result.error is not None
    assert result.error.step is StepName.EXTRACTION
    assert result.error.error_type == "LLMOutputError"
    assert result.error.raw_output == "Sorry, I can't do that."
    assert result.normalized is not None
    assert result.entities is None


def test_repair_attempts_come_from_the_config() -> None:
    llm = MockLLMClient({**GOOD_SCRIPTS, ("d1", "extraction"): "not JSON"})

    result = run_pipeline(DOC, llm, PipelineConfig(max_repair_attempts=0))

    assert result.status is PipelineStatus.FAILED
    assert [c.metadata["step"] for c in llm.calls] == ["extraction"]


def test_failure_keeps_outputs_of_earlier_steps() -> None:
    scripts = {k: v for k, v in GOOD_SCRIPTS.items() if k[1] != "classification"}

    result = run_pipeline(DOC, MockLLMClient(scripts))

    assert result.error is not None
    assert result.error.step is StepName.CLASSIFICATION
    assert result.error.error_type == "MockScriptMissingError"
    assert result.error.raw_output is None
    assert result.entities is not None
    assert result.classification is None


def test_unexpected_exceptions_propagate() -> None:
    class _BrokenClient(LLMClient):
        @property
        def model_name(self) -> str:
            return "broken"

        def complete(self, request: LLMRequest) -> LLMResponse:
            raise ZeroDivisionError("bug")

    with pytest.raises(ZeroDivisionError):
        run_pipeline(DOC, _BrokenClient())
