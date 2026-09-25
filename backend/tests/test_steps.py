"""Tests for the LLM-backed steps (Extraction, Classification, Summarization)."""

from __future__ import annotations

import json

import pytest

from app.llm.base import META_DOC_ID, META_STEP, LLMOutputError
from app.llm.mock import MockLLMClient
from app.pipeline.models import (
    ClassificationInput,
    ClassificationResult,
    CorrespondenceSummary,
    DocumentFormat,
    DocumentType,
    ExtractedEntities,
    InvoiceSummary,
    NormalizedDocument,
    Person,
    SummarizationInput,
)
from app.pipeline.steps import run_classification, run_extraction, run_summarization

DOC = NormalizedDocument(
    doc_id="d1",
    format=DocumentFormat.TEXT,
    text="Invoice from Acme. Total due: $10.00. Contact Jane Roe.",
    char_count=55,
    word_count=10,
    original_char_count=55,
)
ENTITIES = ExtractedEntities(people=[Person(name="Jane Roe")])


def test_extraction_sends_document_and_parses_entities() -> None:
    llm = MockLLMClient({("d1", "extraction"): {"people": [{"name": "Jane Roe"}]}})

    entities = run_extraction(DOC, llm, temperature=0.3)

    assert entities.people[0].name == "Jane Roe"
    request = llm.calls[0]
    assert request.metadata == {META_DOC_ID: "d1", META_STEP: "extraction"}
    assert request.json_mode is True
    assert request.temperature == 0.3
    assert DOC.text in request.messages[1].content


def test_classification_includes_entities_in_prompt() -> None:
    llm = MockLLMClient(
        {("d1", "classification"): {"document_type": "invoice", "rationale": "bill"}}
    )

    result = run_classification(ClassificationInput(document=DOC, entities=ENTITIES), llm)

    assert result.document_type is DocumentType.INVOICE
    assert "Jane Roe" in llm.calls[0].messages[1].content


def test_summarization_uses_schema_of_classified_type() -> None:
    llm = MockLLMClient({("d1", "summarization"): {"headline": "Acme bill", "vendor": "Acme"}})
    step_input = SummarizationInput(
        document=DOC,
        entities=ENTITIES,
        classification=ClassificationResult(document_type=DocumentType.INVOICE, rationale="r"),
    )

    summary = run_summarization(step_input, llm)

    assert isinstance(summary, InvoiceSummary)
    assert summary.vendor == "Acme"
    assert '"line_item_count"' in llm.calls[0].messages[0].content


def test_summarization_follows_upstream_classification() -> None:
    llm = MockLLMClient({("d1", "summarization"): {"headline": "A letter"}})
    step_input = SummarizationInput(
        document=DOC,
        entities=ENTITIES,
        classification=ClassificationResult(
            document_type=DocumentType.CORRESPONDENCE, rationale="r"
        ),
    )

    assert isinstance(run_summarization(step_input, llm), CorrespondenceSummary)


def test_summary_of_wrong_shape_is_an_output_error() -> None:
    llm = MockLLMClient(
        {("d1", "summarization"): json.dumps({"document_type": "contract", "headline": "h"})}
    )
    step_input = SummarizationInput(
        document=DOC,
        entities=ENTITIES,
        classification=ClassificationResult(document_type=DocumentType.INVOICE, rationale="r"),
    )

    with pytest.raises(LLMOutputError):
        run_summarization(step_input, llm)
