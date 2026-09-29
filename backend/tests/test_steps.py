"""Tests for the LLM-backed steps (Extraction, Classification, Summarization)."""

from __future__ import annotations

import json

import pytest

from app.llm.base import META_ATTEMPT, META_DOC_ID, META_STEP, LLMOutputError, Role
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
from app.tracing.tracer import start_trace

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
    assert request.metadata == {META_DOC_ID: "d1", META_STEP: "extraction", META_ATTEMPT: "1"}
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


# --------------------------------------------------------------------------- confidence and repair

GOOD = {"people": [{"name": "Jane Roe"}], "confidence": 4}


def test_confidence_is_split_off_the_answer_and_recorded_on_the_span() -> None:
    llm = MockLLMClient({("d1", "extraction"): GOOD})

    with start_trace("d1", model="mock-llm") as recorder:
        entities = run_extraction(DOC, llm)

    assert entities == ExtractedEntities(people=[Person(name="Jane Roe")])
    (span,) = recorder.spans
    assert (span.confidence, span.confidence_note) == (4, None)
    assert span.features["confidence"] == 4
    assert "confidence" not in span.output


def test_invalid_answer_is_repaired_on_the_second_attempt() -> None:
    llm = MockLLMClient({("d1", "extraction"): ["Sorry, no JSON today.", GOOD]})

    with start_trace("d1", model="mock-llm") as recorder:
        entities = run_extraction(DOC, llm)

    assert entities.people[0].name == "Jane Roe"
    first, second = llm.calls
    assert [first.metadata[META_ATTEMPT], second.metadata[META_ATTEMPT]] == ["1", "2"]
    assert [m.role for m in second.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert second.messages[2].content == "Sorry, no JSON today."
    assert "does not match ExtractedEntities" in second.messages[3].content
    calls = recorder.spans[0].llm_calls
    assert [c.attempt for c in calls] == [1, 2]
    assert calls[0].error is not None and calls[1].error is None
    assert recorder.spans[0].features["repair_attempts"] == 1


def test_repair_gives_up_after_the_allowed_attempts() -> None:
    llm = MockLLMClient({("d1", "extraction"): "still not JSON"})

    with pytest.raises(LLMOutputError) as info:
        run_extraction(DOC, llm, max_repair_attempts=2)

    assert len(llm.calls) == 3
    assert info.value.raw_output == "still not JSON"


def test_no_repair_when_attempts_are_zero() -> None:
    llm = MockLLMClient({("d1", "extraction"): ["not JSON", GOOD]})

    with pytest.raises(LLMOutputError):
        run_extraction(DOC, llm, max_repair_attempts=0)

    assert len(llm.calls) == 1


def test_missing_confidence_is_recorded_but_the_answer_is_kept() -> None:
    llm = MockLLMClient({("d1", "extraction"): {"people": []}})

    with start_trace("d1", model="mock-llm") as recorder:
        run_extraction(DOC, llm)

    span = recorder.spans[0]
    assert (span.confidence, span.confidence_note) == (None, "missing")
    assert len(llm.calls) == 1
