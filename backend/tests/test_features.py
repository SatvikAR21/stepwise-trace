"""Tests for the step-specific measurements (features) recorded on spans."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app.llm.mock import MockLLMClient
from app.pipeline.documents import DocumentManifest, load_document, load_manifest
from app.pipeline.features import (
    classification_features,
    extraction_features,
    intake_features,
    is_grounded,
    numbers_in,
    summarization_features,
)
from app.pipeline.models import (
    ClassificationInput,
    ClassificationResult,
    DateMention,
    DocumentFormat,
    DocumentType,
    ExtractedEntities,
    InvoiceSummary,
    MoneyAmount,
    NormalizedDocument,
    Person,
    RawDocument,
    SummarizationInput,
)
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline

TEXT = "Invoice from Acme Corp.\nTotal due:  $1,284.50 by June 11, 2025. Contact Jane   Roe."
DOC = NormalizedDocument(
    doc_id="d1",
    format=DocumentFormat.TEXT,
    text=TEXT,
    char_count=len(TEXT),
    word_count=len(TEXT.split()),
    original_char_count=len(TEXT),
)
ENTITIES = ExtractedEntities(
    people=[Person(name="Jane Roe"), Person(name="John Smith")],
    dates=[DateMention(raw="June 11, 2025")],
    amounts=[MoneyAmount(raw="$1,284.50", value=Decimal("1284.50"), currency="USD")],
    key_terms=["office supplies"],
)
INVOICE = ClassificationResult(document_type=DocumentType.INVOICE, rationale="A bill.")


def test_grounding_ignores_case_and_whitespace() -> None:
    assert is_grounded("jane roe", TEXT)
    assert is_grounded("Total due: $1,284.50", TEXT)
    assert not is_grounded("John Smith", TEXT)


def test_numbers_are_compared_by_value() -> None:
    assert numbers_in("$1,284.50 and 12% of 3") == {Decimal("1284.5"), Decimal(12), Decimal(3)}


def test_intake_features_measure_truncation() -> None:
    output = DOC.model_copy(
        update={"char_count": 50, "original_char_count": 200, "truncated": True}
    )

    features = intake_features(RawDocument(doc_id="d1", content="x"), output)

    assert features == {
        "truncated": True,
        "original_chars": 200,
        "kept_chars": 50,
        "kept_ratio": 0.25,
        "cleanup_notes": 0,
    }


def test_extraction_features_count_ungrounded_entities() -> None:
    features = extraction_features(DOC, ENTITIES)

    assert features["people"] == 2
    assert features["amounts"] == 1
    assert features["key_terms"] == 1
    assert features["currencies"] == 1
    assert features["entities_checked"] == 4
    assert features["entities_ungrounded"] == 1
    assert features["grounded_ratio"] == 0.75


def test_empty_extraction_counts_as_fully_grounded() -> None:
    assert extraction_features(DOC, ExtractedEntities())["grounded_ratio"] == 1.0


def test_classification_features() -> None:
    features = classification_features(
        ClassificationInput(document=DOC, entities=ENTITIES), INVOICE
    )

    assert features == {"document_type": "invoice", "rationale_chars": 7}


def test_summary_numbers_not_in_the_document_are_counted() -> None:
    summary = InvoiceSummary(
        headline="Acme bills $1,284.50; with fees 1,350.00.",
        total_amount=MoneyAmount(raw="USD 1,350.00", value=Decimal(1350), currency="USD"),
        due_date="2025-06-11",  # type: ignore[arg-type]  # ISO dates are skipped
        line_item_count=7,  # counts are skipped
    )
    step_input = SummarizationInput(document=DOC, entities=ENTITIES, classification=INVOICE)

    features = summarization_features(step_input, summary)

    assert features == {"key_points": 0, "summary_numbers": 2, "summary_numbers_ungrounded": 1}


# --------------------------------------------------------------------------- on the real corpus

UNGROUNDED_ENTITIES = {"contract_no_dates_04": 1, "correspondence_unnamed_ceo_19": 1}
UNGROUNDED_SUMMARY_NUMBERS = {
    "contract_no_dates_04": 1,  # the invented year 2024
    "invoice_multi_currency_09": 1,  # the wrong total 8,350.00
    "report_postmortem_12": 1,  # known false alarm: a correctly computed 87-minute duration
}


@pytest.fixture(scope="module")
def corpus_traces() -> dict[str, Trace]:
    from app.core.config import BACKEND_DIR

    data_dir = BACKEND_DIR / "data"
    manifest: DocumentManifest = load_manifest(data_dir)
    llm = MockLLMClient(scripts_dir=Path(data_dir) / "mock_responses")
    return {
        entry.doc_id: trace_pipeline(load_document(entry, data_dir), llm)[1]
        for entry in manifest.documents
    }


def test_corpus_grounding_flags_exactly_the_invented_entities(
    corpus_traces: dict[str, Trace],
) -> None:
    for doc_id, trace in corpus_traces.items():
        extraction = trace.spans[1]
        assert extraction.features["entities_ungrounded"] == UNGROUNDED_ENTITIES.get(doc_id, 0), (
            doc_id
        )


def test_corpus_summary_numbers_flag_invented_values_and_one_known_false_alarm(
    corpus_traces: dict[str, Trace],
) -> None:
    for doc_id, trace in corpus_traces.items():
        summary = trace.spans[3]
        expected = UNGROUNDED_SUMMARY_NUMBERS.get(doc_id, 0)
        assert summary.features["summary_numbers_ungrounded"] == expected, doc_id


def test_only_the_long_report_is_truncated(corpus_traces: dict[str, Trace]) -> None:
    truncated = {d for d, t in corpus_traces.items() if t.spans[0].features["truncated"] is True}

    assert truncated == {"report_supplier_risk_long_15"}
