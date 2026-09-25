"""Tests for the pipeline's Pydantic models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import TypeAdapter, ValidationError

from app.pipeline.models import (
    SUMMARY_MODELS,
    ContractSummary,
    DocumentSummary,
    DocumentType,
    ExtractedEntities,
    InvoiceSummary,
    MoneyAmount,
    RawDocument,
)


def test_money_amount_keeps_exact_decimal() -> None:
    amount = MoneyAmount.model_validate_json('{"raw": "$0.10", "value": 0.1, "currency": "USD"}')

    assert amount.value == Decimal("0.1")


@pytest.mark.parametrize("currency", ["usd", "US", "DOLLARS"])
def test_currency_must_be_iso_code(currency: str) -> None:
    with pytest.raises(ValidationError):
        MoneyAmount(raw="1", value=Decimal(1), currency=currency)


def test_llm_facing_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ExtractedEntities.model_validate({"people": [], "invented_field": 1})


def test_extracted_entities_defaults_to_empty_lists() -> None:
    entities = ExtractedEntities()

    assert entities.people == []
    assert entities.dates == []
    assert entities.amounts == []


@pytest.mark.parametrize("doc_id", ["", "Has Spaces", "UPPER", "../escape"])
def test_raw_document_rejects_bad_ids(doc_id: str) -> None:
    with pytest.raises(ValidationError):
        RawDocument(doc_id=doc_id, content="text")


def test_discriminated_union_picks_summary_type() -> None:
    adapter: TypeAdapter[DocumentSummary] = TypeAdapter(DocumentSummary)

    summary = adapter.validate_python(
        {"document_type": "invoice", "headline": "h", "due_date": "2025-06-11"}
    )

    assert isinstance(summary, InvoiceSummary)
    assert summary.due_date == date(2025, 6, 11)


def test_summary_rejects_wrong_document_type() -> None:
    with pytest.raises(ValidationError):
        ContractSummary.model_validate({"document_type": "invoice", "headline": "h"})


def test_every_document_type_has_a_summary_model() -> None:
    assert set(SUMMARY_MODELS) == set(DocumentType)
    for doc_type, model in SUMMARY_MODELS.items():
        assert model(headline="h").document_type is doc_type
