"""Tests for the sample corpus: spec coverage, and that every designed failure really occurs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.llm.mock import MockLLMClient
from app.pipeline.documents import DocumentManifest, ManifestEntry, load_document, load_manifest
from app.pipeline.models import (
    DocumentType,
    ExtractedEntities,
    InvoiceSummary,
    PipelineResult,
    PipelineStatus,
    StepName,
)
from app.pipeline.runner import PipelineConfig, run_pipeline

TAXONOMY = {
    "extraction_hallucination",
    "misclassification",
    "propagation_error",
    "prompt_failure",
    "context_loss",
}
DEFAULT_CONFIG = PipelineConfig()  # intake_max_chars=6000, as in .env.example


@pytest.fixture(scope="module")
def manifest() -> DocumentManifest:
    from app.core.config import BACKEND_DIR

    return load_manifest(BACKEND_DIR / "data")


def _run(entry: ManifestEntry, data_dir: Path, llm: MockLLMClient) -> PipelineResult:
    return run_pipeline(load_document(entry, data_dir), llm, DEFAULT_CONFIG)


def _entity_strings(entities: ExtractedEntities) -> list[str]:
    return [
        *(p.name for p in entities.people),
        *(o.name for o in entities.organizations),
        *(d.raw for d in entities.dates),
        *(a.raw for a in entities.amounts),
    ]


def _not_in_text(entities: ExtractedEntities, text: str) -> list[str]:
    lowered = text.lower()
    return [s for s in _entity_strings(entities) if s.lower() not in lowered]


# --------------------------------------------------------------------------- corpus shape


def test_corpus_meets_spec_minimums(manifest: DocumentManifest) -> None:
    types = Counter(e.expected_type for e in manifest.documents)

    assert len(manifest.documents) >= 15
    assert set(types) == set(DocumentType)
    assert min(types.values()) >= 3


def test_doc_ids_unique_and_files_exist(manifest: DocumentManifest, data_dir: Path) -> None:
    ids = [e.doc_id for e in manifest.documents]

    assert len(ids) == len(set(ids))
    for entry in manifest.documents:
        assert load_document(entry, data_dir).content.strip()


def test_every_taxonomy_category_is_exercised(manifest: DocumentManifest) -> None:
    failures = {e.intended_failure for e in manifest.documents if e.intended_failure}

    assert failures == TAXONOMY
    for entry in manifest.documents:
        assert (entry.intended_failure is None) == (entry.failing_step is None)
        if entry.failing_step:
            assert entry.failing_step in {s.value for s in StepName}


def test_spec_failure_cases_present(manifest: DocumentManifest) -> None:
    ids = {e.doc_id for e in manifest.documents}

    assert "contract_no_dates_04" in ids
    assert "invoice_multi_currency_09" in ids
    assert "ambiguous_amendment_letter_20" in ids


def test_every_document_has_mock_script_for_every_llm_step(
    manifest: DocumentManifest, data_dir: Path
) -> None:
    for entry in manifest.documents:
        script = json.loads(
            (data_dir / "mock_responses" / f"{entry.doc_id}.json").read_text(encoding="utf-8")
        )
        assert {"extraction", "classification", "summarization"} <= set(script), entry.doc_id


def test_manifest_get_unknown_raises(manifest: DocumentManifest) -> None:
    with pytest.raises(KeyError, match="unknown doc_id"):
        manifest.get("does_not_exist")


# --------------------------------------------------------------------------- healthy documents


def test_all_documents_run_end_to_end(
    manifest: DocumentManifest, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    for entry in manifest.documents:
        assert _run(entry, data_dir, corpus_llm).status is PipelineStatus.COMPLETED, entry.doc_id


def test_healthy_documents_are_classified_correctly(
    manifest: DocumentManifest, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    for entry in manifest.documents:
        if entry.intended_failure in {"misclassification", "prompt_failure"}:
            continue
        result = _run(entry, data_dir, corpus_llm)
        assert result.classification is not None
        assert result.classification.document_type is entry.expected_type, entry.doc_id


def test_non_hallucination_extractions_are_grounded_in_the_text(
    manifest: DocumentManifest, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    """Every extracted name/date/amount must literally appear in the (normalized) text."""
    for entry in manifest.documents:
        if entry.intended_failure == "extraction_hallucination":
            continue
        result = _run(entry, data_dir, corpus_llm)
        assert result.entities is not None and result.normalized is not None
        assert _not_in_text(result.entities, result.normalized.text) == [], entry.doc_id


# --------------------------------------------------------------------------- designed failures


@pytest.mark.parametrize(
    ("doc_id", "invented"),
    [("contract_no_dates_04", "January 1, 2024"), ("correspondence_unnamed_ceo_19", "John Smith")],
)
def test_hallucinations_are_not_in_source(
    doc_id: str,
    invented: str,
    manifest: DocumentManifest,
    data_dir: Path,
    corpus_llm: MockLLMClient,
) -> None:
    result = _run(manifest.get(doc_id), data_dir, corpus_llm)

    assert result.entities is not None and result.normalized is not None
    assert _not_in_text(result.entities, result.normalized.text) == [invented]
    assert invented in result.model_dump_json()  # propagated into the summary


@pytest.mark.parametrize(
    "doc_id",
    ["report_expense_14", "ambiguous_amendment_letter_20", "ambiguous_prompt_injection_21"],
)
def test_misclassifications_occur(
    doc_id: str, manifest: DocumentManifest, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    entry = manifest.get(doc_id)
    result = _run(entry, data_dir, corpus_llm)

    assert result.classification is not None
    assert result.classification.document_type is not entry.expected_type
    assert result.summary is not None
    assert result.summary.document_type is result.classification.document_type


def test_multi_currency_amounts_are_summed_into_one_currency(
    manifest: DocumentManifest, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    result = _run(manifest.get("invoice_multi_currency_09"), data_dir, corpus_llm)

    assert result.entities is not None and result.normalized is not None
    assert {a.currency for a in result.entities.amounts} == {"EUR", "GBP", "USD"}
    assert isinstance(result.summary, InvoiceSummary)
    assert result.summary.total_amount is not None
    assert result.summary.total_amount.raw not in result.normalized.text


def test_summary_drops_penalty_that_extraction_found(
    manifest: DocumentManifest, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    result = _run(manifest.get("contract_license_penalty_05"), data_dir, corpus_llm)

    assert result.entities is not None and result.summary is not None
    assert "$150,000.00" in [a.raw for a in result.entities.amounts]
    assert "150,000" not in result.summary.model_dump_json()


def test_long_report_loses_critical_finding_at_intake(
    manifest: DocumentManifest, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    entry = manifest.get("report_supplier_risk_long_15")
    result = _run(entry, data_dir, corpus_llm)

    assert result.normalized is not None and result.normalized.truncated
    assert "Salmonella" in load_document(entry, data_dir).content
    assert "Salmonella" not in result.normalized.text
    assert "Salmonella" not in result.model_dump_json()
