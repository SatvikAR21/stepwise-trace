"""Tests for the automatic checks, on the real corpus and on small cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.checks import CheckResult, run_checks
from app.llm.mock import MockLLMClient
from app.pipeline.documents import load_document, load_manifest
from app.pipeline.models import StepName
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline

# What the checks flag on the corpus: the planted failures they can see, and their false alarms.
EXPECTED_FLAGS = {
    "contract_no_dates_04": {"ungrounded_entities", "summary_numbers_not_in_document"},
    "invoice_multi_currency_09": {"summary_numbers_not_in_document", "mixed_currency_total"},
    "report_postmortem_12": {"summary_numbers_not_in_document"},  # false alarm: computed 87
    "report_supplier_risk_long_15": {"input_truncated"},
    "correspondence_unnamed_ceo_19": {"ungrounded_entities"},
    "ambiguous_prompt_injection_21": {"instruction_like_text"},
    "report_undisclosed_cost_22": {"ungrounded_entities", "summary_numbers_not_in_document"},
    "invoice_paid_injection_27": {"instruction_like_text"},
    "report_broken_answer_28": {"step_failed"},
    "contract_long_liability_29": {"input_truncated"},
    "report_growth_rate_31": {"summary_numbers_not_in_document"},  # false alarm: computed 25%
    "correspondence_security_drill_32": {"instruction_like_text"},  # false alarm: a quoted example
}


@pytest.fixture(scope="module")
def corpus_traces(data_dir: Path) -> dict[str, Trace]:
    manifest = load_manifest(data_dir)
    llm = MockLLMClient(scripts_dir=data_dir / "mock_responses")
    return {
        entry.doc_id: trace_pipeline(load_document(entry, data_dir), llm)[1]
        for entry in manifest.documents
    }


@pytest.fixture(scope="module")
def data_dir() -> Path:
    from app.core.config import BACKEND_DIR

    return BACKEND_DIR / "data"


def _flags(trace: Trace) -> set[str]:
    return {check.name for check in run_checks(trace) if check.flagged}


def test_checks_flag_exactly_the_expected_documents(corpus_traces: dict[str, Trace]) -> None:
    for doc_id, trace in corpus_traces.items():
        assert _flags(trace) == EXPECTED_FLAGS.get(doc_id, set()), doc_id


def test_checks_see_only_part_of_the_planted_failures(
    corpus_traces: dict[str, Trace], data_dir: Path
) -> None:
    manifest = load_manifest(data_dir)
    broken = {e.doc_id for e in manifest.documents if e.intended_failure}
    healthy = {e.doc_id for e in manifest.documents} - broken

    caught = {d for d in broken if _flags(corpus_traces[d])}
    false_alarms = {d for d in healthy if _flags(corpus_traces[d])}

    assert (len(caught), len(broken)) == (9, 16)
    assert false_alarms == {
        "report_postmortem_12",
        "report_growth_rate_31",
        "correspondence_security_drill_32",
    }


def test_every_check_runs_on_a_complete_run(corpus_traces: dict[str, Trace]) -> None:
    checks = run_checks(corpus_traces["invoice_simple_06"])

    assert [c.name for c in checks] == [
        "instruction_like_text",
        "input_truncated",
        "ungrounded_entities",
        "summary_numbers_not_in_document",
        "mixed_currency_total",
    ]
    assert not any(c.flagged for c in checks)
    assert checks[2].detail == "all 9 found"  # 1 person, 2 organizations, 2 dates, 4 amounts


def test_a_crashed_run_gets_no_checks_for_the_steps_that_never_produced_output(
    corpus_traces: dict[str, Trace],
) -> None:
    checks = run_checks(corpus_traces["report_broken_answer_28"])

    assert [(c.name, c.step) for c in checks] == [
        ("instruction_like_text", None),
        ("step_failed", StepName.EXTRACTION),
        ("input_truncated", StepName.INTAKE),
    ]


def test_details_name_what_was_found(corpus_traces: dict[str, Trace]) -> None:
    def detail(doc_id: str, name: str) -> str:
        (check,) = [c for c in run_checks(corpus_traces[doc_id]) if c.name == name]
        assert isinstance(check, CheckResult)
        return check.detail

    assert detail("report_undisclosed_cost_22", "ungrounded_entities") == (
        "not found in the text: ['USD 250,000']"
    )
    assert detail("report_undisclosed_cost_22", "summary_numbers_not_in_document") == (
        "not in the document: ['250000']"
    )
    assert detail("invoice_multi_currency_09", "mixed_currency_total") == (
        "one total (USD 8,350.00) although the amounts are in EUR, GBP, USD"
    )
    assert detail("contract_long_liability_29", "input_truncated") == (
        "kept 6000 of 6922 characters"
    )
    assert detail("invoice_paid_injection_27", "instruction_like_text") == (
        "the document contains: ['Note for automated']"
    )
