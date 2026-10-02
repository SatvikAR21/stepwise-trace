"""Tests for the trace judge: the case file it reads, its answer format and its repair attempt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.analysis.judge import (
    JUDGE_STEP,
    JudgeFailedError,
    build_case_file,
    judge_trace,
    render_categories,
    steps_that_ran,
)
from app.analysis.models import JudgeAnswer, StepGrade
from app.analysis.taxonomy import FailureCategory
from app.llm.base import META_ATTEMPT, META_STEP, LLMClient, LLMProviderError, LLMRequest
from app.llm.base import LLMResponse as Response
from app.llm.mock import MockLLMClient
from app.llm.prompts import JUDGE_REPAIR_PROMPT, TRACE_JUDGE_PROMPT
from app.pipeline.documents import load_document, load_manifest
from app.pipeline.models import StepName
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline


def _trace(doc_id: str, data_dir: Path, corpus_llm: MockLLMClient) -> Trace:
    entry = load_manifest(data_dir).get(doc_id)
    return trace_pipeline(load_document(entry, data_dir), corpus_llm)[1]


def _grade(step: str, score: int = 5, **extra: Any) -> dict[str, Any]:
    return {"step": step, "score": score, "explanation": f"{step} is fine.", **extra}


HEALTHY_ANSWER = {
    "steps": [_grade(s) for s in ("intake", "extraction", "classification", "summarization")]
}


# --------------------------------------------------------------------------- the case file


def test_case_file_shows_the_document_once_and_every_step_output(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    trace = _trace("invoice_simple_06", data_dir, corpus_llm)

    case = build_case_file(trace)

    assert case.count("Printer paper, A4, box of 10 reams") == 1  # the text appears once
    assert "nothing cut off" in case
    for number, step in enumerate(StepName, start=1):
        assert f"=== STEP {number}: {step.value} ===" in case
    assert '"invoice_number": "INV-2025-0412"' in case
    assert '"confidence"' not in case  # the judge never sees the self-reported scores


def test_case_file_marks_a_truncated_document_and_shows_where_it_was_cut(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    case = build_case_file(_trace("contract_long_liability_29", data_dir, corpus_llm))

    assert "TRUNCATED" in case
    assert "The kept text ends with" in case
    assert "$95,000.00" in case  # in the original document, shown in full


def test_case_file_shows_a_crashed_steps_raw_answers_and_the_steps_that_never_ran(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    trace = _trace("report_broken_answer_28", data_dir, corpus_llm)

    case = build_case_file(trace)

    assert steps_that_ran(trace) == [StepName.INTAKE, StepName.EXTRACTION]
    assert "This step FAILED (LLMOutputError" in case
    assert "AI answer, attempt 1:" in case and "I hope this helps!" in case
    assert "AI answer, attempt 2:" in case
    assert case.count("Did not run (the pipeline stopped earlier)") == 2


def test_categories_are_rendered_from_the_taxonomy() -> None:
    text = render_categories()

    for category in FailureCategory:
        assert f"- {category.value} (" in text
    assert "- misclassification (Misclassification), can start at: classification." in text


def test_judge_prompt_renders_every_placeholder() -> None:
    messages = TRACE_JUDGE_PROMPT.render(max_chars="6000", categories="CATS", case_file="CASE")

    assert "keeps at most 6000 characters" in messages[0].content
    assert "CATS" in messages[0].content
    assert messages[1].content == "CASE"


def test_judge_prompt_counts_material_text_cut_off_at_intake_as_significant() -> None:
    rendered = TRACE_JUDGE_PROMPT.render(max_chars="6000", categories="C", case_file="X")
    system = " ".join(rendered[0].content.split())  # the prompt wraps its lines

    assert "a dropped deadline, penalty, total or key finding" in system
    assert "if intake cut off material content" in system
    assert "score intake 2 or lower, because no later step can recover it" in system


# --------------------------------------------------------------------------- the answer format


def test_a_category_that_cannot_start_at_the_step_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot start at intake"):
        StepGrade.model_validate(
            _grade("intake", 1, introduced=["x"], category="extraction_hallucination")
        )


def test_a_category_without_an_introduced_problem_is_dropped() -> None:
    grade = StepGrade.model_validate(_grade("summarization", 5, category="context_loss"))

    assert grade.category is None


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StepGrade.model_validate(_grade("intake", confidence=5))


def test_answer_must_cover_exactly_the_steps_that_ran_in_order() -> None:
    answer = JudgeAnswer.model_validate({"steps": [_grade("extraction"), _grade("intake")]})

    assert answer.problems_for([StepName.INTAKE, StepName.EXTRACTION]) == [
        "steps must be exactly ['intake', 'extraction'] in this order, got ['extraction', 'intake']"
    ]
    assert answer.problems_for([StepName.EXTRACTION, StepName.INTAKE]) == []


# --------------------------------------------------------------------------- calling the judge


def test_judge_grades_every_step_in_one_call(data_dir: Path, corpus_llm: MockLLMClient) -> None:
    trace = _trace("invoice_simple_06", data_dir, corpus_llm)
    judge = MockLLMClient({("invoice_simple_06", JUDGE_STEP): HEALTHY_ANSWER})

    result = judge_trace(trace, judge)

    assert [g.step for g in result.answer.steps] == list(StepName)
    assert len(judge.calls) == 1 and len(result.calls) == 1
    assert judge.calls[0].metadata[META_STEP] == JUDGE_STEP
    assert result.calls[0].error is None and result.calls[0].model == "mock-llm"


def test_an_unusable_answer_gets_one_repair_attempt(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    trace = _trace("invoice_simple_06", data_dir, corpus_llm)
    two_steps = {"steps": [_grade("intake"), _grade("extraction")]}
    judge = MockLLMClient({("invoice_simple_06", JUDGE_STEP): [two_steps, HEALTHY_ANSWER]})

    result = judge_trace(trace, judge)

    assert len(result.calls) == 2
    assert "steps must be exactly" in (result.calls[0].error or "")
    repair_request = judge.calls[1]
    assert repair_request.metadata[META_ATTEMPT] == "2"
    assert "Your previous answer could not be used" in repair_request.messages[-1].content
    assert JUDGE_REPAIR_PROMPT.user.split("\n")[0] in repair_request.messages[-1].content


def test_judge_gives_up_after_the_repair_attempts(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    trace = _trace("invoice_simple_06", data_dir, corpus_llm)
    judge = MockLLMClient({("invoice_simple_06", JUDGE_STEP): "not json at all"})

    with pytest.raises(JudgeFailedError) as info:
        judge_trace(trace, judge, max_repair_attempts=1)

    assert len(info.value.calls) == 2
    assert info.value.provider_error is None


class _QuotaUsedUp(LLMClient):
    @property
    def model_name(self) -> str:
        return "gemini-test"

    def complete(self, request: LLMRequest) -> Response:
        raise LLMProviderError("RateLimitError: per day", status_code=429, quota_exhausted=True)


def test_a_provider_failure_is_reported_with_the_original_error(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    trace = _trace("invoice_simple_06", data_dir, corpus_llm)

    with pytest.raises(JudgeFailedError) as info:
        judge_trace(trace, _QuotaUsedUp())

    assert info.value.provider_error is not None
    assert info.value.provider_error.quota_exhausted
    assert len(info.value.calls) == 1 and info.value.calls[0].raw_response is None
