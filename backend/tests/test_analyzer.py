"""Tests for the backward walk: root cause, roles, evidence chain and verdict reuse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.analysis import analyzer
from app.analysis.analyzer import EXCERPT_CHARS, analyze_trace, step_label
from app.analysis.judge import JUDGE_STEP, judge_messages, request_fingerprint
from app.analysis.models import JudgeAnswer, StepRole
from app.analysis.taxonomy import FailureCategory
from app.llm.mock import MockLLMClient
from app.pipeline.documents import load_document, load_manifest
from app.pipeline.models import RawDocument, StepName
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline


def _trace(doc_id: str, data_dir: Path, corpus_llm: MockLLMClient) -> Trace:
    entry = load_manifest(data_dir).get(doc_id)
    return trace_pipeline(load_document(entry, data_dir), corpus_llm)[1]


def _grade(step: str, score: int = 5, **extra: Any) -> dict[str, Any]:
    return {"step": step, "score": score, "explanation": f"{step} verdict", **extra}


def _judge(doc_id: str, *grades: dict[str, Any]) -> MockLLMClient:
    return MockLLMClient({(doc_id, JUDGE_STEP): {"steps": list(grades)}})


NO_DATES = "contract_no_dates_04"
INVENTED = {
    "introduced": ["invented the date January 1, 2024"],
    "category": "extraction_hallucination",
}
REPEATED = {"inherited": ["repeats the invented date January 1, 2024"]}


def test_root_cause_is_the_earliest_significant_drop_and_later_steps_carry_it(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    judge = _judge(
        NO_DATES,
        _grade("intake"),
        _grade("extraction", 1, **INVENTED),
        _grade("classification"),
        _grade("summarization", 5, **REPEATED),
    )

    analysis = analyze_trace(_trace(NO_DATES, data_dir, corpus_llm), judge)

    assert analysis.root_cause_step is StepName.EXTRACTION
    assert analysis.category is FailureCategory.EXTRACTION_HALLUCINATION
    assert [f.role for f in analysis.steps] == [
        StepRole.HEALTHY,
        StepRole.ROOT_CAUSE,
        StepRole.HEALTHY,
        StepRole.PROPAGATED,
    ]
    assert analysis.summary == (
        "Root cause: Step 2 (Extraction), Extraction Hallucination. extraction verdict. "
        "It propagated to Step 4 (Summarization): repeats the invented date January 1, 2024."
    )


def test_an_earlier_culprit_beats_a_later_one_which_becomes_secondary(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    judge = _judge(
        NO_DATES,
        _grade("intake"),
        _grade("extraction", 2, **INVENTED),
        _grade("classification"),
        _grade("summarization", 2, introduced=["drops the fee"], category="context_loss"),
    )

    analysis = analyze_trace(_trace(NO_DATES, data_dir, corpus_llm), judge)

    assert analysis.root_cause_step is StepName.EXTRACTION
    assert analysis.steps[3].role is StepRole.SECONDARY
    assert "Also failed: Step 4 (Summarization): drops the fee." in analysis.summary


def test_a_problem_above_the_drop_line_is_minor_and_names_no_root_cause(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    judge = _judge(
        NO_DATES,
        _grade("intake"),
        _grade("extraction", 4, introduced=["key terms are a bit vague"], category="context_loss"),
        _grade("classification"),
        _grade("summarization"),
    )

    analysis = analyze_trace(_trace(NO_DATES, data_dir, corpus_llm), judge)

    assert analysis.root_cause_step is None and analysis.category is None
    assert analysis.steps[1].role is StepRole.MINOR
    assert analysis.summary == (
        "No step showed a significant quality drop (every step scored above 2/5). "
        "Minor issues: Step 2 (Extraction): key terms are a bit vague."
    )


@pytest.mark.parametrize(("drop_score", "expected"), [(2, None), (3, StepName.EXTRACTION)])
def test_the_drop_line_is_configurable(
    drop_score: int, expected: StepName | None, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    judge = _judge(
        NO_DATES,
        _grade("intake"),
        _grade("extraction", 3, **INVENTED),
        _grade("classification"),
        _grade("summarization"),
    )

    analysis = analyze_trace(_trace(NO_DATES, data_dir, corpus_llm), judge, drop_score=drop_score)

    assert analysis.root_cause_step is expected
    assert analysis.drop_score == drop_score


def test_a_crashed_run_is_diagnosed_on_the_steps_that_ran(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    doc_id = "report_broken_answer_28"
    judge = _judge(
        doc_id,
        _grade("intake"),
        _grade("extraction", 1, introduced=["no valid JSON"], category="prompt_failure"),
    )

    analysis = analyze_trace(_trace(doc_id, data_dir, corpus_llm), judge)

    assert analysis.root_cause_step is StepName.EXTRACTION
    assert analysis.category is FailureCategory.PROMPT_FAILURE
    assert [f.role for f in analysis.steps[2:]] == [StepRole.NOT_RUN, StepRole.NOT_RUN]
    assert analysis.steps[1].produced is not None
    assert analysis.steps[1].produced.startswith("FAILED (LLMOutputError")


def test_evidence_excerpts_show_what_each_step_received_and_produced(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    judge = _judge(
        "contract_long_liability_29",
        *(_grade(s.value) for s in StepName),
    )

    analysis = analyze_trace(_trace("contract_long_liability_29", data_dir, corpus_llm), judge)

    intake, extraction, classification, _ = analysis.steps
    assert intake.received is not None and intake.received.startswith("SOFTWARE IMPLEMENTATION")
    assert len(intake.received) == EXCERPT_CHARS and intake.received.endswith("…")
    assert extraction.produced is not None and '"Copperline Systems Inc."' in extraction.produced
    assert classification.received is not None
    assert classification.received.startswith('{"entities"')  # the document text is not repeated


def test_an_identical_request_reuses_the_saved_verdict_without_calling_the_judge(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    trace = _trace(NO_DATES, data_dir, corpus_llm)
    saved = JudgeAnswer.model_validate(
        {
            "steps": [
                _grade("intake"),
                _grade("extraction", 1, **INVENTED),
                _grade("classification"),
                _grade("summarization"),
            ]
        }
    )
    judge = MockLLMClient({})  # would raise if it were called
    asked: list[str] = []

    def reuse(fingerprint: str) -> JudgeAnswer:
        asked.append(fingerprint)
        return saved

    analysis = analyze_trace(trace, judge, reuse=reuse)

    assert analysis.reused_verdict and analysis.judge_calls == [] and judge.calls == []
    assert asked == [analysis.judge_fingerprint]
    assert analysis.root_cause_step is StepName.EXTRACTION


def test_fingerprint_depends_on_the_content_and_the_model_not_on_ids_or_times(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    first = _trace(NO_DATES, data_dir, corpus_llm)
    second = _trace(NO_DATES, data_dir, corpus_llm)
    other = _trace("invoice_simple_06", data_dir, corpus_llm)

    assert first.trace_id != second.trace_id
    fp = request_fingerprint(judge_messages(first), "m")
    assert fp == request_fingerprint(judge_messages(second), "m")
    assert fp != request_fingerprint(judge_messages(first), "another-model")
    assert fp != request_fingerprint(judge_messages(other), "m")


def test_analysis_records_the_judge_and_its_calls(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    judge = _judge(NO_DATES, *(_grade(s.value) for s in StepName))

    analysis = analyze_trace(_trace(NO_DATES, data_dir, corpus_llm), judge)

    assert analysis.judge_model == "mock-llm"
    assert analysis.judge_prompt_version == "1.2.0"
    assert len(analysis.judge_calls) == 1 and not analysis.reused_verdict
    assert analysis.duration_ms >= 0
    assert len(analysis.analysis_id) == 32
    # the automatic checks ride along as a second opinion; the judge said "healthy" and wins
    assert {c.name for c in analysis.checks if c.flagged} == {
        "ungrounded_entities",
        "summary_numbers_not_in_document",
    }
    assert analysis.root_cause_step is None


def test_a_problem_written_as_a_full_sentence_gets_no_double_full_stop(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    judge = _judge(
        NO_DATES,
        _grade("intake"),
        _grade("extraction", 1, **INVENTED),
        _grade("classification"),
        _grade("summarization", 5, inherited=["Repeats the invented date."]),
    )

    analysis = analyze_trace(_trace(NO_DATES, data_dir, corpus_llm), judge)

    assert analysis.summary.endswith("Step 4 (Summarization): Repeats the invented date.")
    assert ".." not in analysis.summary


def test_step_labels() -> None:
    assert step_label(StepName.INTAKE) == "Step 1 (Intake)"
    assert step_label(StepName.SUMMARIZATION) == "Step 4 (Summarization)"


def test_a_run_that_fails_at_intake_is_diagnosed_on_intake_alone(
    corpus_llm: MockLLMClient,
) -> None:
    _, trace = trace_pipeline(RawDocument(doc_id="blank_doc", content="  \n\t "), corpus_llm)
    judge = _judge(
        "blank_doc",
        _grade("intake", 1, introduced=["nothing usable remained"], category="context_loss"),
    )

    analysis = analyze_trace(trace, judge)

    assert "This step FAILED (IntakeError" in judge.calls[0].messages[1].content
    assert analysis.root_cause_step is StepName.INTAKE
    assert analysis.steps[0].produced is not None
    assert analysis.steps[0].produced.startswith("FAILED (IntakeError")
    assert [f.role for f in analysis.steps[1:]] == [StepRole.NOT_RUN] * 3


def test_a_missing_value_has_no_excerpt() -> None:
    assert analyzer._excerpt(None) is None
    assert analyzer._excerpt({"a": 1}) == '{"a": 1}'
