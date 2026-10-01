"""Tests for re-running a recorded run from any step."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.llm.mock import MockLLMClient
from app.pipeline.documents import load_document, load_manifest
from app.pipeline.models import (
    ExtractedEntities,
    Person,
    PipelineConfig,
    PipelineResult,
    PipelineStatus,
    StepName,
)
from app.pipeline.resume import resume_pipeline
from app.pipeline.runner import run_from
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline


def _traced(
    doc_id: str, data_dir: Path, llm: MockLLMClient, config: PipelineConfig | None = None
) -> tuple[PipelineResult, Trace]:
    entry = load_manifest(data_dir).get(doc_id)
    return trace_pipeline(load_document(entry, data_dir), llm, config)


@pytest.mark.parametrize("from_step", list(StepName))
def test_resuming_from_any_step_reproduces_every_completed_corpus_run(
    from_step: StepName, data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    for entry in load_manifest(data_dir).documents:
        if entry.doc_id == "report_broken_answer_28":
            continue  # crashed: covered below
        original, trace = trace_pipeline(load_document(entry, data_dir), corpus_llm)

        resumed = resume_pipeline(trace, from_step, corpus_llm)

        assert resumed == original, (entry.doc_id, from_step)


def test_resuming_the_crashed_run_fails_again_or_says_what_is_missing(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    original, trace = _traced("report_broken_answer_28", data_dir, corpus_llm)

    again = resume_pipeline(trace, StepName.EXTRACTION, corpus_llm)

    assert again == original and again.status is PipelineStatus.FAILED
    with pytest.raises(ValueError, match="no output for extraction"):
        resume_pipeline(trace, StepName.CLASSIFICATION, corpus_llm)


def test_a_replaced_output_is_what_the_later_steps_receive(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    _, trace = _traced("invoice_simple_06", data_dir, corpus_llm)
    recorded = ExtractedEntities.model_validate(trace.spans[1].output)
    injected = recorded.model_copy(update={"people": [Person(name="Zed Phantom", role="Signer")]})
    calls_before = len(corpus_llm.calls)

    result = resume_pipeline(
        trace, StepName.CLASSIFICATION, corpus_llm, replace={StepName.EXTRACTION: injected}
    )

    assert result.entities == injected
    new_calls = corpus_llm.calls[calls_before:]
    assert [c.metadata["step"] for c in new_calls] == ["classification", "summarization"]
    assert all("Zed Phantom" in c.messages[1].content for c in new_calls)
    # The mock replays its script whatever it receives; a real model would react to the fault.


def test_resume_uses_the_settings_the_run_recorded(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    _, trace = _traced(
        "invoice_simple_06", data_dir, corpus_llm, PipelineConfig(intake_max_chars=50)
    )

    resumed = resume_pipeline(trace, StepName.INTAKE, corpus_llm)

    assert resumed.normalized is not None and resumed.normalized.truncated
    assert resumed.normalized.char_count <= 50


def test_resume_of_an_older_trace_without_settings_uses_the_defaults(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    original, trace = _traced("invoice_simple_06", data_dir, corpus_llm)
    old: Trace = trace.model_copy(update={"config": None})

    assert resume_pipeline(old, StepName.INTAKE, corpus_llm) == original


def test_only_earlier_steps_can_be_replaced(data_dir: Path, corpus_llm: MockLLMClient) -> None:
    _, trace = _traced("invoice_simple_06", data_dir, corpus_llm)
    entities = ExtractedEntities.model_validate(trace.spans[1].output)

    with pytest.raises(ValueError, match="can only replace outputs of steps before extraction"):
        resume_pipeline(
            trace, StepName.EXTRACTION, corpus_llm, replace={StepName.EXTRACTION: entities}
        )


def test_running_later_steps_without_the_earlier_outputs_is_refused(
    data_dir: Path, corpus_llm: MockLLMClient
) -> None:
    entry = load_manifest(data_dir).get("invoice_simple_06")

    with pytest.raises(ValueError, match="output of intake is needed"):
        run_from(
            load_document(entry, data_dir),
            corpus_llm,
            PipelineConfig(),
            start=StepName.SUMMARIZATION,
        )
