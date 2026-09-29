"""Tests for recording traces and spans around the pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from app.core.config import LogFormat
from app.core.logging import configure_logging, get_logger
from app.llm.mock import MockLLMClient
from app.pipeline.models import PipelineConfig, PipelineStatus, RawDocument, StepName
from app.tracing.models import FeatureValue, SpanStatus, TraceStatus
from app.tracing.service import trace_pipeline
from app.tracing.tracer import current_span, span, start_trace, traced_step

DOC = RawDocument(doc_id="d1", content="Invoice 42 from Acme. Total due $10.00.")
SCRIPTS: dict[tuple[str, str], Any] = {
    ("d1", "extraction"): {"organizations": [{"name": "Acme"}], "confidence": 5},
    ("d1", "classification"): {"document_type": "invoice", "rationale": "bill", "confidence": 4},
    ("d1", "summarization"): {"headline": "Acme bill", "invoice_number": "42", "confidence": 5},
}
STEPS = ["intake", "extraction", "classification", "summarization"]


def _as_int(value: FeatureValue) -> int:
    assert isinstance(value, int)
    return value


def test_decorator_is_a_no_op_outside_a_trace() -> None:
    @traced_step("double")
    def double(x: int) -> int:
        assert current_span() is None
        return x * 2

    assert double(21) == 42


def test_trace_records_one_span_per_step_in_order() -> None:
    result, trace = trace_pipeline(DOC, MockLLMClient(SCRIPTS))

    assert result.status is PipelineStatus.COMPLETED
    assert [s.name for s in trace.spans] == STEPS
    assert re.fullmatch(r"[0-9a-f]{32}", trace.trace_id)
    assert all(re.fullmatch(r"[0-9a-f]{16}", s.span_id) for s in trace.spans)
    assert {s.trace_id for s in trace.spans} == {trace.trace_id}
    assert all(s.status is SpanStatus.OK and s.duration_ms >= 0 for s in trace.spans)
    assert trace.doc_id == "d1"
    assert trace.model == "mock-llm"
    assert trace.pipeline_status is PipelineStatus.COMPLETED
    assert trace.error_step is None
    assert trace.final_output["invoice_number"] == "42"


def test_trace_records_the_settings_the_run_used() -> None:
    custom = PipelineConfig(intake_max_chars=20, temperature=0.3, max_repair_attempts=0)

    _, default_trace = trace_pipeline(DOC, MockLLMClient(SCRIPTS))
    _, custom_trace = trace_pipeline(DOC, MockLLMClient(SCRIPTS), custom)

    assert default_trace.config == PipelineConfig()
    assert custom_trace.config == custom
    assert custom_trace.spans[0].features["kept_chars"] == 20  # the recorded settings were used


def test_confident_run_is_success_scored_by_its_weakest_step() -> None:
    _, trace = trace_pipeline(DOC, MockLLMClient(SCRIPTS))

    assert [s.confidence for s in trace.spans] == [None, 5, 4, 5]
    assert trace.status is TraceStatus.SUCCESS
    assert trace.status_reasons == []
    assert trace.final_score == 4


def test_repaired_step_makes_the_run_degraded() -> None:
    scripts = {**SCRIPTS, ("d1", "classification"): ["oops", SCRIPTS[("d1", "classification")]]}

    result, trace = trace_pipeline(DOC, MockLLMClient(scripts))

    assert result.status is PipelineStatus.COMPLETED
    assert trace.status is TraceStatus.DEGRADED
    assert trace.status_reasons == ["classification: needed 1 repair attempt(s)"]


def test_spans_capture_inputs_outputs_prompts_and_llm_calls() -> None:
    _, trace = trace_pipeline(DOC, MockLLMClient(SCRIPTS))
    intake, extraction, classification, _ = trace.spans

    assert intake.input["doc_id"] == "d1"
    assert intake.output["text"] == DOC.content
    assert intake.llm_calls == []
    assert intake.prompt_name is None

    assert extraction.input == intake.output
    assert extraction.output["organizations"] == [{"name": "Acme", "role": None}]
    assert (extraction.prompt_name, extraction.prompt_version) == ("extraction", "1.1.0")
    call = extraction.llm_calls[0]
    assert call.attempt == 1
    assert [m.role.value for m in call.messages] == ["system", "user"]
    assert DOC.content in call.messages[1].content
    assert json.loads(call.raw_response or "") == SCRIPTS[("d1", "extraction")]
    assert "confidence" not in (extraction.output or {})
    assert call.model == "mock-llm"
    assert call.prompt_tokens > 0 and call.latency_ms > 0
    assert call.error is None
    assert classification.input["entities"] == extraction.output


def test_common_features_are_logged_for_every_span() -> None:
    _, trace = trace_pipeline(DOC, MockLLMClient(SCRIPTS))
    intake, extraction = trace.spans[0], trace.spans[1]

    assert _as_int(intake.features["input_chars"]) > 0
    assert _as_int(intake.features["output_chars"]) > 0
    assert "llm_attempts" not in intake.features
    assert extraction.features["llm_attempts"] == 1
    assert extraction.features["repair_attempts"] == 0
    assert extraction.features["model"] == "mock-llm"
    assert extraction.features["prompt_version"] == "1.1.0"
    assert extraction.features["prompt_tokens"] == extraction.llm_calls[0].prompt_tokens


def test_failed_step_is_recorded_and_later_steps_are_absent() -> None:
    scripts = {**SCRIPTS, ("d1", "extraction"): "Sorry, I can't do that."}

    result, trace = trace_pipeline(DOC, MockLLMClient(scripts))

    assert result.status is PipelineStatus.FAILED
    assert trace.status is TraceStatus.FAILURE
    assert trace.error_step is StepName.EXTRACTION
    assert [s.name for s in trace.spans] == ["intake", "extraction"]
    failed = trace.spans[1]
    assert failed.status is SpanStatus.ERROR
    assert failed.error is not None and failed.error.error_type == "LLMOutputError"
    assert failed.output is None
    assert [c.raw_response for c in failed.llm_calls] == ["Sorry, I can't do that."] * 2
    assert all(c.error is not None for c in failed.llm_calls)
    assert "extraction failed: LLMOutputError" in trace.status_reasons
    assert trace.final_output is None


def test_provider_error_is_recorded_without_a_response() -> None:
    scripts = {k: v for k, v in SCRIPTS.items() if k[1] != "classification"}

    _, trace = trace_pipeline(DOC, MockLLMClient(scripts))

    call = trace.spans[2].llm_calls[0]
    assert call.raw_response is None
    assert call.error is not None and call.error.startswith("MockScriptMissingError")


def test_span_context_manager_records_errors_and_reraises() -> None:
    with (
        start_trace("d1", model="m") as recorder,
        pytest.raises(ValueError, match="boom"),
        span("custom"),
    ):
        raise ValueError("boom")

    (recorded,) = recorder.spans
    assert recorded.status is SpanStatus.ERROR
    assert recorded.error is not None
    assert (recorded.error.error_type, recorded.error.message) == ("ValueError", "boom")


def test_nested_spans_point_to_their_parent() -> None:
    with start_trace("d1", model="m") as recorder, span("outer") as outer, span("inner"):
        pass

    inner, outer_span = recorder.spans
    assert outer is not None
    assert inner.parent_span_id == outer.span_id == outer_span.span_id
    assert outer_span.parent_span_id is None


def test_feature_extractor_output_is_merged_and_its_bugs_are_contained() -> None:
    @traced_step("good", features=lambda i, o: {"doubled": o})
    def good(x: int) -> int:
        return x * 2

    @traced_step("bad", features=lambda i, o: {"ratio": 1 / 0})
    def bad(x: int) -> int:
        return x

    with start_trace("d1", model="m") as recorder:
        assert good(2) == 4
        assert bad(3) == 3

    assert recorder.spans[0].features["doubled"] == 4
    assert recorder.spans[1].features["feature_error"] == "ZeroDivisionError"


def test_log_lines_inside_a_trace_carry_the_trace_id(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", LogFormat.JSON)

    with start_trace("d1", model="m") as recorder:
        get_logger("test").info("inside_trace")
    get_logger("test").info("outside_trace")

    inside, outside = (json.loads(line) for line in capsys.readouterr().err.strip().splitlines())
    assert inside["trace_id"] == recorder.trace_id
    assert "trace_id" not in outside
