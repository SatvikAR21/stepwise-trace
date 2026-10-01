"""Tests for grading the analyzer against the answer key (no network: fake judges only)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.analysis.evaluation import (
    EvaluationReport,
    Outcome,
    StopReason,
    evaluate_corpus,
    save_report,
    wilson_interval,
)
from app.analysis.judge import JUDGE_STEP
from app.analysis.store import AnalysisStore
from app.core.config import BACKEND_DIR
from app.llm.base import META_DOC_ID, LLMClient, LLMProviderError, LLMRequest, LLMResponse
from app.llm.mock import MockLLMClient
from app.pipeline.documents import CorpusSplit, ManifestEntry, load_manifest
from app.pipeline.models import PipelineConfig
from app.tracing.store import TraceStore

DATA_DIR = BACKEND_DIR / "data"
MANIFEST = load_manifest(DATA_DIR)
PRACTICE = [e for e in MANIFEST.documents if e.split is CorpusSplit.PRACTICE]


class FakeRealJudge(MockLLMClient):
    """The pretend judge's verdicts under a non-mock model name, so it counts as 'real'."""

    @property
    def model_name(self) -> str:
        return "fake-real-model"


class QuotaAfter(LLMClient):
    """Answers like the pretend judge for ``n`` calls, then reports a used-up daily quota."""

    def __init__(self, n: int) -> None:
        self._inner = MockLLMClient(scripts_dir=DATA_DIR / "mock_judge")
        self.left = n

    @property
    def model_name(self) -> str:
        return "fake-real-model"

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.left == 0:
            raise LLMProviderError("429 per day", status_code=429, quota_exhausted=True)
        self.left -= 1
        return self._inner.complete(request)


@pytest.fixture
def stores(tmp_path: Path) -> Iterator[tuple[TraceStore, AnalysisStore]]:
    traces = TraceStore(tmp_path / "traces", tmp_path / "e.db")
    analyses = AnalysisStore(tmp_path / "traces" / "analyses", tmp_path / "e.db")
    yield traces, analyses
    traces.close()
    analyses.close()


def _evaluate(
    entries: list[ManifestEntry],
    judge: LLMClient,
    stores: tuple[TraceStore, AnalysisStore],
    **kwargs: Any,
) -> EvaluationReport:
    traces, analyses = stores
    return evaluate_corpus(
        entries,
        data_dir=DATA_DIR,
        pipeline_llm=MockLLMClient(scripts_dir=DATA_DIR / "mock_responses"),
        judge=judge,
        trace_store=traces,
        analysis_store=analyses,
        config=PipelineConfig(),
        drop_score=2,
        **kwargs,
    )


def test_the_pretend_judge_scores_perfectly_and_is_marked_as_scripted(
    stores: tuple[TraceStore, AnalysisStore],
) -> None:
    report = _evaluate(PRACTICE, MockLLMClient(scripts_dir=DATA_DIR / "mock_judge"), stores)

    t = report.totals
    assert report.scripted_judge
    assert (t.broken, t.right_step, t.right_category, t.correct) == (8, 8, 8, 8)
    assert (t.healthy, t.false_alarms, t.not_judged) == (13, 0, 0)
    # checks flag 04, 09, 15, 19 (at the planted step) and 21 (the document itself); and #12
    assert (t.checks_caught, t.checks_right_step, t.checks_false_alarms) == (5, 4, 1)
    assert t.judge_calls == 21 and t.reused_verdicts == 0


def test_a_second_run_reuses_every_verdict(stores: tuple[TraceStore, AnalysisStore]) -> None:
    judge = FakeRealJudge(scripts_dir=DATA_DIR / "mock_judge")
    _evaluate(PRACTICE, judge, stores)
    calls_before = len(judge.calls)

    again = _evaluate(PRACTICE, judge, stores, max_calls=1)

    assert len(judge.calls) == calls_before  # not a single new request
    assert again.totals.judge_calls == 0 and again.totals.reused_verdicts == 21
    assert not again.scripted_judge


def test_the_call_limit_is_hard_and_later_documents_are_not_judged(
    stores: tuple[TraceStore, AnalysisStore],
) -> None:
    judge = FakeRealJudge(scripts_dir=DATA_DIR / "mock_judge")

    report = _evaluate(PRACTICE, judge, stores, max_calls=3)

    assert len(judge.calls) == 3
    assert report.totals.judge_calls == 3
    outcomes = [row.outcome for row in report.rows]
    assert Outcome.NOT_JUDGED not in outcomes[:3]
    assert set(outcomes[3:]) == {Outcome.NOT_JUDGED}
    assert report.rows[3].note == "call limit reached"
    assert report.stopped is None


def test_repairs_cannot_push_a_run_past_the_limit(
    stores: tuple[TraceStore, AnalysisStore],
) -> None:
    entry = MANIFEST.get("invoice_simple_06")
    judge = MockLLMClient({("invoice_simple_06", JUDGE_STEP): "not json"})

    report = _evaluate([entry], judge, stores, max_calls=1, max_repair_attempts=1)

    assert len(judge.calls) == 1  # the repair attempt was not sent
    assert report.rows[0].outcome is Outcome.NOT_JUDGED
    assert report.rows[0].note is not None and report.rows[0].note.startswith("unusable answer")


def test_a_used_up_daily_quota_stops_the_run_at_once(
    stores: tuple[TraceStore, AnalysisStore],
) -> None:
    judge = QuotaAfter(2)
    seen: list[str] = []

    report = _evaluate(
        PRACTICE, judge, stores, max_calls=20, on_result=lambda r: seen.append(r.doc_id)
    )

    assert report.stopped is StopReason.QUOTA_EXHAUSTED
    assert report.stop_detail == "daily quota used up"
    assert report.totals.judge_calls == 3  # two answers and the refused third request
    assert len(seen) == 21  # every row is reported, including the refused and unreached ones
    assert [r.note for r in report.rows[3:]] == ["not reached (run stopped)"] * 18


class Overloaded(LLMClient):
    @property
    def model_name(self) -> str:
        return "fake-real-model"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError("InternalServerError: 503 high demand", status_code=503)


def test_an_overloaded_model_stops_the_run_as_a_provider_error(
    stores: tuple[TraceStore, AnalysisStore],
) -> None:
    report = _evaluate(PRACTICE[:2], Overloaded(), stores, max_calls=5)

    assert report.stopped is StopReason.PROVIDER_ERROR
    assert report.stop_detail is not None and "503 high demand" in report.stop_detail
    assert report.totals.judge_calls == 1


def test_a_quota_used_up_from_the_start_judges_nothing(
    stores: tuple[TraceStore, AnalysisStore],
) -> None:
    report = _evaluate(PRACTICE[:3], QuotaAfter(0), stores, max_calls=5)

    assert report.stopped is not None
    assert [r.outcome for r in report.rows] == [Outcome.NOT_JUDGED] * 3
    assert report.totals.judge_calls == 1


def test_outcomes_cover_every_way_a_diagnosis_can_go(
    stores: tuple[TraceStore, AnalysisStore],
) -> None:
    def grades(doc_id: str, root: str | None, category: str | None) -> dict[str, Any]:
        steps = []
        for step in ("intake", "extraction", "classification", "summarization"):
            grade: dict[str, Any] = {"step": step, "score": 5, "explanation": "x"}
            if step == root:
                grade |= {"score": 1, "introduced": ["problem"], "category": category}
            steps.append(grade)
        return {"steps": steps}

    cases = {
        "contract_no_dates_04": (("extraction", "extraction_hallucination"), Outcome.CORRECT),
        "correspondence_unnamed_ceo_19": (("extraction", "context_loss"), Outcome.STEP_ONLY),
        "invoice_multi_currency_09": (("extraction", "context_loss"), Outcome.WRONG),
        "report_expense_14": ((None, None), Outcome.MISSED),
        "invoice_simple_06": ((None, None), Outcome.HEALTHY_OK),
        "report_postmortem_12": (("summarization", "prompt_failure"), Outcome.FALSE_ALARM),
    }
    judge = MockLLMClient(
        {(doc, JUDGE_STEP): grades(doc, *verdict) for doc, (verdict, _) in cases.items()}
    )

    report = _evaluate([MANIFEST.get(doc) for doc in cases], judge, stores)

    assert {r.doc_id: r.outcome for r in report.rows} == {d: o for d, (_, o) in cases.items()}
    t = report.totals
    assert (t.right_step, t.right_category, t.correct, t.missed, t.false_alarms) == (2, 1, 1, 1, 1)


def test_wilson_interval_is_wide_for_few_cases() -> None:
    low, high = wilson_interval(8, 8)
    assert (round(low, 3), high) == (0.676, 1.0)
    low, high = wilson_interval(13, 16)
    assert (round(low, 3), round(high, 3)) == (0.57, 0.934)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_reports_are_saved_with_a_timestamped_name(
    stores: tuple[TraceStore, AnalysisStore], tmp_path: Path
) -> None:
    report = _evaluate(
        [MANIFEST.get("invoice_simple_06")],
        MockLLMClient(scripts_dir=DATA_DIR / "mock_judge"),
        stores,
    )

    path = save_report(report, tmp_path / "evaluations")

    assert path.name.endswith("-practice.json")
    assert EvaluationReport.model_validate_json(path.read_text(encoding="utf-8")) == report


def test_the_judge_request_names_the_document(stores: tuple[TraceStore, AnalysisStore]) -> None:
    judge = MockLLMClient(scripts_dir=DATA_DIR / "mock_judge")

    _evaluate([MANIFEST.get("invoice_simple_06")], judge, stores)

    assert judge.calls[0].metadata[META_DOC_ID] == "invoice_simple_06"
