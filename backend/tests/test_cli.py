"""Tests for the pipeline CLI (always with mock settings, never the developer's .env)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.core.config import LLMProvider, Settings
from app.llm.base import LLMProviderError, LLMRequest, LLMResponse
from app.llm.mock import MockLLMClient
from app.pipeline import cli


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch: pytest.MonkeyPatch, data_dir: Path, tmp_path: Path) -> Settings:
    test_settings = Settings(
        _env_file=None,
        data_dir=data_dir,
        traces_dir=tmp_path / "traces",
        database_path=tmp_path / "traces.db",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: test_settings)
    return test_settings


def test_list_prints_every_document(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list"]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("DOC_ID")
    assert len(lines) == 33
    assert any(
        "contract_no_dates_04" in line and "extraction_hallucination" in line for line in lines
    )


def test_run_one_prints_result_json_and_saves_a_trace(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["run", "invoice_simple_06"]) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "completed"
    assert result["summary"]["total_amount"]["raw"] == "$1,284.50"
    assert "success (score 5)" in captured.err
    (saved,) = (tmp_path / "traces").glob("*.json")
    assert saved.stem in captured.err


def test_run_one_reports_why_a_trace_is_degraded(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["run", "ambiguous_amendment_letter_20"])

    assert "degraded (score 2): classification: low confidence (2/5)" in capsys.readouterr().err


def test_run_all_prints_table_and_saves_every_trace(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["run", "--all"]) == 1  # the corpus has one designed crash

    out = capsys.readouterr().out
    assert "32 documents, 1 pipeline error(s), 3 degraded" in out
    assert out.count("type mismatch") == 4
    assert "input truncated" in out
    assert "extraction: LLMOutputError" in out
    assert len(list((tmp_path / "traces").glob("*.json"))) == 32


def test_traces_list_filters_by_status(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["run", "--all"])
    capsys.readouterr()

    assert cli.main(["traces", "list", "--status", "degraded"]) == 0

    out = capsys.readouterr().out
    assert "3 of 3 trace(s)" in out
    assert "report_supplier_risk_long_15" in out
    assert "ambiguous_amendment_letter_20" in out
    assert "contract_long_liability_29" in out


def test_traces_list_by_document_with_limit(capsys: pytest.CaptureFixture[str]) -> None:
    for _ in range(2):
        cli.main(["run", "invoice_simple_06"])
    capsys.readouterr()

    cli.main(["traces", "list", "--doc", "invoice_simple_06", "--limit", "1"])

    assert "1 of 2 trace(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("limit", "message"), [("0", "must be 1 or more"), ("-3", "must be 1 or more"), ("x", "whole")]
)
def test_traces_list_rejects_a_bad_limit(
    capsys: pytest.CaptureFixture[str], limit: str, message: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["traces", "list", "--limit", limit])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_traces_show_prints_one_trace(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cli.main(["run", "invoice_simple_06"])
    capsys.readouterr()
    (saved,) = (tmp_path / "traces").glob("*.json")

    assert cli.main(["traces", "show", saved.stem]) == 0

    trace = json.loads(capsys.readouterr().out)
    assert trace["trace_id"] == saved.stem
    assert len(trace["spans"]) == 4


def test_traces_show_unknown_id_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["traces", "show", "0" * 32]) == 1

    assert "no trace with id" in capsys.readouterr().err


def test_provider_override_is_applied() -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY is required"):
        cli.main(["--provider", "openai", "run", "invoice_simple_06"])


def test_note_reports_step_errors() -> None:
    from app.pipeline.models import PipelineResult, PipelineStatus, StepError, StepName

    failed = PipelineResult(
        doc_id="x",
        status=PipelineStatus.FAILED,
        error=StepError(step=StepName.EXTRACTION, error_type="LLMOutputError", message="m"),
    )

    assert cli._note(failed, "invoice") == "extraction: LLMOutputError"
    assert cli._classified(failed) == "-"


# --------------------------------------------------------------------------- analyze / evaluate


class _FakeRealJudge(MockLLMClient):
    """The pretend judge's verdicts under a non-mock name: counts as a real judge, no network."""

    @property
    def model_name(self) -> str:
        return "fake-real-model"


def _use_real_judge(monkeypatch: pytest.MonkeyPatch, settings: Settings, data_dir: Path) -> None:
    real = settings.model_copy(
        update={"judge_provider": LLMProvider.OPENAI, "judge_api_key": SecretStr("k")}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: real)
    monkeypatch.setattr(
        cli,
        "build_judge_client",
        lambda s: _FakeRealJudge(scripts_dir=data_dir / "mock_judge"),
    )


def test_analyze_the_latest_trace_of_a_document(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["run", "contract_no_dates_04"])
    capsys.readouterr()

    assert cli.main(["analyze", "--doc", "contract_no_dates_04"]) == 0

    captured = capsys.readouterr()
    assert captured.out.startswith("Root cause: Step 2 (Extraction), Extraction Hallucination.")
    assert "extraction      root_cause  1      extraction_hallucination" in captured.out
    assert "[FLAG] ungrounded_entities (extraction)" in captured.out
    assert cli.SCRIPTED_BANNER in captured.out
    assert "saved to" in captured.err


def test_analyze_by_trace_id_as_json(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cli.main(["run", "invoice_simple_06"])
    capsys.readouterr()
    (saved,) = (tmp_path / "traces").glob("*.json")

    assert cli.main(["analyze", saved.stem, "--json"]) == 0

    analysis = json.loads(capsys.readouterr().out)
    assert analysis["trace_id"] == saved.stem
    assert analysis["root_cause_step"] is None


def test_analyze_unknown_trace_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", "--doc", "never_ran"]) == 1

    assert "no saved trace" in capsys.readouterr().err


def test_evaluate_practice_with_the_scripted_judge(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["evaluate"]) == 0

    out = capsys.readouterr().out
    assert out.startswith(cli.SCRIPTED_BANNER)
    assert out.count("practice ") == 21
    assert "right step               8/8 (100%; 95% range 68%-100%)" in out
    assert "false alarms             0/13 (0%; 95% range 0%-23%)" in out
    assert "Cost: 21 judge call(s), 0 verdict(s) reused" in out
    assert out.count(cli.SCRIPTED_BANNER) == 2  # above the table and under the report card
    (report,) = (tmp_path / "traces" / "evaluations").glob("*-practice.json")
    assert report.stat().st_size > 0


def test_evaluate_selected_documents_and_reuse_on_the_second_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        "evaluate",
        "--split",
        "all",
        "--docs",
        "report_broken_answer_28,invoice_two_currencies_30",
    ]
    cli.main(args)
    capsys.readouterr()

    assert cli.main(args) == 0

    out = capsys.readouterr().out
    assert "report_broken_answer_28" in out and "invoice_two_currencies_30" in out
    assert "extraction/prompt" in out
    assert "Cost: 0 judge call(s), 2 verdict(s) reused" in out


def test_evaluate_rejects_unknown_documents(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["evaluate", "--docs", "report_growth_rate_31"]) == 1  # an exam document

    assert "not in the practice documents: report_growth_rate_31" in capsys.readouterr().err


def test_a_real_judge_needs_a_call_limit(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _mock_settings: Settings,
    data_dir: Path,
) -> None:
    _use_real_judge(monkeypatch, _mock_settings, data_dir)

    assert cli.main(["evaluate"]) == 1

    assert "a real judge needs --max-calls N" in capsys.readouterr().err


def test_a_real_judge_stops_at_the_call_limit(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _mock_settings: Settings,
    data_dir: Path,
) -> None:
    _use_real_judge(monkeypatch, _mock_settings, data_dir)

    assert cli.main(["evaluate", "--max-calls", "2"]) == 2

    captured = capsys.readouterr()
    assert "Real judge fake-real-model" in captured.err
    assert "at most 2 new request(s)" in captured.err
    assert cli.SCRIPTED_BANNER not in captured.out
    assert captured.out.count("(call limit reached)") == 19
    assert "19 document(s) not judged" in captured.out


def test_judge_overrides_are_applied(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Settings] = []

    def fake_factory(settings: Settings) -> MockLLMClient:
        seen.append(settings)
        return MockLLMClient(scripts_dir=settings.data_dir / "mock_judge")

    monkeypatch.setattr(cli, "build_judge_client", fake_factory)

    cli.main(["--judge-model", "gemini-3.8-flash", "evaluate", "--docs", "invoice_simple_06"])
    cli.main(["--judge-provider", "mock", "evaluate", "--docs", "invoice_simple_06"])

    assert seen[0].judge_model == "gemini-3.8-flash"
    assert seen[1].judge_provider is LLMProvider.MOCK


def test_analyze_reports_a_judge_that_gives_no_usable_verdict(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.main(["run", "invoice_simple_06"])
    capsys.readouterr()
    monkeypatch.setattr(
        cli,
        "build_judge_client",
        lambda s: MockLLMClient({("invoice_simple_06", "judge"): "not json"}),
    )

    assert cli.main(["analyze", "--doc", "invoice_simple_06"]) == 1

    assert "the judge gave no usable verdict" in capsys.readouterr().err


def test_analyze_with_a_real_judge_announces_it_and_shows_no_scripted_banner(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _mock_settings: Settings,
    data_dir: Path,
) -> None:
    cli.main(["run", "invoice_simple_06"])
    capsys.readouterr()
    _use_real_judge(monkeypatch, _mock_settings, data_dir)

    assert cli.main(["analyze", "--doc", "invoice_simple_06"]) == 0

    captured = capsys.readouterr()
    assert "Real judge fake-real-model" in captured.err
    assert "at most 2 request(s)" in captured.err
    assert cli.SCRIPTED_BANNER not in captured.out


class _QuotaUsedUp(MockLLMClient):
    @property
    def model_name(self) -> str:
        return "fake-real-model"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError("429 per day", status_code=429, quota_exhausted=True)


def test_evaluate_stops_early_when_the_daily_quota_is_used_up(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _mock_settings: Settings,
    data_dir: Path,
) -> None:
    _use_real_judge(monkeypatch, _mock_settings, data_dir)
    monkeypatch.setattr(cli, "build_judge_client", lambda s: _QuotaUsedUp())

    assert cli.main(["evaluate", "--max-calls", "5"]) == 2

    out = capsys.readouterr().out
    assert "STOPPED EARLY: daily quota used up." in out
    assert "again after the provider's daily reset" in out
    assert out.count("(not reached (run stopped)") == 20


class _Overloaded(_QuotaUsedUp):
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError("InternalServerError: 503 high demand", status_code=503)


def test_evaluate_after_an_overloaded_model_suggests_trying_again_later(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _mock_settings: Settings,
    data_dir: Path,
) -> None:
    _use_real_judge(monkeypatch, _mock_settings, data_dir)
    monkeypatch.setattr(cli, "build_judge_client", lambda s: _Overloaded())

    assert cli.main(["evaluate", "--max-calls", "5"]) == 2

    out = capsys.readouterr().out
    assert "STOPPED EARLY: InternalServerError: 503 high demand." in out
    assert "again later (an overloaded model usually recovers within minutes)" in out
