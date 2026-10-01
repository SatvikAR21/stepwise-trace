"""Tests for the pipeline CLI (always with mock settings, never the developer's .env)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
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
