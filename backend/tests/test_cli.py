"""Tests for the pipeline CLI (always with mock settings, never the developer's .env)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.pipeline import cli


@pytest.fixture(autouse=True)
def _mock_settings(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    test_settings = Settings(_env_file=None, data_dir=data_dir)
    monkeypatch.setattr(cli, "get_settings", lambda: test_settings)


def test_list_prints_every_document(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list"]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("DOC_ID")
    assert len(lines) == 22
    assert any(
        "contract_no_dates_04" in line and "extraction_hallucination" in line for line in lines
    )


def test_run_one_prints_pipeline_result_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "invoice_simple_06"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["summary"]["total_amount"]["raw"] == "$1,284.50"


def test_run_all_prints_table_and_flags_mismatches(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--all"]) == 0

    out = capsys.readouterr().out
    assert "21 documents, 0 pipeline error(s)" in out
    assert out.count("type mismatch") == 3
    assert "input truncated" in out


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
