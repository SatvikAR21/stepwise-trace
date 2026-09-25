"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging

import pytest
import structlog

from app.core.config import LogFormat
from app.core.logging import configure_logging, get_logger


def test_json_format_emits_one_json_object_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", LogFormat.JSON)

    get_logger("test").info("step_finished", step="extraction", latency_ms=42)

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "step_finished"
    assert record["step"] == "extraction"
    assert record["latency_ms"] == 42
    assert record["level"] == "info"
    assert "timestamp" in record


def test_context_vars_are_merged(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", LogFormat.JSON)

    with structlog.contextvars.bound_contextvars(trace_id="abc123"):
        get_logger("test").info("inside_trace")

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["trace_id"] == "abc123"


def test_level_filters_lower_messages(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", LogFormat.JSON)

    get_logger("test").info("hidden")
    get_logger("test").warning("shown")

    err = capsys.readouterr().err
    assert "hidden" not in err
    assert "shown" in err


def test_stdlib_loggers_use_same_format(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", LogFormat.JSON)

    logging.getLogger("uvicorn.error").info("from uvicorn")

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "from uvicorn"


def test_console_format_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", LogFormat.CONSOLE)

    get_logger("test").info("hello_console", key="value")

    err = capsys.readouterr().err
    assert "hello_console" in err
    assert "key" in err


def test_reconfiguring_does_not_duplicate_handlers() -> None:
    configure_logging("INFO", LogFormat.JSON)
    configure_logging("INFO", LogFormat.JSON)

    assert len(logging.getLogger().handlers) == 1
