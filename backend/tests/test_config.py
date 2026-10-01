"""Tests for settings loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import (
    REPO_ROOT,
    AppEnv,
    LLMProvider,
    LogFormat,
    Settings,
    get_settings,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults_are_safe_and_free() -> None:
    settings = _settings()

    assert settings.app_env is AppEnv.DEVELOPMENT
    assert settings.llm_provider is LLMProvider.MOCK
    assert settings.llm_api_key is None
    assert settings.log_format is LogFormat.CONSOLE
    assert settings.judge_provider is LLMProvider.MOCK
    assert settings.judge_max_retries == 0  # no hidden retries against a free quota
    assert settings.judge_max_rpm == 4  # below the Gemini free tier's 5 per minute
    assert settings.judge_drop_score == 2


def test_judge_key_and_url_fall_back_to_the_llm_ones() -> None:
    settings = _settings(llm_api_key="llm-key", llm_base_url="https://llm.example.test/v1")

    assert settings.require_judge_api_key() == "llm-key"
    assert settings.judge_endpoint == "https://llm.example.test/v1"

    own = _settings(
        llm_api_key="llm-key", judge_api_key="judge-key", judge_base_url="https://j.example.test"
    )
    assert own.require_judge_api_key() == "judge-key"
    assert own.judge_endpoint == "https://j.example.test"


def test_require_judge_api_key_raises_when_no_key_at_all() -> None:
    with pytest.raises(RuntimeError, match="JUDGE_API_KEY or LLM_API_KEY is required"):
        _settings().require_judge_api_key()


def test_judge_drop_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        _settings(judge_drop_score=5)


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("LOG_FORMAT", "json")

    settings = _settings()

    assert settings.llm_provider is LLMProvider.OPENAI
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.log_format is LogFormat.JSON


def test_reads_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("APP_ENV=production\nLLM_API_KEY=sk-from-file\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.app_env is AppEnv.PRODUCTION
    assert settings.require_llm_api_key() == "sk-from-file"


def test_repair_attempts_default_to_one_and_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings().llm_max_repair_attempts == 1
    monkeypatch.setenv("LLM_MAX_REPAIR_ATTEMPTS", "0")
    assert _settings().llm_max_repair_attempts == 0
    with pytest.raises(ValidationError):
        _settings(llm_max_repair_attempts=4)


def test_log_level_is_normalized() -> None:
    assert _settings(log_level=" debug ").log_level == "DEBUG"


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid log level"):
        _settings(log_level="LOUD")


def test_invalid_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(llm_provider="not-a-provider")


def test_blank_optional_strings_become_none() -> None:
    settings = _settings(llm_api_key="  ", llm_base_url="", judge_api_key=" ", judge_base_url="")

    assert settings.llm_api_key is None
    assert settings.llm_base_url is None
    assert settings.judge_api_key is None
    assert settings.judge_base_url is None


def test_relative_paths_resolve_against_repo_root() -> None:
    settings = _settings(traces_dir="traces")

    assert settings.traces_dir == (REPO_ROOT / "traces").resolve()


def test_absolute_paths_kept(tmp_path: Path) -> None:
    assert _settings(database_path=tmp_path / "x.db").database_path == tmp_path / "x.db"


def test_api_key_is_masked_in_repr() -> None:
    settings = _settings(llm_api_key="sk-super-secret")

    assert "sk-super-secret" not in repr(settings)
    assert settings.require_llm_api_key() == "sk-super-secret"


def test_require_api_key_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY is required"):
        _settings().require_llm_api_key()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
