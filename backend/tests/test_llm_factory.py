"""Tests for choosing the LLM client from settings."""

from __future__ import annotations

import pytest

from app.core.config import LLMProvider, Settings
from app.llm.factory import build_judge_client, build_llm_client
from app.llm.mock import MockLLMClient
from app.llm.openai_compatible import OpenAICompatibleClient
from app.llm.throttle import ThrottledLLMClient


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_mock_provider_builds_mock_client() -> None:
    assert isinstance(build_llm_client(_settings()), MockLLMClient)


def test_openai_provider_builds_real_client() -> None:
    client = build_llm_client(
        _settings(
            llm_provider="openai",
            llm_api_key="k",
            llm_model="gemini-3.5-flash",
            llm_base_url="https://llm.example.test/v1",
        )
    )

    assert isinstance(client, OpenAICompatibleClient)
    assert client.model_name == "gemini-3.5-flash"


def test_max_rpm_wraps_the_real_client_in_a_throttle() -> None:
    client = build_llm_client(
        _settings(llm_provider="openai", llm_api_key="k", llm_model="m", llm_max_rpm=5)
    )

    assert isinstance(client, ThrottledLLMClient)
    assert client.model_name == "m"


def test_openai_provider_without_key_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY is required"):
        build_llm_client(_settings(llm_provider=LLMProvider.OPENAI))


# --------------------------------------------------------------------------- the judge

REAL_PIPELINE = {
    "llm_provider": "openai",
    "llm_api_key": "k",
    "llm_model": "gemini-3.5-flash",
    "llm_base_url": "https://llm.example.test/v1",
}


def test_judge_defaults_to_the_scripted_mock() -> None:
    settings = _settings()

    judge = build_judge_client(settings)

    assert isinstance(judge, MockLLMClient)
    assert judge._scripts_dir == settings.data_dir / "mock_judge"


def test_real_judge_reuses_the_llm_key_and_url_is_throttled_and_never_retries_silently() -> None:
    judge = build_judge_client(
        _settings(**REAL_PIPELINE, judge_provider="openai", judge_model="gemini-3.8-flash")
    )

    assert isinstance(judge, ThrottledLLMClient)
    assert judge._max_rpm == 4
    inner = judge._inner
    assert isinstance(inner, OpenAICompatibleClient)
    assert inner.model_name == "gemini-3.8-flash"
    assert inner._client.max_retries == 0
    assert inner._client.api_key == "k"
    assert str(inner._client.base_url).rstrip("/") == "https://llm.example.test/v1"


def test_real_judge_can_have_its_own_key_url_and_no_throttle() -> None:
    judge = build_judge_client(
        _settings(
            judge_provider="openai",
            judge_api_key="judge-key",
            judge_base_url="https://judge.example.test/v1",
            judge_max_rpm=0,
        )
    )

    assert isinstance(judge, OpenAICompatibleClient)
    assert judge._client.api_key == "judge-key"
    assert str(judge._client.base_url).rstrip("/") == "https://judge.example.test/v1"


def test_judge_on_the_pipelines_own_model_shares_its_client_and_rate_limiter() -> None:
    settings = _settings(**REAL_PIPELINE, llm_max_rpm=5, judge_provider="openai")
    pipeline = build_llm_client(settings)

    assert build_judge_client(settings, pipeline_client=pipeline) is pipeline


@pytest.mark.parametrize(
    "difference",
    [
        {"judge_model": "gemini-3.8-flash"},
        {"judge_base_url": "https://other.example.test/v1"},
        {"judge_api_key": "another-key"},
    ],
)
def test_judge_on_a_different_model_endpoint_or_key_gets_its_own_client(
    difference: dict[str, str],
) -> None:
    settings = _settings(**REAL_PIPELINE, judge_provider="openai", **difference)
    pipeline = build_llm_client(settings)

    assert build_judge_client(settings, pipeline_client=pipeline) is not pipeline


def test_real_judge_without_any_key_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="JUDGE_API_KEY or LLM_API_KEY is required"):
        build_judge_client(_settings(judge_provider=LLMProvider.OPENAI))
