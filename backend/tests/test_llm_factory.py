"""Tests for choosing the LLM client from settings."""

from __future__ import annotations

import pytest

from app.core.config import LLMProvider, Settings
from app.llm.factory import build_llm_client
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
