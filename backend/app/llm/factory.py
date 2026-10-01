"""Build the configured ``LLMClient`` for the pipeline and for the root-cause judge."""

from __future__ import annotations

from app.core.config import LLMProvider, Settings
from app.llm.base import LLMClient
from app.llm.mock import MockLLMClient
from app.llm.openai_compatible import OpenAICompatibleClient
from app.llm.throttle import ThrottledLLMClient

MOCK_SCRIPTS_SUBDIR = "mock_responses"
MOCK_JUDGE_SUBDIR = "mock_judge"


def build_llm_client(settings: Settings) -> LLMClient:
    """Return a mock client for ``LLM_PROVIDER=mock``, else an OpenAI-compatible client.

    With ``LLM_MAX_RPM`` > 0 the real client is wrapped in a throttle that respects the limit.
    """
    if settings.llm_provider is LLMProvider.MOCK:
        return MockLLMClient(scripts_dir=settings.data_dir / MOCK_SCRIPTS_SUBDIR)
    client: LLMClient = OpenAICompatibleClient(
        api_key=settings.require_llm_api_key(),
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    if settings.llm_max_rpm > 0:
        client = ThrottledLLMClient(client, max_rpm=settings.llm_max_rpm)
    return client


def build_judge_client(
    settings: Settings, *, pipeline_client: LLMClient | None = None
) -> LLMClient:
    """Return the judge's client: scripted verdicts for ``JUDGE_PROVIDER=mock``, else a real one.

    The real judge makes no hidden SDK retries by default (``JUDGE_MAX_RETRIES=0``) and is
    throttled to ``JUDGE_MAX_RPM``. If it would call the same real model at the same endpoint with
    the same key as ``pipeline_client``, that client is returned instead, so both share one rate
    limiter and cannot exceed the provider's per-model quota together.
    """
    if settings.judge_provider is LLMProvider.MOCK:
        return MockLLMClient(scripts_dir=settings.data_dir / MOCK_JUDGE_SUBDIR)
    if pipeline_client is not None and _judge_uses_pipeline_endpoint(settings):
        return pipeline_client
    client: LLMClient = OpenAICompatibleClient(
        api_key=settings.require_judge_api_key(),
        model=settings.judge_model,
        base_url=settings.judge_endpoint,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.judge_max_retries,
    )
    if settings.judge_max_rpm > 0:
        client = ThrottledLLMClient(client, max_rpm=settings.judge_max_rpm)
    return client


def _judge_uses_pipeline_endpoint(settings: Settings) -> bool:
    return (
        settings.llm_provider is LLMProvider.OPENAI
        and settings.judge_model == settings.llm_model
        and settings.judge_endpoint == settings.llm_base_url
        and settings.judge_api_key in (None, settings.llm_api_key)
    )
