"""Build the configured ``LLMClient``."""

from __future__ import annotations

from app.core.config import LLMProvider, Settings
from app.llm.base import LLMClient
from app.llm.mock import MockLLMClient
from app.llm.openai_compatible import OpenAICompatibleClient

MOCK_SCRIPTS_SUBDIR = "mock_responses"


def build_llm_client(settings: Settings) -> LLMClient:
    """Return a mock client for ``LLM_PROVIDER=mock``, else an OpenAI-compatible client."""
    if settings.llm_provider is LLMProvider.MOCK:
        return MockLLMClient(scripts_dir=settings.data_dir / MOCK_SCRIPTS_SUBDIR)
    return OpenAICompatibleClient(
        api_key=settings.require_llm_api_key(),
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
