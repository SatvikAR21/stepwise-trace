"""Application settings loaded from environment variables and the repo-root ``.env`` file."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class AppEnv(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """Log renderer: human-friendly console output or one JSON object per line."""

    CONSOLE = "console"
    JSON = "json"


class LLMProvider(StrEnum):
    """Which LLM client to use. ``openai`` covers any OpenAI-compatible endpoint."""

    MOCK = "mock"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Typed application configuration. Every field can be set via an env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE

    llm_provider: LLMProvider = LLMProvider.MOCK
    llm_api_key: SecretStr | None = None
    llm_model: str = "gemini-2.5-flash"
    llm_base_url: str | None = None

    traces_dir: Path = Path("traces")
    database_path: Path = Path("backend/failure_forensics.db")

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(f"invalid log level: {value!r}")
        return level

    @field_validator("llm_api_key", "llm_base_url", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("traces_dir", "database_path")
    @classmethod
    def _resolve_against_repo_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    def require_llm_api_key(self) -> str:
        """Return the API key for a real provider, or raise if it is missing."""
        if self.llm_api_key is None:
            raise RuntimeError("LLM_API_KEY is required when LLM_PROVIDER is not 'mock'")
        return self.llm_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (reset in tests via ``cache_clear``)."""
    return Settings()
