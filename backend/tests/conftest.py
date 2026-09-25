"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import AppEnv, Settings, get_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop a developer's real .env or shell env vars from leaking into tests."""
    for var in (
        "APP_ENV",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "TRACES_DIR",
        "DATABASE_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing all storage at a temp dir, ignoring any .env file."""
    return Settings(
        _env_file=None,
        app_env=AppEnv.TEST,
        traces_dir=tmp_path / "traces",
        database_path=tmp_path / "test.db",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """FastAPI test client wired to the temp-dir settings."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
