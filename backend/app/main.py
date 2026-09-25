"""FastAPI application factory and ASGI entrypoint (``uvicorn app.main:app``)."""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Pass ``settings`` to override env-derived config (e.g. in tests)."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_format)

    app = FastAPI(
        title="Failure Forensics",
        version=__version__,
        description="Tracing and root-cause analysis for multi-step AI document pipelines.",
    )
    app.dependency_overrides[get_settings] = lambda: resolved
    app.include_router(health_router)

    get_logger(__name__).info(
        "app_created", env=resolved.app_env.value, llm_provider=resolved.llm_provider.value
    )
    return app


app = create_app()
