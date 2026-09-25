"""Liveness endpoint."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.core.config import AppEnv, LLMProvider, Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Payload returned by ``GET /health``."""

    status: Literal["ok"]
    version: str
    env: AppEnv
    llm_provider: LLMProvider


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the service is up, with non-secret runtime info."""
    return HealthResponse(
        status="ok",
        version=__version__,
        env=settings.app_env,
        llm_provider=settings.llm_provider,
    )
