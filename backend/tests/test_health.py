"""Tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "env": "test",
        "llm_provider": "mock",
    }


def test_health_does_not_leak_secrets(client: TestClient) -> None:
    body = client.get("/health").text.lower()

    assert "api_key" not in body
    assert "secret" not in body


def test_openapi_schema_lists_health(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/health" in schema["paths"]
