"""Deterministic, zero-cost LLM client that replays scripted responses.

Scripts are keyed by ``(doc_id, step)`` taken from ``LLMRequest.metadata``. They come either from
an in-memory mapping (handy in tests) or from ``<scripts_dir>/<doc_id>.json`` files shaped like::

    {"extraction": {...json object...}, "classification": {...}, "summarization": "raw text"}

Object values are serialized to JSON; string values are returned verbatim, which lets a script
return deliberately malformed output. Keys starting with ``_`` (e.g. ``_comment``) are ignored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.llm.base import (
    META_DOC_ID,
    META_STEP,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)

MOCK_MODEL_NAME = "mock-llm"
_CHARS_PER_TOKEN = 4


class MockScriptMissingError(LLMError):
    """No scripted response exists for the requested ``(doc_id, step)``."""


class MockLLMClient(LLMClient):
    """Replays scripted responses. Records every request in ``calls`` for test inspection."""

    def __init__(
        self,
        scripts: Mapping[tuple[str, str], Any] | None = None,
        *,
        scripts_dir: Path | None = None,
    ) -> None:
        self._scripts: dict[tuple[str, str], Any] = dict(scripts or {})
        self._scripts_dir = scripts_dir
        self._loaded_docs: set[str] = set()
        self.calls: list[LLMRequest] = []

    @property
    def model_name(self) -> str:
        """The mock's fixed model identifier."""
        return MOCK_MODEL_NAME

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the scripted response for the request's ``doc_id`` and ``step``."""
        self.calls.append(request)
        doc_id = request.metadata.get(META_DOC_ID, "")
        step = request.metadata.get(META_STEP, "")
        content = self._render(self._lookup(doc_id, step))
        prompt_chars = sum(len(message.content) for message in request.messages)
        usage = TokenUsage(
            prompt_tokens=prompt_chars // _CHARS_PER_TOKEN,
            completion_tokens=len(content) // _CHARS_PER_TOKEN,
        )
        # Fake but deterministic latency so dashboards have something realistic to show.
        latency_ms = 40.0 + usage.completion_tokens * 1.5
        return LLMResponse(
            content=content, model=MOCK_MODEL_NAME, usage=usage, latency_ms=latency_ms
        )

    def _lookup(self, doc_id: str, step: str) -> Any:
        key = (doc_id, step)
        if key not in self._scripts:
            self._load_file(doc_id)
        if key not in self._scripts:
            raise MockScriptMissingError(f"no mock script for doc_id={doc_id!r} step={step!r}")
        return self._scripts[key]

    def _load_file(self, doc_id: str) -> None:
        if self._scripts_dir is None or not doc_id or doc_id in self._loaded_docs:
            return
        self._loaded_docs.add(doc_id)
        path = self._scripts_dir / f"{doc_id}.json"
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise MockScriptMissingError(f"mock script {path.name} must be a JSON object")
        for step, response in data.items():
            if not step.startswith("_"):  # "_comment" etc. document the script
                self._scripts.setdefault((doc_id, step), response)

    @staticmethod
    def _render(response: Any) -> str:
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
