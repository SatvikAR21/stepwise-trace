"""Turn an LLM's text answer into a validated Pydantic model."""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.base import LLMOutputError

ModelT = TypeVar("ModelT", bound=BaseModel)

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.DOTALL)


def extract_json_text(raw: str) -> str:
    """Strip Markdown code fences and any prose around the outermost JSON object."""
    text = raw.strip()
    fence = _CODE_FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return text


def parse_json_output(raw: str, model: type[ModelT]) -> ModelT:
    """Parse ``raw`` into ``model``. Raises ``LLMOutputError`` (keeping the raw text) on failure."""
    try:
        return model.model_validate_json(extract_json_text(raw))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        raise LLMOutputError(
            f"LLM output does not match {model.__name__}: {problems}", raw_output=raw
        ) from exc
