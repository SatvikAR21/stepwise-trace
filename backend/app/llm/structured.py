"""Turn an LLM's text answer into a validated Pydantic model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from app.llm.base import LLMOutputError

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.DOTALL)

CONFIDENCE_KEY = "confidence"
CONFIDENCE_MIN, CONFIDENCE_MAX = 1, 5


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


def parse_json_output[ModelT: BaseModel](raw: str, model: type[ModelT]) -> ModelT:
    """Parse ``raw`` into ``model``. Raises ``LLMOutputError`` (keeping the raw text) on failure."""
    try:
        return model.model_validate_json(extract_json_text(raw))
    except ValidationError as exc:
        raise _output_error(raw, model, exc) from exc


@dataclass(frozen=True)
class Scored[ModelT: BaseModel]:
    """A validated answer plus the model's self-reported confidence (1-5), if usable."""

    value: ModelT
    confidence: int | None
    confidence_note: str | None = None


def parse_scored_output[ModelT: BaseModel](raw: str, model: type[ModelT]) -> Scored[ModelT]:
    """Parse an answer that also carries a ``"confidence"`` score from 1 to 5.

    The score is split off before the answer reaches the step's own model, so downstream steps never
    see it. A missing or invalid score does not reject the answer: it comes back as ``None`` with a
    note. Anything else wrong with the answer raises ``LLMOutputError`` as usual.
    """
    try:
        parsed = _with_confidence(model).model_validate_json(extract_json_text(raw))
    except ValidationError as exc:
        raise _output_error(raw, model, exc) from exc
    value = model.model_validate({name: getattr(parsed, name) for name in model.model_fields})
    present = CONFIDENCE_KEY in parsed.model_fields_set
    confidence, note = _normalize_confidence(getattr(parsed, CONFIDENCE_KEY), present=present)
    return Scored(value=value, confidence=confidence, confidence_note=note)


@cache
def _with_confidence(model: type[BaseModel]) -> type[BaseModel]:
    """``model`` plus an optional, loosely typed ``confidence`` field (checked separately)."""
    return create_model(f"{model.__name__}WithConfidence", __base__=model, confidence=(Any, None))


def _normalize_confidence(value: Any, *, present: bool) -> tuple[int | None, str | None]:
    if not present or value is None:
        return None, "missing"
    number: int | None = None
    if isinstance(value, bool):
        number = None
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    if number is None or not CONFIDENCE_MIN <= number <= CONFIDENCE_MAX:
        return None, f"invalid: {value!r}"
    return number, None


def _output_error(raw: str, model: type[BaseModel], exc: ValidationError) -> LLMOutputError:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()[:5]
    )
    return LLMOutputError(f"LLM output does not match {model.__name__}: {problems}", raw_output=raw)
