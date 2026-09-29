"""Tests for parsing LLM text into validated models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from app.llm.base import LLMOutputError
from app.llm.structured import extract_json_text, parse_json_output, parse_scored_output


class _Answer(BaseModel):
    label: str
    score: int


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Sure! Here is the JSON:\n{"a": 1}\nHope that helps.', '{"a": 1}'),
        ("  no json here  ", "no json here"),
    ],
)
def test_extract_json_text(raw: str, expected: str) -> None:
    assert extract_json_text(raw) == expected


def test_parse_valid_output() -> None:
    assert parse_json_output('```json\n{"label": "x", "score": 3}\n```', _Answer) == _Answer(
        label="x", score=3
    )


def test_invalid_json_raises_output_error_with_raw_text() -> None:
    with pytest.raises(LLMOutputError, match="does not match _Answer") as info:
        parse_json_output("I cannot help with that.", _Answer)

    assert info.value.raw_output == "I cannot help with that."


def test_schema_mismatch_names_the_bad_field() -> None:
    with pytest.raises(LLMOutputError, match="score"):
        parse_json_output('{"label": "x", "score": "high"}', _Answer)


# --------------------------------------------------------------------------- scored answers


def test_scored_output_splits_off_a_valid_confidence() -> None:
    scored = parse_scored_output('{"label": "x", "confidence": 4}', _Strict)

    assert scored.value == _Strict(label="x")
    assert (scored.confidence, scored.confidence_note) == (4, None)


@pytest.mark.parametrize(
    ("confidence_json", "expected"),
    [
        ("4.0", (4, None)),
        ('"3"', (3, None)),
        ("null", (None, "missing")),
        ("7", (None, "invalid: 7")),
        ("0", (None, "invalid: 0")),
        ('"high"', (None, "invalid: 'high'")),
        ("true", (None, "invalid: True")),
        ("4.5", (None, "invalid: 4.5")),
    ],
)
def test_confidence_is_normalized_or_rejected_without_rejecting_the_answer(
    confidence_json: str, expected: tuple[int | None, str | None]
) -> None:
    scored = parse_scored_output(f'{{"label": "x", "confidence": {confidence_json}}}', _Strict)

    assert scored.value.label == "x"
    assert (scored.confidence, scored.confidence_note) == expected


def test_missing_confidence_key_is_noted() -> None:
    scored = parse_scored_output('{"label": "x"}', _Strict)

    assert (scored.confidence, scored.confidence_note) == (None, "missing")


def test_scored_output_still_rejects_unknown_fields_and_bad_json() -> None:
    with pytest.raises(LLMOutputError, match="does not match _Strict: extra"):
        parse_scored_output('{"label": "x", "confidence": 5, "extra": 1}', _Strict)
    with pytest.raises(LLMOutputError, match="does not match _Strict"):
        parse_scored_output("no JSON here", _Strict)
