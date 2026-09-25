"""Tests for parsing LLM text into validated models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.base import LLMOutputError
from app.llm.structured import extract_json_text, parse_json_output


class _Answer(BaseModel):
    label: str
    score: int


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
