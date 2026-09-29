"""Tests for the deterministic mock LLM client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm.base import META_ATTEMPT, META_DOC_ID, META_STEP, ChatMessage, LLMRequest, Role
from app.llm.mock import MOCK_MODEL_NAME, MockLLMClient, MockScriptMissingError


def _request(
    doc_id: str = "doc1", step: str = "extraction", text: str = "x" * 40, attempt: int = 1
) -> LLMRequest:
    return LLMRequest(
        messages=[ChatMessage(role=Role.USER, content=text)],
        metadata={META_DOC_ID: doc_id, META_STEP: step, META_ATTEMPT: str(attempt)},
    )


def test_list_script_answers_per_attempt_and_repeats_the_last() -> None:
    llm = MockLLMClient({("doc1", "extraction"): ["bad", {"key_terms": []}]})

    answers = [llm.complete(_request(attempt=n)).content for n in (1, 2, 3)]

    assert answers == ["bad", '{"key_terms": []}', '{"key_terms": []}']


def test_empty_script_list_is_an_error() -> None:
    with pytest.raises(MockScriptMissingError, match="empty"):
        MockLLMClient({("doc1", "extraction"): []}).complete(_request())


def test_returns_in_memory_script_as_json() -> None:
    llm = MockLLMClient({("doc1", "extraction"): {"key_terms": ["a"]}})

    response = llm.complete(_request())

    assert json.loads(response.content) == {"key_terms": ["a"]}
    assert response.model == MOCK_MODEL_NAME == llm.model_name


def test_string_script_returned_verbatim() -> None:
    llm = MockLLMClient({("doc1", "extraction"): "not json at all"})

    assert llm.complete(_request()).content == "not json at all"


def test_usage_and_latency_are_deterministic() -> None:
    llm = MockLLMClient({("doc1", "extraction"): "y" * 80})

    first, second = llm.complete(_request()), llm.complete(_request())

    assert first == second
    assert first.usage.prompt_tokens == 10
    assert first.usage.completion_tokens == 20
    assert first.usage.total_tokens == 30
    assert first.latency_ms == 70.0


def test_records_calls() -> None:
    llm = MockLLMClient({("doc1", "extraction"): {}})

    llm.complete(_request())

    assert len(llm.calls) == 1
    assert llm.calls[0].metadata[META_STEP] == "extraction"


def test_missing_script_raises() -> None:
    with pytest.raises(MockScriptMissingError, match="doc_id='nope'"):
        MockLLMClient().complete(_request(doc_id="nope"))


def test_loads_scripts_from_directory_and_ignores_comment_keys(tmp_path: Path) -> None:
    (tmp_path / "doc1.json").write_text(
        json.dumps({"_comment": "why", "extraction": {"key_terms": []}}), encoding="utf-8"
    )
    llm = MockLLMClient(scripts_dir=tmp_path)

    assert json.loads(llm.complete(_request()).content) == {"key_terms": []}
    with pytest.raises(MockScriptMissingError):
        llm.complete(_request(step="_comment"))


def test_in_memory_script_overrides_file(tmp_path: Path) -> None:
    (tmp_path / "doc1.json").write_text(json.dumps({"extraction": "file"}), encoding="utf-8")
    llm = MockLLMClient({("doc1", "extraction"): "memory"}, scripts_dir=tmp_path)

    assert llm.complete(_request()).content == "memory"


def test_non_object_script_file_rejected(tmp_path: Path) -> None:
    (tmp_path / "doc1.json").write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(MockScriptMissingError, match="must be a JSON object"):
        MockLLMClient(scripts_dir=tmp_path).complete(_request())


def test_missing_file_in_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(MockScriptMissingError):
        MockLLMClient(scripts_dir=tmp_path).complete(_request(doc_id="absent"))
