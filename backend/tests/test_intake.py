"""Tests for Step 1 (Intake)."""

from __future__ import annotations

import pytest

from app.pipeline.errors import IntakeError
from app.pipeline.models import DocumentFormat, RawDocument
from app.pipeline.steps.intake import run_intake


def _doc(content: str, fmt: DocumentFormat = DocumentFormat.TEXT) -> RawDocument:
    return RawDocument(doc_id="d1", content=content, format=fmt)


def test_normalizes_whitespace_and_line_endings() -> None:
    result = run_intake(_doc("  Hello\t\tworld  \r\n\r\n\r\n\r\nBye   now "), max_chars=1000)

    assert result.text == "Hello world\n\nBye now"
    assert result.word_count == 4
    assert result.char_count == len(result.text)
    assert result.truncated is False
    assert result.cleanup_notes == []


def test_pdf_cleanup_removes_page_markers_form_feeds_and_hyphen_breaks() -> None:
    content = "Page 1 of 2\nThe agree-\nment starts.\f- 2 -\nPAGE 2 OF 2\nEnd."

    result = run_intake(_doc(content, DocumentFormat.PDF_TEXT), max_chars=1000)

    assert result.text == "The agreement starts.\n\nEnd."
    assert result.cleanup_notes == [
        "replaced 1 form feed(s)",
        "removed 3 page marker line(s)",
        "rejoined 1 hyphenated line break(s)",
    ]


def test_pdf_cleanup_not_applied_to_plain_text() -> None:
    result = run_intake(_doc("Page 1 of 2\nagree-\nment"), max_chars=1000)

    assert result.text == "Page 1 of 2\nagree-\nment"


def test_truncates_long_documents_and_records_it() -> None:
    result = run_intake(_doc("word " * 100), max_chars=50)

    assert result.truncated is True
    assert result.char_count <= 50
    assert result.original_char_count == 500
    assert result.cleanup_notes == [f"truncated from 499 to {result.char_count} characters"]


@pytest.mark.parametrize("content", [" ", "\n\n\t", "Page 3 of 9"])
def test_empty_document_raises(content: str) -> None:
    with pytest.raises(IntakeError, match="empty"):
        run_intake(_doc(content, DocumentFormat.PDF_TEXT), max_chars=100)
