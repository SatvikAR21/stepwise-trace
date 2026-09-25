"""Step 1 (Intake): turn a raw document into clean, bounded text. No LLM involved."""

from __future__ import annotations

import re

from app.pipeline.errors import IntakeError
from app.pipeline.models import DocumentFormat, NormalizedDocument, RawDocument

# Lines that are only page furniture from PDF-to-text conversion: "Page 2 of 3", "- 2 -".
_PAGE_MARKER_RE = re.compile(
    r"^[ \t]*(?:page \d+(?: of \d+)?|-[ \t]*\d+[ \t]*-)[ \t]*$", re.I | re.M
)
# A word split across lines with a hyphen: "agree-\nment" -> "agreement".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def run_intake(document: RawDocument, *, max_chars: int) -> NormalizedDocument:
    """Normalize ``document`` and cap it at ``max_chars`` characters.

    Truncation is recorded in ``truncated``/``cleanup_notes``. It is a deliberate, realistic limit
    (think cost control) that can drop information later steps need.
    Raises ``IntakeError`` if nothing usable remains.
    """
    notes: list[str] = []
    text = document.content.replace("\r\n", "\n").replace("\r", "\n")

    if document.format is DocumentFormat.PDF_TEXT:
        text, pdf_notes = _clean_pdf_artifacts(text)
        notes.extend(pdf_notes)

    text = _normalize_whitespace(text)
    if not text:
        raise IntakeError(f"document {document.doc_id!r} is empty after normalization")

    normalized_length = len(text)
    truncated = normalized_length > max_chars
    if truncated:
        text = text[:max_chars].rstrip()
        notes.append(f"truncated from {normalized_length} to {len(text)} characters")

    return NormalizedDocument(
        doc_id=document.doc_id,
        format=document.format,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
        original_char_count=len(document.content),
        truncated=truncated,
        cleanup_notes=notes,
    )


def _clean_pdf_artifacts(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    form_feeds = text.count("\f")
    if form_feeds:
        text = text.replace("\f", "\n")
        notes.append(f"replaced {form_feeds} form feed(s)")
    text, markers = _PAGE_MARKER_RE.subn("", text)
    if markers:
        notes.append(f"removed {markers} page marker line(s)")
    text, joins = _HYPHEN_BREAK_RE.subn(r"\1\2", text)
    if joins:
        notes.append(f"rejoined {joins} hyphenated line break(s)")
    return text, notes


def _normalize_whitespace(text: str) -> str:
    lines = [_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()
