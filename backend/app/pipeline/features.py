"""Step-specific measurements recorded on each span.

They are clues that a future failure classifier can learn from. Nothing decides anything with them
yet, except the ``truncated`` flag, which marks a run as degraded.

The key clue is grounding: an extracted name, date or amount that does not appear word for word in
the document, or a number in the summary that appears nowhere in the document, is a warning sign of
an invented (hallucinated) or miscalculated value.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from app.pipeline.models import (
    AnySummary,
    ClassificationInput,
    ClassificationResult,
    ExtractedEntities,
    NormalizedDocument,
    RawDocument,
    SummarizationInput,
)
from app.tracing.models import FeatureValue

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def intake_features(document: RawDocument, output: NormalizedDocument) -> dict[str, FeatureValue]:
    """How much of the document survived cleaning and the length limit."""
    return {
        "truncated": output.truncated,
        "original_chars": output.original_char_count,
        "kept_chars": output.char_count,
        "kept_ratio": round(output.char_count / max(output.original_char_count, 1), 4),
        "cleanup_notes": len(output.cleanup_notes),
    }


def extraction_features(
    document: NormalizedDocument, entities: ExtractedEntities
) -> dict[str, FeatureValue]:
    """Counts per entity type, and how many names, dates and amounts are grounded in the text."""
    mentions = [
        *(p.name for p in entities.people),
        *(o.name for o in entities.organizations),
        *(d.raw for d in entities.dates),
        *(a.raw for a in entities.amounts),
    ]
    grounded = sum(is_grounded(mention, document.text) for mention in mentions)
    return {
        "people": len(entities.people),
        "organizations": len(entities.organizations),
        "dates": len(entities.dates),
        "amounts": len(entities.amounts),
        "key_terms": len(entities.key_terms),
        "currencies": len({a.currency for a in entities.amounts}),
        "entities_checked": len(mentions),
        "entities_ungrounded": len(mentions) - grounded,
        "grounded_ratio": round(grounded / len(mentions), 4) if mentions else 1.0,
    }


def classification_features(
    step_input: ClassificationInput, result: ClassificationResult
) -> dict[str, FeatureValue]:
    """The chosen type and how much the model wrote to justify it."""
    return {
        "document_type": result.document_type.value,
        "rationale_chars": len(result.rationale),
    }


def summarization_features(
    step_input: SummarizationInput, summary: AnySummary
) -> dict[str, FeatureValue]:
    """How many distinct numbers the summary states, and how many appear nowhere in the document."""
    in_document = numbers_in(step_input.document.text)
    in_summary = _summary_numbers(summary.model_dump(mode="json"))
    return {
        "key_points": len(summary.key_points),
        "summary_numbers": len(in_summary),
        "summary_numbers_ungrounded": len(in_summary - in_document),
    }


def is_grounded(mention: str, text: str) -> bool:
    """True if ``mention`` appears in ``text``, ignoring case and runs of whitespace."""
    return _squash(mention) in _squash(text)


def numbers_in(text: str) -> set[Decimal]:
    """Every number written in ``text``, as exact values: "$1,284.50" and "1284.5" are equal."""
    return {Decimal(token.replace(",", "")) for token in _NUMBER_RE.findall(text)}


def _summary_numbers(value: Any) -> set[Decimal]:
    """Numbers inside the summary's text fields. ISO dates and counts are skipped: they are
    reformatted or computed by the model, so they are not expected to appear verbatim."""
    if isinstance(value, dict):
        return set().union(*(_summary_numbers(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_summary_numbers(v) for v in value))
    if isinstance(value, str) and not _ISO_DATE_RE.match(value):
        return numbers_in(value)
    return set()


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()
