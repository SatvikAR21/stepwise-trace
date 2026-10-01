"""Tests for the failure taxonomy."""

from __future__ import annotations

from app.analysis.taxonomy import (
    TAXONOMY,
    FailureCategory,
    categories_for,
    definition_of,
    is_allowed,
)
from app.core.config import BACKEND_DIR
from app.pipeline.documents import load_manifest
from app.pipeline.models import StepName


def test_every_category_has_exactly_one_definition() -> None:
    assert [d.category for d in TAXONOMY] == list(FailureCategory)


def test_categories_match_the_corpus_answer_key() -> None:
    manifest = load_manifest(BACKEND_DIR / "data")
    used = {e.intended_failure for e in manifest.documents if e.intended_failure}

    assert used == {c.value for c in FailureCategory}


def test_every_planted_failure_starts_at_a_step_its_category_allows() -> None:
    manifest = load_manifest(BACKEND_DIR / "data")

    for entry in manifest.documents:
        if entry.intended_failure and entry.failing_step:
            category = FailureCategory(entry.intended_failure)
            assert is_allowed(category, StepName(entry.failing_step)), entry.doc_id


def test_categories_per_step() -> None:
    assert categories_for(StepName.INTAKE) == [FailureCategory.CONTEXT_LOSS]
    assert categories_for(StepName.CLASSIFICATION) == [
        FailureCategory.MISCLASSIFICATION,
        FailureCategory.PROMPT_FAILURE,
    ]
    assert FailureCategory.PROPAGATION_ERROR in categories_for(StepName.SUMMARIZATION)
    assert not is_allowed(FailureCategory.MISCLASSIFICATION, StepName.EXTRACTION)


def test_every_step_can_be_blamed_for_something() -> None:
    for step in StepName:
        assert categories_for(step), step


def test_definitions_are_readable() -> None:
    for definition in TAXONOMY:
        assert definition.label and definition.definition.endswith(".")
        assert definition_of(definition.category) is definition
