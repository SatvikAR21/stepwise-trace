"""The failure taxonomy: the kinds of mistake a root-cause diagnosis can name.

Each category says where in the pipeline it can start. The judge prompt is rendered from this
list, so the prompt never repeats the definitions. To add a category, add an enum member and one
``CategoryDefinition`` to ``TAXONOMY``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.pipeline.models import StepName


class FailureCategory(StrEnum):
    """Machine-readable failure category (the same values as the corpus manifest)."""

    EXTRACTION_HALLUCINATION = "extraction_hallucination"
    MISCLASSIFICATION = "misclassification"
    PROPAGATION_ERROR = "propagation_error"
    PROMPT_FAILURE = "prompt_failure"
    CONTEXT_LOSS = "context_loss"


class CategoryDefinition(BaseModel):
    """What a category means and at which steps it can originate."""

    model_config = ConfigDict(frozen=True)

    category: FailureCategory
    label: str
    definition: str
    steps: frozenset[StepName]


TAXONOMY: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        category=FailureCategory.EXTRACTION_HALLUCINATION,
        label="Extraction Hallucination",
        definition=(
            "Extraction reported a name, date, amount or other entity that does not appear in "
            "the document it received."
        ),
        steps=frozenset({StepName.EXTRACTION}),
    ),
    CategoryDefinition(
        category=FailureCategory.MISCLASSIFICATION,
        label="Misclassification",
        definition="Classification chose the wrong document type for the document it received.",
        steps=frozenset({StepName.CLASSIFICATION}),
    ),
    CategoryDefinition(
        category=FailureCategory.PROPAGATION_ERROR,
        label="Propagation Error",
        definition=(
            "The earlier steps' outputs were correct, but this step misread or misused them, "
            "for example by swapping roles, dates or amounts that were correctly extracted."
        ),
        steps=frozenset({StepName.SUMMARIZATION}),
    ),
    CategoryDefinition(
        category=FailureCategory.PROMPT_FAILURE,
        label="Prompt Failure",
        definition=(
            "The step obeyed instructions written inside the document, or did not answer in "
            "the required JSON shape. Use it only for these two cases: an invented, wrong or "
            "dropped value belongs to one of the other categories."
        ),
        steps=frozenset({StepName.EXTRACTION, StepName.CLASSIFICATION, StepName.SUMMARIZATION}),
    ),
    CategoryDefinition(
        category=FailureCategory.CONTEXT_LOSS,
        label="Context Loss",
        definition=(
            "Important information that the step received was dropped from its output, so "
            "later steps could not use it."
        ),
        steps=frozenset({StepName.INTAKE, StepName.EXTRACTION, StepName.SUMMARIZATION}),
    ),
)

_BY_CATEGORY = {definition.category: definition for definition in TAXONOMY}


def definition_of(category: FailureCategory) -> CategoryDefinition:
    """The definition of ``category``."""
    return _BY_CATEGORY[category]


def categories_for(step: StepName) -> list[FailureCategory]:
    """The categories that can originate at ``step``, in taxonomy order."""
    return [d.category for d in TAXONOMY if step in d.steps]


def is_allowed(category: FailureCategory, step: StepName) -> bool:
    """True if ``category`` can originate at ``step``."""
    return step in definition_of(category).steps
