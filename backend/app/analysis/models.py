"""Data shapes for root-cause analysis: the judge's answer and the resulting diagnosis."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.analysis.taxonomy import FailureCategory, categories_for, is_allowed
from app.pipeline.models import StepName

# --------------------------------------------------------------------------- the judge's answer


class StepGrade(BaseModel):
    """The judge's verdict on one step (LLM-facing, so unknown fields are rejected)."""

    model_config = ConfigDict(extra="forbid")

    step: StepName
    score: int = Field(ge=1, le=5)
    introduced: list[str] = Field(default_factory=list)
    inherited: list[str] = Field(default_factory=list)
    category: FailureCategory | None = None
    explanation: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _category_fits_the_step(self) -> Self:
        if not self.introduced:
            self.category = None  # nothing introduced, nothing to categorize
        elif self.category is not None and not is_allowed(self.category, self.step):
            allowed = ", ".join(c.value for c in categories_for(self.step))
            raise ValueError(
                f"category {self.category.value!r} cannot start at {self.step.value}; "
                f"allowed there: {allowed}"
            )
        return self


class JudgeAnswer(BaseModel):
    """The judge's verdicts on every step that ran, in pipeline order."""

    model_config = ConfigDict(extra="forbid")

    steps: list[StepGrade]

    def problems_for(self, expected: list[StepName]) -> list[str]:
        """Why this answer does not cover exactly ``expected``, in order (empty if it does)."""
        got = [grade.step for grade in self.steps]
        if got == expected:
            return []
        return [
            "steps must be exactly "
            f"{[s.value for s in expected]} in this order, got {[s.value for s in got]}"
        ]
