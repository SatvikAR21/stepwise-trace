"""Data shapes for root-cause analysis: the judge's answer and the resulting diagnosis."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.analysis.taxonomy import FailureCategory, categories_for, is_allowed
from app.pipeline.models import StepName
from app.tracing.models import TRACE_ID_PATTERN, LLMCallRecord

ANALYSIS_ID_PATTERN = r"^[0-9a-f]{32}$"

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


# --------------------------------------------------------------------------- the diagnosis


class StepRole(StrEnum):
    """What the backward walk concluded about one step."""

    ROOT_CAUSE = "root_cause"  # the earliest step with a significant quality drop
    PROPAGATED = "propagated"  # a later step that carried the root problem forward
    SECONDARY = "secondary"  # a later step that also dropped quality significantly on its own
    MINOR = "minor"  # introduced a problem, but scored above the drop line
    HEALTHY = "healthy"
    NOT_RUN = "not_run"  # the pipeline stopped before this step


class StepFinding(BaseModel):
    """The evidence for one step: what it received and produced, and the judge's verdict."""

    step: StepName
    role: StepRole
    score: int | None = Field(default=None, ge=1, le=5)
    introduced: list[str] = Field(default_factory=list)
    inherited: list[str] = Field(default_factory=list)
    category: FailureCategory | None = None
    explanation: str | None = None
    evidence: list[str] = Field(default_factory=list)
    received: str | None = Field(default=None, description="Excerpt of what the step received")
    produced: str | None = Field(default=None, description="Excerpt of what the step produced")


class Analysis(BaseModel):
    """A root-cause diagnosis of one trace: the root cause, the evidence chain and its cost."""

    analysis_id: str = Field(pattern=ANALYSIS_ID_PATTERN)
    trace_id: str = Field(pattern=TRACE_ID_PATTERN)
    doc_id: str
    created_at: datetime
    duration_ms: float = Field(ge=0.0, description="Time from start to diagnosis")
    judge_model: str
    judge_prompt_version: str
    judge_fingerprint: str = Field(description="Hash of the exact judge request and model")
    reused_verdict: bool = Field(
        default=False, description="The verdict came from an earlier identical request"
    )
    drop_score: int = Field(ge=1, le=5)
    root_cause_step: StepName | None
    category: FailureCategory | None
    summary: str
    steps: list[StepFinding]
    verdict: JudgeAnswer
    judge_calls: list[LLMCallRecord] = Field(default_factory=list)
