"""Re-run a recorded pipeline run from any step.

The outputs of the steps before ``from_step`` are rebuilt from the trace (every span stores its
output as JSON that validates back into the step's model), optionally replaced, and the pipeline
continues from there with the settings the run used. This is the basis for fault injection
(replace one step's output with a known fault and see what happens downstream) and for
counterfactual checks (replace a bad output with a corrected one).
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from app.llm.base import LLMClient
from app.pipeline.models import (
    ClassificationResult,
    ExtractedEntities,
    NormalizedDocument,
    PipelineConfig,
    PipelineResult,
    RawDocument,
    StepName,
)
from app.pipeline.runner import run_from
from app.tracing.models import SpanStatus, Trace

_ORDER = list(StepName)
_OUTPUT_MODELS: dict[StepName, type[BaseModel]] = {
    StepName.INTAKE: NormalizedDocument,
    StepName.EXTRACTION: ExtractedEntities,
    StepName.CLASSIFICATION: ClassificationResult,
}


def resume_pipeline(
    trace: Trace,
    from_step: StepName,
    llm: LLMClient,
    *,
    config: PipelineConfig | None = None,
    replace: Mapping[StepName, BaseModel] | None = None,
) -> PipelineResult:
    """Run ``from_step`` and every later step, reusing what ``trace`` recorded before it.

    ``config`` defaults to the settings the trace recorded (or the defaults for older traces).
    ``replace`` substitutes recorded outputs of earlier steps before resuming. Raises
    ``ValueError`` if an earlier step has no usable output in the trace or a replacement is not
    for an earlier step.
    """
    spans = {StepName(span.name): span for span in trace.spans}
    document = RawDocument.model_validate(spans[StepName.INTAKE].input)
    earlier = _ORDER[: _ORDER.index(from_step)]
    replacements = dict(replace or {})
    invalid = [step for step in replacements if step not in earlier]
    if invalid:
        raise ValueError(
            f"can only replace outputs of steps before {from_step.value}, got "
            f"{[step.value for step in invalid]}"
        )
    outputs: dict[StepName, BaseModel] = {}
    for step in earlier:
        if step in replacements:
            outputs[step] = _OUTPUT_MODELS[step].model_validate(replacements[step].model_dump())
            continue
        span = spans.get(step)
        if span is None or span.status is not SpanStatus.OK:
            raise ValueError(f"the trace has no output for {step.value} to resume from")
        outputs[step] = _OUTPUT_MODELS[step].model_validate(span.output)
    normalized = outputs.get(StepName.INTAKE)
    entities = outputs.get(StepName.EXTRACTION)
    classification = outputs.get(StepName.CLASSIFICATION)
    return run_from(
        document,
        llm,
        config or trace.config or PipelineConfig(),
        start=from_step,
        normalized=normalized if isinstance(normalized, NormalizedDocument) else None,
        entities=entities if isinstance(entities, ExtractedEntities) else None,
        classification=(
            classification if isinstance(classification, ClassificationResult) else None
        ),
    )
