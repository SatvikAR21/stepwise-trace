"""Chain the four steps into one pipeline run."""

from __future__ import annotations

from app.core.logging import get_logger
from app.llm.base import LLMClient, LLMError, LLMOutputError
from app.pipeline.errors import PipelineStepError
from app.pipeline.models import (
    AnySummary,
    ClassificationInput,
    ClassificationResult,
    ExtractedEntities,
    NormalizedDocument,
    PipelineConfig,
    PipelineResult,
    PipelineStatus,
    RawDocument,
    StepError,
    StepName,
    SummarizationInput,
)
from app.pipeline.steps import run_classification, run_extraction, run_intake, run_summarization

logger = get_logger(__name__)


_ORDER = list(StepName)


def run_pipeline(
    document: RawDocument, llm: LLMClient, config: PipelineConfig | None = None
) -> PipelineResult:
    """Run Intake → Extraction → Classification → Summarization on ``document``.

    Stops at the first step that raises an expected error (``PipelineStepError`` or ``LLMError``),
    returning the outputs produced so far plus a ``StepError``. Unexpected exceptions are bugs and
    propagate unchanged.
    """
    return run_from(document, llm, config or PipelineConfig(), start=StepName.INTAKE)


def run_from(
    document: RawDocument,
    llm: LLMClient,
    cfg: PipelineConfig,
    *,
    start: StepName,
    normalized: NormalizedDocument | None = None,
    entities: ExtractedEntities | None = None,
    classification: ClassificationResult | None = None,
) -> PipelineResult:
    """Run ``start`` and every later step, given the outputs of the steps before ``start``.

    ``run_pipeline`` starts at intake; ``resume_pipeline`` (``app.pipeline.resume``) starts later
    with outputs rebuilt from a trace. Raises ``ValueError`` if an earlier output is missing.
    """
    log = logger.bind(doc_id=document.doc_id)
    first = _ORDER.index(start)
    summary: AnySummary | None = None
    step = start

    try:
        if first <= _ORDER.index(StepName.INTAKE):
            step = StepName.INTAKE
            normalized = run_intake(document, max_chars=cfg.intake_max_chars)
            log.info("step_completed", step=step.value, truncated=normalized.truncated)

        if first <= _ORDER.index(StepName.EXTRACTION):
            step = StepName.EXTRACTION
            entities = run_extraction(
                _given(normalized, StepName.INTAKE),
                llm,
                temperature=cfg.temperature,
                max_repair_attempts=cfg.max_repair_attempts,
            )
            log.info("step_completed", step=step.value)

        if first <= _ORDER.index(StepName.CLASSIFICATION):
            step = StepName.CLASSIFICATION
            classification = run_classification(
                ClassificationInput(
                    document=_given(normalized, StepName.INTAKE),
                    entities=_given(entities, StepName.EXTRACTION),
                ),
                llm,
                temperature=cfg.temperature,
                max_repair_attempts=cfg.max_repair_attempts,
            )
            log.info("step_completed", step=step.value, document_type=classification.document_type)

        step = StepName.SUMMARIZATION
        summary = run_summarization(
            SummarizationInput(
                document=_given(normalized, StepName.INTAKE),
                entities=_given(entities, StepName.EXTRACTION),
                classification=_given(classification, StepName.CLASSIFICATION),
            ),
            llm,
            temperature=cfg.temperature,
            max_repair_attempts=cfg.max_repair_attempts,
        )
        log.info("step_completed", step=step.value)
    except (PipelineStepError, LLMError) as exc:
        error = StepError(
            step=step,
            error_type=type(exc).__name__,
            message=str(exc),
            raw_output=exc.raw_output if isinstance(exc, LLMOutputError) else None,
        )
        log.warning("pipeline_failed", step=step.value, error_type=error.error_type)
        return PipelineResult(
            doc_id=document.doc_id,
            status=PipelineStatus.FAILED,
            normalized=normalized,
            entities=entities,
            classification=classification,
            summary=summary,
            error=error,
        )

    log.info("pipeline_completed")
    return PipelineResult(
        doc_id=document.doc_id,
        status=PipelineStatus.COMPLETED,
        normalized=normalized,
        entities=entities,
        classification=classification,
        summary=summary,
    )


def _given[OutputT](output: OutputT | None, step: StepName) -> OutputT:
    if output is None:
        raise ValueError(f"the output of {step.value} is needed to run the later steps")
    return output
