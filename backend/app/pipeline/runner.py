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


def run_pipeline(
    document: RawDocument, llm: LLMClient, config: PipelineConfig | None = None
) -> PipelineResult:
    """Run Intake → Extraction → Classification → Summarization on ``document``.

    Stops at the first step that raises an expected error (``PipelineStepError`` or ``LLMError``),
    returning the outputs produced so far plus a ``StepError``. Unexpected exceptions are bugs and
    propagate unchanged.
    """
    cfg = config or PipelineConfig()
    log = logger.bind(doc_id=document.doc_id)
    normalized: NormalizedDocument | None = None
    entities: ExtractedEntities | None = None
    classification: ClassificationResult | None = None
    summary: AnySummary | None = None
    step = StepName.INTAKE

    try:
        normalized = run_intake(document, max_chars=cfg.intake_max_chars)
        log.info("step_completed", step=step.value, truncated=normalized.truncated)

        step = StepName.EXTRACTION
        entities = run_extraction(
            normalized,
            llm,
            temperature=cfg.temperature,
            max_repair_attempts=cfg.max_repair_attempts,
        )
        log.info("step_completed", step=step.value)

        step = StepName.CLASSIFICATION
        classification = run_classification(
            ClassificationInput(document=normalized, entities=entities),
            llm,
            temperature=cfg.temperature,
            max_repair_attempts=cfg.max_repair_attempts,
        )
        log.info("step_completed", step=step.value, document_type=classification.document_type)

        step = StepName.SUMMARIZATION
        summary = run_summarization(
            SummarizationInput(
                document=normalized, entities=entities, classification=classification
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
