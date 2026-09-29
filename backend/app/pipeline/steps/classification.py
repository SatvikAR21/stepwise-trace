"""Step 3 (Classification): decide the document type with an LLM."""

from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.prompts import CLASSIFICATION_PROMPT
from app.pipeline.models import ClassificationInput, ClassificationResult, StepName
from app.pipeline.steps.common import call_structured, to_prompt_json
from app.tracing.tracer import traced_step


@traced_step(StepName.CLASSIFICATION)
def run_classification(
    step_input: ClassificationInput,
    llm: LLMClient,
    *,
    temperature: float = 0.0,
    max_repair_attempts: int = 1,
) -> ClassificationResult:
    """Classify the document as contract, invoice, report or correspondence."""
    return call_structured(
        llm,
        CLASSIFICATION_PROMPT,
        {
            "entities_json": to_prompt_json(step_input.entities),
            "document_text": step_input.document.text,
        },
        ClassificationResult,
        doc_id=step_input.document.doc_id,
        step=StepName.CLASSIFICATION,
        temperature=temperature,
        max_repair_attempts=max_repair_attempts,
    )
