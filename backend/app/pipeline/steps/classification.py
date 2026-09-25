"""Step 3 (Classification): decide the document type with an LLM."""

from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.prompts import CLASSIFICATION_PROMPT
from app.pipeline.models import ClassificationInput, ClassificationResult, StepName
from app.pipeline.steps.common import call_structured, to_prompt_json


def run_classification(
    step_input: ClassificationInput, llm: LLMClient, *, temperature: float = 0.0
) -> ClassificationResult:
    """Classify the document as contract, invoice, report or correspondence."""
    messages = CLASSIFICATION_PROMPT.render(
        entities_json=to_prompt_json(step_input.entities),
        document_text=step_input.document.text,
    )
    return call_structured(
        llm,
        messages,
        ClassificationResult,
        doc_id=step_input.document.doc_id,
        step=StepName.CLASSIFICATION,
        temperature=temperature,
    )
