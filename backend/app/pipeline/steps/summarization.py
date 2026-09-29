"""Step 4 (Summarization): write a structured summary shaped by the document type."""

from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.prompts import SUMMARIZATION_PROMPT, SUMMARY_SCHEMAS
from app.pipeline.features import summarization_features
from app.pipeline.models import SUMMARY_MODELS, AnySummary, StepName, SummarizationInput
from app.pipeline.steps.common import call_structured, to_prompt_json
from app.tracing.tracer import traced_step


@traced_step(StepName.SUMMARIZATION, features=summarization_features)
def run_summarization(
    step_input: SummarizationInput,
    llm: LLMClient,
    *,
    temperature: float = 0.0,
    max_repair_attempts: int = 1,
) -> AnySummary:
    """Summarize the document using the schema for its classified type.

    The summary type follows the classification, so a misclassification upstream produces a
    summary of the wrong shape here, exactly the kind of propagation the analyzer looks for.
    """
    document_type = step_input.classification.document_type
    return call_structured(
        llm,
        SUMMARIZATION_PROMPT,
        {
            "document_type": document_type.value,
            "summary_schema": SUMMARY_SCHEMAS[document_type.value],
            "entities_json": to_prompt_json(step_input.entities),
            "document_text": step_input.document.text,
        },
        SUMMARY_MODELS[document_type],
        doc_id=step_input.document.doc_id,
        step=StepName.SUMMARIZATION,
        temperature=temperature,
        max_repair_attempts=max_repair_attempts,
    )
