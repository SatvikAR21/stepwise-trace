"""Step 2 (Extraction): pull structured entities out of the document with an LLM."""

from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.prompts import EXTRACTION_PROMPT
from app.pipeline.features import extraction_features
from app.pipeline.models import ExtractedEntities, NormalizedDocument, StepName
from app.pipeline.steps.common import call_structured
from app.tracing.tracer import traced_step


@traced_step(StepName.EXTRACTION, features=extraction_features)
def run_extraction(
    document: NormalizedDocument,
    llm: LLMClient,
    *,
    temperature: float = 0.0,
    max_repair_attempts: int = 1,
) -> ExtractedEntities:
    """Extract people, organizations, dates, amounts and key terms from ``document``."""
    return call_structured(
        llm,
        EXTRACTION_PROMPT,
        {"document_text": document.text},
        ExtractedEntities,
        doc_id=document.doc_id,
        step=StepName.EXTRACTION,
        temperature=temperature,
        max_repair_attempts=max_repair_attempts,
    )
