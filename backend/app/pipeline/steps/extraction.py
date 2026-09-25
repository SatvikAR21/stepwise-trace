"""Step 2 (Extraction): pull structured entities out of the document with an LLM."""

from __future__ import annotations

from app.llm.base import LLMClient
from app.llm.prompts import EXTRACTION_PROMPT
from app.pipeline.models import ExtractedEntities, NormalizedDocument, StepName
from app.pipeline.steps.common import call_structured


def run_extraction(
    document: NormalizedDocument, llm: LLMClient, *, temperature: float = 0.0
) -> ExtractedEntities:
    """Extract people, organizations, dates, amounts and key terms from ``document``."""
    messages = EXTRACTION_PROMPT.render(document_text=document.text)
    return call_structured(
        llm,
        messages,
        ExtractedEntities,
        doc_id=document.doc_id,
        step=StepName.EXTRACTION,
        temperature=temperature,
    )
