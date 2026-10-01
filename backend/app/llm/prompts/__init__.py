"""Prompt templates, one module per pipeline step. Never inline prompts elsewhere."""

from app.llm.prompts.base import PromptTemplate
from app.llm.prompts.classification import CLASSIFICATION_PROMPT
from app.llm.prompts.extraction import EXTRACTION_PROMPT
from app.llm.prompts.judge import JUDGE_REPAIR_PROMPT, TRACE_JUDGE_PROMPT
from app.llm.prompts.repair import REPAIR_PROMPT
from app.llm.prompts.summarization import SUMMARIZATION_PROMPT, SUMMARY_SCHEMAS

__all__ = [
    "CLASSIFICATION_PROMPT",
    "EXTRACTION_PROMPT",
    "JUDGE_REPAIR_PROMPT",
    "REPAIR_PROMPT",
    "SUMMARIZATION_PROMPT",
    "SUMMARY_SCHEMAS",
    "TRACE_JUDGE_PROMPT",
    "PromptTemplate",
]
