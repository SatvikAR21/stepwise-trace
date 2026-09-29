"""Follow-up prompt sent when an LLM answer could not be parsed or validated."""

from app.llm.prompts.base import PromptTemplate

REPAIR_PROMPT = PromptTemplate(
    name="repair",
    version="1.0.0",
    system="",
    user="""\
Your previous answer could not be used. Problems found:
$problems

Reply again with ONE corrected JSON object in exactly the requested shape, including "confidence",
and nothing else.""",
)
