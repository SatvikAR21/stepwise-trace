"""Shared helper for LLM-backed steps: send a rendered prompt, parse the structured answer."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.llm.base import META_DOC_ID, META_STEP, ChatMessage, LLMClient, LLMRequest
from app.llm.structured import parse_json_output
from app.pipeline.models import StepName

ModelT = TypeVar("ModelT", bound=BaseModel)


def call_structured(
    llm: LLMClient,
    messages: list[ChatMessage],
    output_model: type[ModelT],
    *,
    doc_id: str,
    step: StepName,
    temperature: float = 0.0,
) -> ModelT:
    """Make one JSON-mode LLM call and validate the answer against ``output_model``.

    Raises ``LLMProviderError`` if the call fails and ``LLMOutputError`` if the answer is invalid.
    """
    request = LLMRequest(
        messages=messages,
        temperature=temperature,
        json_mode=True,
        metadata={META_DOC_ID: doc_id, META_STEP: step.value},
    )
    response = llm.complete(request)
    return parse_json_output(response.content, output_model)


def to_prompt_json(model: BaseModel) -> str:
    """Serialize a model compactly for inclusion in a prompt."""
    return model.model_dump_json(exclude_none=True)
