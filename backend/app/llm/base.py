"""Provider-agnostic LLM interface: request/response models, errors, and the ``LLMClient`` ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, Field

# Well-known ``LLMRequest.metadata`` keys. The mock client scripts responses by them, and the
# tracing layer records them.
META_DOC_ID = "doc_id"
META_STEP = "step"


class Role(StrEnum):
    """Chat message author."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """One message in a chat-style prompt."""

    role: Role
    content: str


class LLMRequest(BaseModel):
    """Everything needed to make one LLM call."""

    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    json_mode: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token counts reported by the provider (or estimated by the mock)."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    """The raw result of one LLM call, before any parsing."""

    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)


class LLMError(Exception):
    """Base class for every error raised by the LLM layer."""


class LLMProviderError(LLMError):
    """The provider could not be reached or returned an error (network, auth, rate limit...)."""


class LLMOutputError(LLMError):
    """The provider answered, but the answer is not the structured output we asked for."""

    def __init__(self, message: str, *, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class LLMClient(ABC):
    """Interface every LLM provider implements. Pipeline code depends only on this."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model, recorded with every call."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request`` and return the raw response. Raises ``LLMProviderError`` on failure."""
