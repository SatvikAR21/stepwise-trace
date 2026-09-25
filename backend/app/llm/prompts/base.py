"""Versioned prompt templates.

Templates use ``$name`` placeholders (``string.Template``), so literal JSON braces in the prompt
text need no escaping. Bump ``version`` whenever the wording changes: traces record it, which lets
us tell whether a regression came from a prompt edit.
"""

from __future__ import annotations

from string import Template

from pydantic import BaseModel, ConfigDict

from app.llm.base import ChatMessage, Role


class PromptTemplate(BaseModel):
    """A named, versioned system + user prompt pair."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    system: str
    user: str

    def render(self, **values: str) -> list[ChatMessage]:
        """Fill the placeholders. Raises ``KeyError`` if a placeholder has no value."""
        return [
            ChatMessage(role=Role.SYSTEM, content=Template(self.system).substitute(values)),
            ChatMessage(role=Role.USER, content=Template(self.user).substitute(values)),
        ]
