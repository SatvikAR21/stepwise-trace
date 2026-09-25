"""Errors raised by pipeline steps themselves (LLM errors live in ``app.llm.base``)."""


class PipelineStepError(Exception):
    """Base class for expected, recordable step failures."""


class IntakeError(PipelineStepError):
    """The raw document cannot be turned into usable text."""
