# StepWise

> See every step, find the root cause: tracing and root-cause analysis for multi-step LLM pipelines,
> in the spirit of LangSmith and Braintrust.

When a multi-step AI pipeline produces a bad answer, the hard question is *which step broke*.
StepWise runs a four-step document pipeline (Intake → Extraction → Classification → Summarization),
records what every step received and produced, and walks backward through that record to find the
first step that went wrong.

**Status:** work in progress.

- Done: typed four-step pipeline (Pydantic models for every step), versioned prompts, a mock LLM
  client plus an OpenAI-compatible client (tested with Gemini), a 21-document sample corpus with
  8 deliberately failing cases, and a CLI.
- Next: tracing layer, backward root-cause analyzer, visual trace explorer, feedback-to-eval loop.

**Stack:** Python 3.12 · FastAPI · Pydantic v2 · structlog · uv · ruff · mypy (strict) · pytest

## Quickstart (backend)
```bash
cd backend
uv sync
uv run pytest
uv run python -m app.pipeline.cli run --all    # run the sample corpus through the mock pipeline
uv run uvicorn app.main:app --reload           # then open http://127.0.0.1:8000/health
```

Copy `.env.example` to `.env` to configure. The default `LLM_PROVIDER=mock` needs no API key.
