# StepWise

> See every step, find the root cause: tracing and root-cause analysis for multi-step LLM pipelines,
> in the spirit of LangSmith and Braintrust.

When a multi-step AI pipeline produces a bad answer, the hard question is *which step broke*.
StepWise runs a four-step document pipeline (Intake → Extraction → Classification → Summarization),
records what every step received and produced, and walks backward through that record to find the
first step that went wrong.

**Status:** work in progress.

- Done:
  - A typed four-step pipeline (Pydantic models for every step), versioned prompts, a mock LLM
    client plus an OpenAI-compatible client (tested with Gemini), a 21-document sample corpus with
    8 deliberately failing cases, and a CLI.
  - Tracing: every run is recorded step by step (inputs, outputs, prompts, raw LLM answers, tokens,
    latency, the model's self-reported confidence and grounding checks) along with the settings it
    used, given a status (success / degraded / failure), saved as JSON with a SQLite index, and
    served by an API.
    Invalid LLM output gets one repair attempt, and real providers can be rate limited.
- Next: backward root-cause analyzer, visual trace explorer, feedback-to-eval loop.

**Stack:** Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy + SQLite · structlog · uv · ruff ·
mypy (strict) · pytest · pre-commit

## Quickstart (backend)
```bash
cd backend
uv sync
uv run pytest
uv run python -m app.pipeline.cli run --all          # run the sample corpus, recording a trace per run
uv run python -m app.pipeline.cli traces list --status degraded   # runs that look suspicious
uv run python -m app.pipeline.cli traces show <trace_id>          # one full trace as JSON
uv run uvicorn app.main:app --reload                 # API docs at http://127.0.0.1:8000/docs
```

The API serves `GET /traces` (newest first; filter with `status`, `doc_id`, `limit`, `offset`) and
`GET /traces/{trace_id}`. Example traces live in [`traces/samples/`](traces/samples/).

Copy `.env.example` to `.env` to configure. The default `LLM_PROVIDER=mock` needs no API key.
