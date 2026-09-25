# Failure Forensics

> Observability and root-cause analysis for multi-step AI pipelines, a mini LangSmith/Braintrust.

**Status:** under construction.

## Quickstart (backend)
```bash
cd backend
uv sync
uv run pytest
uv run uvicorn app.main:app --reload   # then open http://127.0.0.1:8000/health
```

Copy `.env.example` to `.env` to configure. The default `LLM_PROVIDER=mock` needs no API key.
