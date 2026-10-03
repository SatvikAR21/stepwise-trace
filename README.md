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
    client plus an OpenAI-compatible client (tested with Gemini), a 32-document sample corpus with
    16 deliberately failing cases (split into a practice set and a held-out exam set), and a CLI.
  - Tracing: every run is recorded step by step (inputs, outputs, prompts, raw LLM answers, tokens,
    latency, the model's self-reported confidence and grounding checks) along with the settings it
    used, given a status (success / degraded / failure), saved as JSON with a SQLite index, and
    served by an API.
    Invalid LLM output gets one repair attempt, and real providers can be rate limited.
  - Root-cause analysis: an LLM judge grades every step of a recorded run in one request (score,
    problems introduced vs. inherited, failure category, quoted evidence); a backward walk names
    the earliest step with a significant quality drop and builds the evidence chain. Plain-code
    checks run alongside as a second opinion. Diagnoses are saved, served by the API, and graded
    against the corpus answer key with a report card (measured results below). Any recorded run
    can be resumed from any step.
- Next: fault injection (known faults planted in step outputs, a more independent test of the
  root-cause analysis), visual trace explorer, feedback-to-eval loop.

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
uv run python -m app.pipeline.cli analyze --doc contract_no_dates_04   # root cause of a saved run
uv run python -m app.pipeline.cli evaluate --split practice      # report card against the answer key
uv run uvicorn app.main:app --reload                 # API docs at http://127.0.0.1:8000/docs
```

The API serves `GET /traces` (newest first; filter with `status`, `doc_id`, `limit`, `offset`),
`GET /traces/{trace_id}`, `POST /traces/{trace_id}/analysis` (diagnose and save) and
`GET /traces/{trace_id}/analysis` (latest diagnosis). Example traces live in
[`traces/samples/`](traces/samples/).

Copy `.env.example` to `.env` to configure. The default `LLM_PROVIDER=mock` and `JUDGE_PROVIDER=mock`
need no API key; the mock judge's verdicts are scripted, so its report card is not a measurement.
A real judge (`--judge-provider openai`) makes no hidden retries, is rate limited, reuses saved
verdicts for identical requests, and `evaluate` requires a hard `--max-calls` limit.

## Measured results: root-cause analysis

The judge was `gemini-3.5-flash` (judge prompt 1.2.0; a step scoring 2/5 or lower counts as a
significant drop). It graded recorded runs of the sample corpus, whose pipeline answers are scripted,
so the right answer for every document is known.

| Set | Broken documents: right step and category | Healthy documents: false alarms |
|---|---|---|
| Practice: 21 documents, used to tune the judge prompt | 8 of 8 (95% range 68% to 100%) | 0 of 13 (0% to 23%) |
| Exam: 11 held-out documents, run once | 8 of 8 (68% to 100%) | 0 of 3 (0% to 56%) |

On the exam, the plain-code checks alone flagged 4 of the 8 broken documents (3 at the right step)
and raised false alarms on 2 of the 3 healthy ones, which is why they only advise and the judge
decides. Each diagnosis took one judge request (on average about 1,900 prompt and 380 completion
tokens).

Read these numbers with care:
- The sets are small, so the ranges (Wilson 95% intervals) are wide.
- One prompt change came from the practice set: text cut off at intake that removes material
  content now counts as a significant drop. It is worded generally, but the exam contains a similar
  case.
- The same author wrote the documents, the answer key and the judge prompt. Fault injection and
  human-confirmed cases are planned as more independent tests.
- The judge is a Gemini model. Here it grades scripted answers, but with a Gemini pipeline the judge
  and the pipeline would share a model family and could share blind spots. The judge is configured
  separately (`JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY`), so another model family can be used.

To reproduce, from `backend/` (needs an API key in `.env`; the second command uses 11 requests):
```bash
uv run python -m app.pipeline.cli --judge-provider openai --judge-model gemini-3.5-flash evaluate --split practice --max-calls 21
uv run python -m app.pipeline.cli --judge-provider openai --judge-model gemini-3.5-flash evaluate --split exam --max-calls 11
```
