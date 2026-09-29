"""Run the pipeline with tracing switched on."""

from __future__ import annotations

from app.llm.base import LLMClient
from app.pipeline.models import PipelineResult, RawDocument
from app.pipeline.runner import PipelineConfig, run_pipeline
from app.tracing.models import Trace
from app.tracing.tracer import start_trace


def trace_pipeline(
    document: RawDocument, llm: LLMClient, config: PipelineConfig | None = None
) -> tuple[PipelineResult, Trace]:
    """Run all four steps inside one trace and return the result together with its trace."""
    with start_trace(document.doc_id, model=llm.model_name) as recorder:
        result = run_pipeline(document, llm, config)
    return result, recorder.finish(result)
