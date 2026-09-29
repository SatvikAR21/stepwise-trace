"""Run the pipeline with tracing switched on."""

from __future__ import annotations

from app.llm.base import LLMClient
from app.pipeline.models import PipelineConfig, PipelineResult, RawDocument
from app.pipeline.runner import run_pipeline
from app.tracing.models import Trace
from app.tracing.tracer import start_trace


def trace_pipeline(
    document: RawDocument, llm: LLMClient, config: PipelineConfig | None = None
) -> tuple[PipelineResult, Trace]:
    """Run all four steps inside one trace and return the result together with its trace."""
    cfg = config or PipelineConfig()
    with start_trace(document.doc_id, model=llm.model_name, config=cfg) as recorder:
        result = run_pipeline(document, llm, cfg)
    return result, recorder.finish(result)
