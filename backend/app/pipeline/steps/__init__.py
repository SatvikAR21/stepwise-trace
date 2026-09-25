"""The four pipeline steps. Each is a plain function: typed input in, typed output out."""

from app.pipeline.steps.classification import run_classification
from app.pipeline.steps.extraction import run_extraction
from app.pipeline.steps.intake import run_intake
from app.pipeline.steps.summarization import run_summarization

__all__ = ["run_classification", "run_extraction", "run_intake", "run_summarization"]
