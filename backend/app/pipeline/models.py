"""Typed input/output models for every pipeline step.

Data flow::

    RawDocument ─Intake─► NormalizedDocument ─Extraction─► ExtractedEntities
                ─Classification─► ClassificationResult ─Summarization─► DocumentSummary
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StepName(StrEnum):
    """The four pipeline steps, in execution order."""

    INTAKE = "intake"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"


class DocumentFormat(StrEnum):
    """How the raw document text was produced."""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF_TEXT = "pdf_text"


class DocumentType(StrEnum):
    """The categories the Classification step chooses between."""

    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    CORRESPONDENCE = "correspondence"


class _StrictModel(BaseModel):
    """Base for LLM-facing models: reject unknown fields so schema drift is caught."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- Step 1: Intake


class RawDocument(BaseModel):
    """A document as it arrives at the pipeline."""

    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=100)
    content: str = Field(min_length=1)
    format: DocumentFormat = DocumentFormat.TEXT


class NormalizedDocument(BaseModel):
    """Intake output: cleaned text plus notes on what was changed."""

    doc_id: str
    format: DocumentFormat
    text: str
    char_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    original_char_count: int = Field(ge=0)
    truncated: bool = False
    cleanup_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- Step 2: Extraction


class Person(_StrictModel):
    """A named person mentioned in the document."""

    name: str = Field(min_length=1)
    role: str | None = None


class Organization(_StrictModel):
    """A named organization mentioned in the document."""

    name: str = Field(min_length=1)
    role: str | None = None


class DateMention(_StrictModel):
    """A date as written in the document, plus its ISO form when it is an absolute date."""

    raw: str = Field(min_length=1, description="Exact text from the document")
    iso_date: date | None = Field(default=None, description="YYYY-MM-DD, or null if relative")
    context: str | None = Field(default=None, description="What the date refers to")


class MoneyAmount(_StrictModel):
    """A monetary amount with an ISO 4217 currency code."""

    raw: str = Field(min_length=1, description="Exact text from the document")
    value: Decimal
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    context: str | None = None


class ExtractedEntities(_StrictModel):
    """Extraction output: structured entities found in the document."""

    people: list[Person] = Field(default_factory=list)
    organizations: list[Organization] = Field(default_factory=list)
    dates: list[DateMention] = Field(default_factory=list)
    amounts: list[MoneyAmount] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- Step 3: Classification


class ClassificationInput(BaseModel):
    """What the Classification step receives."""

    document: NormalizedDocument
    entities: ExtractedEntities


class ClassificationResult(_StrictModel):
    """Classification output."""

    document_type: DocumentType
    rationale: str = Field(min_length=1)


# --------------------------------------------------------------------------- Step 4: Summarization


class SummarizationInput(BaseModel):
    """What the Summarization step receives."""

    document: NormalizedDocument
    entities: ExtractedEntities
    classification: ClassificationResult


class _SummaryBase(_StrictModel):
    headline: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)


class ContractSummary(_SummaryBase):
    """Summary shape for contracts."""

    document_type: Literal[DocumentType.CONTRACT] = DocumentType.CONTRACT
    parties: list[str] = Field(default_factory=list)
    effective_date: date | None = None
    term: str | None = None
    key_obligations: list[str] = Field(default_factory=list)
    termination_terms: str | None = None


class InvoiceSummary(_SummaryBase):
    """Summary shape for invoices."""

    document_type: Literal[DocumentType.INVOICE] = DocumentType.INVOICE
    vendor: str | None = None
    customer: str | None = None
    invoice_number: str | None = None
    total_amount: MoneyAmount | None = None
    due_date: date | None = None
    line_item_count: int = Field(default=0, ge=0)


class ReportSummary(_SummaryBase):
    """Summary shape for reports."""

    document_type: Literal[DocumentType.REPORT] = DocumentType.REPORT
    title: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class CorrespondenceSummary(_SummaryBase):
    """Summary shape for letters and emails."""

    document_type: Literal[DocumentType.CORRESPONDENCE] = DocumentType.CORRESPONDENCE
    sender: str | None = None
    recipient: str | None = None
    purpose: str | None = None
    action_items: list[str] = Field(default_factory=list)


AnySummary = ContractSummary | InvoiceSummary | ReportSummary | CorrespondenceSummary
DocumentSummary = Annotated[AnySummary, Field(discriminator="document_type")]

SUMMARY_MODELS: dict[DocumentType, type[AnySummary]] = {
    DocumentType.CONTRACT: ContractSummary,
    DocumentType.INVOICE: InvoiceSummary,
    DocumentType.REPORT: ReportSummary,
    DocumentType.CORRESPONDENCE: CorrespondenceSummary,
}


# --------------------------------------------------------------------------- Settings and result


class PipelineConfig(BaseModel):
    """Knobs for one pipeline run."""

    intake_max_chars: int = Field(default=6000, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_repair_attempts: int = Field(default=1, ge=0, le=3)


class PipelineStatus(StrEnum):
    """Whether every step ran. (Traces add a separate success/failure/degraded quality status.)"""

    COMPLETED = "completed"
    FAILED = "failed"


class StepError(BaseModel):
    """A recorded step failure."""

    step: StepName
    error_type: str
    message: str
    raw_output: str | None = None


class PipelineResult(BaseModel):
    """The outputs of every step that ran, plus the error if one stopped the pipeline."""

    doc_id: str
    status: PipelineStatus
    normalized: NormalizedDocument | None = None
    entities: ExtractedEntities | None = None
    classification: ClassificationResult | None = None
    summary: DocumentSummary | None = None
    error: StepError | None = None
