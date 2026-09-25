"""Load the sample-document corpus described by ``<data_dir>/documents/manifest.json``."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.pipeline.models import DocumentFormat, DocumentType, RawDocument

DOCUMENTS_SUBDIR = "documents"
MANIFEST_NAME = "manifest.json"


class ManifestEntry(BaseModel):
    """Metadata for one sample document, including the failure it is designed to trigger."""

    doc_id: str
    file: str
    format: DocumentFormat
    expected_type: DocumentType
    intended_failure: str | None = Field(
        default=None, description="Failure category this document is designed to trigger"
    )
    failing_step: str | None = Field(default=None, description="Step where the failure originates")
    description: str


class DocumentManifest(BaseModel):
    """The whole corpus index."""

    version: int
    documents: list[ManifestEntry]

    def get(self, doc_id: str) -> ManifestEntry:
        """Return the entry for ``doc_id`` or raise ``KeyError``."""
        for entry in self.documents:
            if entry.doc_id == doc_id:
                return entry
        raise KeyError(f"unknown doc_id: {doc_id!r}")


def documents_dir(data_dir: Path) -> Path:
    """Directory holding the sample documents and their manifest."""
    return data_dir / DOCUMENTS_SUBDIR


def load_manifest(data_dir: Path) -> DocumentManifest:
    """Read and validate the manifest."""
    path = documents_dir(data_dir) / MANIFEST_NAME
    return DocumentManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_document(entry: ManifestEntry, data_dir: Path) -> RawDocument:
    """Read one document's file into a ``RawDocument``."""
    content = (documents_dir(data_dir) / entry.file).read_text(encoding="utf-8")
    return RawDocument(doc_id=entry.doc_id, content=content, format=entry.format)
