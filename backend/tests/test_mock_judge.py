"""The pretend judge's scripted verdicts, run through the whole analyzer on the corpus.

The pretend judge is ideal by construction, so these tests check the analyzer and the scripts,
not how good a real judge is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.analyzer import analyze_trace
from app.analysis.judge import build_case_file
from app.analysis.models import JudgeAnswer
from app.analysis.taxonomy import FailureCategory
from app.core.config import BACKEND_DIR, Settings
from app.llm.factory import build_judge_client
from app.llm.mock import MockLLMClient
from app.pipeline.documents import DocumentManifest, load_document, load_manifest
from app.pipeline.models import StepName
from app.tracing.models import Trace
from app.tracing.service import trace_pipeline

DATA_DIR = BACKEND_DIR / "data"


@pytest.fixture(scope="module")
def manifest() -> DocumentManifest:
    return load_manifest(DATA_DIR)


@pytest.fixture(scope="module")
def corpus_traces(manifest: DocumentManifest) -> dict[str, Trace]:
    llm = MockLLMClient(scripts_dir=DATA_DIR / "mock_responses")
    return {
        entry.doc_id: trace_pipeline(load_document(entry, DATA_DIR), llm)[1]
        for entry in manifest.documents
    }


def _script(doc_id: str) -> JudgeAnswer:
    data = json.loads((DATA_DIR / "mock_judge" / f"{doc_id}.json").read_text(encoding="utf-8"))
    return JudgeAnswer.model_validate(data["judge"])


def test_every_document_has_a_pretend_verdict(manifest: DocumentManifest) -> None:
    files = {path.stem for path in (DATA_DIR / "mock_judge").glob("*.json")}

    assert files == {entry.doc_id for entry in manifest.documents}


def test_the_pretend_judge_reproduces_the_answer_key_on_every_document(
    manifest: DocumentManifest, corpus_traces: dict[str, Trace]
) -> None:
    judge = build_judge_client(Settings(_env_file=None, data_dir=DATA_DIR))

    for entry in manifest.documents:
        analysis = analyze_trace(corpus_traces[entry.doc_id], judge)

        expected_step = StepName(entry.failing_step) if entry.failing_step else None
        expected_category = (
            FailureCategory(entry.intended_failure) if entry.intended_failure else None
        )
        assert analysis.root_cause_step is expected_step, entry.doc_id
        assert analysis.category is expected_category, entry.doc_id


def test_every_quoted_piece_of_evidence_is_really_in_the_case_file(
    manifest: DocumentManifest, corpus_traces: dict[str, Trace]
) -> None:
    for entry in manifest.documents:
        case_file = build_case_file(corpus_traces[entry.doc_id])
        for grade in _script(entry.doc_id).steps:
            for quote in grade.evidence:
                assert quote in case_file, (entry.doc_id, quote)


def test_pretend_verdicts_cover_exactly_the_steps_that_ran(
    manifest: DocumentManifest, corpus_traces: dict[str, Trace]
) -> None:
    for entry in manifest.documents:
        expected = [StepName(span.name) for span in corpus_traces[entry.doc_id].spans]
        assert _script(entry.doc_id).problems_for(expected) == [], entry.doc_id


def test_mock_judge_folder_is_where_the_factory_looks(tmp_path: Path) -> None:
    judge = build_judge_client(Settings(_env_file=None, data_dir=tmp_path))

    assert isinstance(judge, MockLLMClient)
    assert judge._scripts_dir == tmp_path / "mock_judge"
