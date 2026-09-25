"""Command-line entrypoint: ``uv run python -m app.pipeline.cli --help``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.core.config import LLMProvider, Settings, get_settings
from app.core.logging import configure_logging
from app.llm.base import LLMClient
from app.llm.factory import build_llm_client
from app.pipeline.documents import DocumentManifest, load_document, load_manifest
from app.pipeline.models import PipelineResult, PipelineStatus
from app.pipeline.runner import PipelineConfig, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.pipeline.cli", description=__doc__)
    parser.add_argument(
        "--provider",
        choices=[p.value for p in LLMProvider],
        help="override LLM_PROVIDER for this run",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the sample documents")
    run = sub.add_parser("run", help="run the pipeline on one document, or all with --all")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("doc_id", nargs="?", help="document id from the manifest")
    target.add_argument("--all", action="store_true", help="run every document")
    return parser


def _cmd_list(manifest: DocumentManifest) -> int:
    print(f"{'DOC_ID':<34} {'FORMAT':<9} {'EXPECTED':<15} INTENDED FAILURE")
    for e in manifest.documents:
        print(
            f"{e.doc_id:<34} {e.format.value:<9} {e.expected_type.value:<15} "
            f"{e.intended_failure or '-'}"
        )
    return 0


def _run_one(doc_id: str, manifest: DocumentManifest, llm: LLMClient, settings: Settings) -> int:
    document = load_document(manifest.get(doc_id), settings.data_dir)
    result = run_pipeline(document, llm, _pipeline_config(settings))
    print(result.model_dump_json(indent=2, exclude_none=True))
    return 0 if result.status is PipelineStatus.COMPLETED else 1


def _run_all(manifest: DocumentManifest, llm: LLMClient, settings: Settings) -> int:
    print(f"{'DOC_ID':<34} {'STATUS':<10} {'EXPECTED':<15} {'CLASSIFIED':<15} NOTE")
    failures = 0
    for entry in manifest.documents:
        result = run_pipeline(
            load_document(entry, settings.data_dir), llm, _pipeline_config(settings)
        )
        failures += result.status is PipelineStatus.FAILED
        print(
            f"{entry.doc_id:<34} {result.status.value:<10} {entry.expected_type.value:<15} "
            f"{_classified(result):<15} {_note(result, entry.expected_type.value)}"
        )
    print(f"\n{len(manifest.documents)} documents, {failures} pipeline error(s)")
    return 0 if failures == 0 else 1


def _classified(result: PipelineResult) -> str:
    return result.classification.document_type.value if result.classification else "-"


def _note(result: PipelineResult, expected: str) -> str:
    if result.error:
        return f"{result.error.step.value}: {result.error.error_type}"
    notes = []
    if result.normalized and result.normalized.truncated:
        notes.append("input truncated")
    if _classified(result) != expected:
        notes.append("type mismatch")
    return ", ".join(notes)


def _pipeline_config(settings: Settings) -> PipelineConfig:
    return PipelineConfig(
        intake_max_chars=settings.intake_max_chars, temperature=settings.llm_temperature
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the command, return the process exit code."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": LLMProvider(args.provider)})
    configure_logging("WARNING", settings.log_format)

    manifest = load_manifest(settings.data_dir)
    if args.command == "list":
        return _cmd_list(manifest)
    llm = build_llm_client(settings)
    if args.all:
        return _run_all(manifest, llm, settings)
    return _run_one(args.doc_id, manifest, llm, settings)


if __name__ == "__main__":
    sys.exit(main())
