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
from app.pipeline.models import PipelineConfig, PipelineResult, PipelineStatus
from app.tracing.models import Trace, TraceStatus
from app.tracing.service import trace_pipeline
from app.tracing.store import TraceStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.pipeline.cli", description=__doc__)
    parser.add_argument(
        "--provider",
        choices=[p.value for p in LLMProvider],
        help="override LLM_PROVIDER for this run",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the sample documents")
    run = sub.add_parser(
        "run", help="run the pipeline on one document, or all with --all, recording traces"
    )
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("doc_id", nargs="?", help="document id from the manifest")
    target.add_argument("--all", action="store_true", help="run every document")
    traces = sub.add_parser("traces", help="list or show recorded traces")
    traces_sub = traces.add_subparsers(dest="traces_command", required=True)
    listing = traces_sub.add_parser("list", help="recorded traces, newest first")
    listing.add_argument("--status", choices=[s.value for s in TraceStatus])
    listing.add_argument("--doc", dest="doc_id", help="only traces of this document")
    listing.add_argument(
        "--limit", type=_positive_int, default=20, help="how many to show (default 20)"
    )
    show = traces_sub.add_parser("show", help="print one trace as JSON")
    show.add_argument("trace_id")
    return parser


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def _cmd_list(manifest: DocumentManifest) -> int:
    print(f"{'DOC_ID':<34} {'FORMAT':<9} {'EXPECTED':<15} INTENDED FAILURE")
    for e in manifest.documents:
        print(
            f"{e.doc_id:<34} {e.format.value:<9} {e.expected_type.value:<15} "
            f"{e.intended_failure or '-'}"
        )
    return 0


def _run_one(
    doc_id: str, manifest: DocumentManifest, llm: LLMClient, settings: Settings, store: TraceStore
) -> int:
    document = load_document(manifest.get(doc_id), settings.data_dir)
    result, trace = trace_pipeline(document, llm, _pipeline_config(settings))
    path = store.save(trace)
    print(result.model_dump_json(indent=2, exclude_none=True))
    print(f"trace {trace.trace_id}: {_verdict(trace)}\nsaved to {path}", file=sys.stderr)
    return 0 if result.status is PipelineStatus.COMPLETED else 1


def _run_all(
    manifest: DocumentManifest, llm: LLMClient, settings: Settings, store: TraceStore
) -> int:
    print(
        f"{'DOC_ID':<34} {'RUN':<10} {'TRACE':<9} {'SCORE':<6} {'EXPECTED':<15} "
        f"{'CLASSIFIED':<15} NOTE"
    )
    failures = degraded = 0
    for entry in manifest.documents:
        result, trace = trace_pipeline(
            load_document(entry, settings.data_dir), llm, _pipeline_config(settings)
        )
        store.save(trace)
        failures += result.status is PipelineStatus.FAILED
        degraded += trace.status is TraceStatus.DEGRADED
        print(
            f"{entry.doc_id:<34} {result.status.value:<10} {trace.status.value:<9} "
            f"{_score(trace):<6} {entry.expected_type.value:<15} {_classified(result):<15} "
            f"{_note(result, entry.expected_type.value)}"
        )
    print(
        f"\n{len(manifest.documents)} documents, {failures} pipeline error(s), "
        f"{degraded} degraded; traces saved to {settings.traces_dir}"
    )
    return 0 if failures == 0 else 1


def _cmd_traces(args: argparse.Namespace, store: TraceStore) -> int:
    if args.traces_command == "show":
        trace = store.get(args.trace_id)
        if trace is None:
            print(f"no trace with id {args.trace_id!r}", file=sys.stderr)
            return 1
        print(trace.model_dump_json(indent=2))
        return 0
    page = store.list_traces(
        status=TraceStatus(args.status) if args.status else None,
        doc_id=args.doc_id,
        limit=args.limit,
    )
    print(f"{'TRACE_ID':<32}  {'STARTED (UTC)':<19}  {'DOC_ID':<34} {'STATUS':<9} SCORE")
    for item in page.items:
        print(
            f"{item.trace_id}  {item.started_at:%Y-%m-%d %H:%M:%S}  {item.doc_id:<34} "
            f"{item.status.value:<9} {item.final_score or '-'}"
        )
    print(f"\n{len(page.items)} of {page.total} trace(s)")
    return 0


def _verdict(trace: Trace) -> str:
    verdict = f"{trace.status.value} (score {_score(trace)})"
    return f"{verdict}: {'; '.join(trace.status_reasons)}" if trace.status_reasons else verdict


def _score(trace: Trace) -> str:
    return str(trace.final_score) if trace.final_score is not None else "-"


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
        intake_max_chars=settings.intake_max_chars,
        temperature=settings.llm_temperature,
        max_repair_attempts=settings.llm_max_repair_attempts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the command, return the process exit code."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": LLMProvider(args.provider)})
    configure_logging("WARNING", settings.log_format)

    store = TraceStore(settings.traces_dir, settings.database_path)
    try:
        if args.command == "traces":
            return _cmd_traces(args, store)
        manifest = load_manifest(settings.data_dir)
        if args.command == "list":
            return _cmd_list(manifest)
        llm = build_llm_client(settings)
        if args.all:
            return _run_all(manifest, llm, settings, store)
        return _run_one(args.doc_id, manifest, llm, settings, store)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
