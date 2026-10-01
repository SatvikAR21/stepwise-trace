"""Command-line entrypoint: ``uv run python -m app.pipeline.cli --help``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.analysis.analyzer import analyze_trace
from app.analysis.evaluation import (
    DocResult,
    EvaluationReport,
    evaluate_corpus,
    save_report,
    wilson_interval,
)
from app.analysis.judge import JudgeFailedError
from app.analysis.models import Analysis
from app.analysis.store import AnalysisStore, analyses_dir
from app.analysis.taxonomy import FailureCategory
from app.core.config import LLMProvider, Settings, get_settings
from app.core.logging import configure_logging
from app.llm.base import LLMClient
from app.llm.factory import MOCK_SCRIPTS_SUBDIR, build_judge_client, build_llm_client
from app.llm.mock import MockLLMClient
from app.pipeline.documents import (
    CorpusSplit,
    DocumentManifest,
    ManifestEntry,
    load_document,
    load_manifest,
)
from app.pipeline.models import PipelineConfig, PipelineResult, PipelineStatus, StepName
from app.tracing.models import Trace, TraceStatus
from app.tracing.service import trace_pipeline
from app.tracing.store import TraceStore

EVALUATIONS_SUBDIR = "evaluations"  # inside the traces directory
SCRIPTED_BANNER = (
    "SCRIPTED JUDGE (JUDGE_PROVIDER=mock): its verdicts were written to agree with the answer "
    "key, so this is NOT a measurement."
)
_SHORT_CATEGORY = {
    FailureCategory.EXTRACTION_HALLUCINATION: "hallucination",
    FailureCategory.MISCLASSIFICATION: "misclassification",
    FailureCategory.PROPAGATION_ERROR: "propagation",
    FailureCategory.PROMPT_FAILURE: "prompt",
    FailureCategory.CONTEXT_LOSS: "context_loss",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.pipeline.cli", description=__doc__)
    parser.add_argument(
        "--provider",
        choices=[p.value for p in LLMProvider],
        help="override LLM_PROVIDER for this run",
    )
    parser.add_argument(
        "--judge-provider",
        choices=[p.value for p in LLMProvider],
        help="override JUDGE_PROVIDER for this run (mock = scripted verdicts, no requests)",
    )
    parser.add_argument("--judge-model", help="override JUDGE_MODEL for this run")
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
    analyze = sub.add_parser(
        "analyze", help="diagnose the root cause of a saved trace with the configured judge"
    )
    which = analyze.add_mutually_exclusive_group(required=True)
    which.add_argument("trace_id", nargs="?", help="a saved trace id")
    which.add_argument("--doc", dest="doc_id", help="the latest saved trace of this document")
    analyze.add_argument("--json", action="store_true", help="print the full analysis as JSON")
    evaluate = sub.add_parser(
        "evaluate",
        help="run corpus documents through the scripted pipeline, diagnose each run and grade "
        "the diagnoses against the answer key",
    )
    evaluate.add_argument(
        "--split",
        choices=[*(s.value for s in CorpusSplit), "all"],
        default=CorpusSplit.PRACTICE.value,
        help="which documents to grade (default: practice)",
    )
    evaluate.add_argument("--docs", help="only these comma-separated document ids")
    evaluate.add_argument(
        "--max-calls",
        type=_positive_int,
        help="hard limit on judge requests, repairs included (required for a real judge)",
    )
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


# --------------------------------------------------------------------------- analyze


def _cmd_analyze(args: argparse.Namespace, settings: Settings, store: TraceStore) -> int:
    trace = store.get(args.trace_id) if args.trace_id else _latest_trace(store, args.doc_id)
    if trace is None:
        print(f"no saved trace for {args.trace_id or args.doc_id!r}", file=sys.stderr)
        return 1
    judge = build_judge_client(settings)
    _announce_judge(settings, judge, f"at most {1 + settings.judge_max_repair_attempts} request(s)")
    analyses = AnalysisStore(analyses_dir(settings.traces_dir), settings.database_path)
    try:
        analysis = analyze_trace(
            trace,
            judge,
            drop_score=settings.judge_drop_score,
            temperature=settings.judge_temperature,
            max_repair_attempts=settings.judge_max_repair_attempts,
            reuse=analyses.verdict_for,
        )
        path = analyses.save(analysis)
    except JudgeFailedError as exc:
        print(f"the judge gave no usable verdict: {exc}", file=sys.stderr)
        return 1
    finally:
        analyses.close()
    if args.json:
        print(analysis.model_dump_json(indent=2))
    else:
        _print_analysis(analysis, scripted=settings.judge_provider is LLMProvider.MOCK)
    print(f"analysis {analysis.analysis_id} saved to {path}", file=sys.stderr)
    return 0


def _latest_trace(store: TraceStore, doc_id: str) -> Trace | None:
    page = store.list_traces(doc_id=doc_id, limit=1)
    return store.get(page.items[0].trace_id) if page.items else None


def _print_analysis(analysis: Analysis, *, scripted: bool) -> None:
    print(analysis.summary)
    print(f"\n{'STEP':<15} {'ROLE':<11} {'SCORE':<6} CATEGORY")
    for finding in analysis.steps:
        print(
            f"{finding.step.value:<15} {finding.role.value:<11} {finding.score or '-'!s:<6} "
            f"{finding.category.value if finding.category else '-'}"
        )
    print("\nAutomatic checks (a second opinion, not part of the verdict):")
    for check in analysis.checks:
        where = check.step.value if check.step else "document"
        print(f"  [{'FLAG' if check.flagged else 'ok'}] {check.name} ({where}): {check.detail}")
    cost = "verdict reused" if analysis.reused_verdict else f"{len(analysis.judge_calls)} call(s)"
    print(
        f"\nJudge: {analysis.judge_model}, prompt {analysis.judge_prompt_version}, {cost}, "
        f"{analysis.duration_ms:.0f} ms"
    )
    if scripted:
        print(SCRIPTED_BANNER)


def _announce_judge(settings: Settings, judge: LLMClient, budget: str) -> None:
    if settings.judge_provider is LLMProvider.MOCK:
        return
    print(
        f"Real judge {judge.model_name} at {settings.judge_endpoint or 'the default endpoint'}: "
        f"{budget}; saved verdicts for identical requests are reused.",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- evaluate


def _cmd_evaluate(
    args: argparse.Namespace, manifest: DocumentManifest, settings: Settings, store: TraceStore
) -> int:
    entries = _select(manifest, args.split, args.docs)
    if isinstance(entries, str):
        print(entries, file=sys.stderr)
        return 1
    real = settings.judge_provider is not LLMProvider.MOCK
    if real and args.max_calls is None:
        print(
            "a real judge needs --max-calls N: a hard limit on judge requests (repairs included)",
            file=sys.stderr,
        )
        return 1
    judge = build_judge_client(settings)
    _announce_judge(settings, judge, f"at most {args.max_calls} new request(s) in this run")
    if not real:
        print(SCRIPTED_BANNER)
    print(
        "The pipeline always runs on the scripted mock LLM here: the planted failures live in its "
        "scripts.\n"
    )
    print(f"{'DOC_ID':<34} {'SPLIT':<9} {'PLANTED':<28} {'DIAGNOSED':<28} {'OUTCOME':<12} CALLS")
    analyses = AnalysisStore(analyses_dir(settings.traces_dir), settings.database_path)
    try:
        report = evaluate_corpus(
            entries,
            data_dir=settings.data_dir,
            pipeline_llm=MockLLMClient(scripts_dir=settings.data_dir / MOCK_SCRIPTS_SUBDIR),
            judge=judge,
            trace_store=store,
            analysis_store=analyses,
            config=_pipeline_config(settings),
            drop_score=settings.judge_drop_score,
            temperature=settings.judge_temperature,
            max_repair_attempts=settings.judge_max_repair_attempts,
            max_calls=args.max_calls,
            on_result=_print_row,
        )
    finally:
        analyses.close()
    path = save_report(report, settings.traces_dir / EVALUATIONS_SUBDIR)
    _print_report_card(report)
    print(f"\nreport saved to {path}")
    return 2 if report.stopped or report.totals.not_judged else 0


def _select(manifest: DocumentManifest, split: str, docs: str | None) -> list[ManifestEntry] | str:
    entries = [e for e in manifest.documents if split == "all" or e.split.value == split]
    if docs:
        wanted = [d.strip() for d in docs.split(",") if d.strip()]
        known = {e.doc_id for e in entries}
        unknown = [d for d in wanted if d not in known]
        if unknown:
            return f"not in the {split} documents: {', '.join(unknown)}"
        entries = [e for e in entries if e.doc_id in wanted]
    return entries or f"no documents selected (split {split})"


def _print_row(row: DocResult) -> None:
    planted = _step_and_category(row.planted_step, row.planted_category)
    found = _step_and_category(row.diagnosed_step, row.diagnosed_category)
    if row.outcome.value == "not-judged":
        found = f"({row.note})"[:28]
    calls = "reused" if row.reused_verdict else str(row.judge_calls)
    print(
        f"{row.doc_id:<34} {row.split.value:<9} {planted:<28} {found:<28} "
        f"{row.outcome.value:<12} {calls}"
    )


def _step_and_category(step: StepName | None, category: FailureCategory | None) -> str:
    if step is None:
        return "healthy"
    return f"{step.value}/{_SHORT_CATEGORY[category] if category else '?'}"


def _print_report_card(report: EvaluationReport) -> None:
    t = report.totals
    splits = " + ".join(s.value for s in report.splits)
    print(
        f"\nREPORT CARD ({splits}; judge {report.judge_model}, prompt "
        f"{report.judge_prompt_version}, drop line {report.drop_score})"
    )
    print(f"Broken documents: {t.broken} ({t.broken_judged} judged)")
    for label, count in (
        ("right step", t.right_step),
        ("right category", t.right_category),
        ("both right", t.correct),
        ("missed (no root cause)", t.missed),
    ):
        print(f"  {label:<24} {_rate(count, t.broken_judged)}")
    print(f"Healthy documents: {t.healthy} ({t.healthy_judged} judged)")
    print(f"  {'false alarms':<24} {_rate(t.false_alarms, t.healthy_judged)}")
    print(
        f"Automatic checks alone: flagged {t.checks_caught}/{t.broken_judged} broken "
        f"({t.checks_right_step} at the planted step), {t.checks_false_alarms}/"
        f"{t.healthy_judged} healthy"
    )
    print(
        f"Cost: {t.judge_calls} judge call(s), {t.reused_verdicts} verdict(s) reused, "
        f"{t.prompt_tokens:,} prompt + {t.completion_tokens:,} completion tokens, "
        f"judge time {t.judge_latency_ms / 1000:.1f} s"
    )
    if report.stopped:
        print(
            f"\nSTOPPED EARLY: {report.stop_detail}. Finished documents are saved; run the same "
            "command again after the provider's limit resets and they will be reused, not re-asked."
        )
    elif t.not_judged:
        print(
            f"\n{t.not_judged} document(s) not judged (see the table). Run again to continue; "
            "finished documents are reused, not re-asked."
        )
    if report.scripted_judge:
        print(SCRIPTED_BANNER)


def _rate(count: int, total: int) -> str:
    if total == 0:
        return f"{count}/0"
    low, high = wilson_interval(count, total)
    return f"{count}/{total} ({count / total:.0%}; 95% range {low:.0%}-{high:.0%})"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the command, return the process exit code."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": LLMProvider(args.provider)})
    if args.judge_provider:
        settings = settings.model_copy(update={"judge_provider": LLMProvider(args.judge_provider)})
    if args.judge_model:
        settings = settings.model_copy(update={"judge_model": args.judge_model})
    configure_logging("WARNING", settings.log_format)

    store = TraceStore(settings.traces_dir, settings.database_path)
    try:
        if args.command == "traces":
            return _cmd_traces(args, store)
        if args.command == "analyze":
            return _cmd_analyze(args, settings, store)
        manifest = load_manifest(settings.data_dir)
        if args.command == "list":
            return _cmd_list(manifest)
        if args.command == "evaluate":
            return _cmd_evaluate(args, manifest, settings, store)
        llm = build_llm_client(settings)
        if args.all:
            return _run_all(manifest, llm, settings, store)
        return _run_one(args.doc_id, manifest, llm, settings, store)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
