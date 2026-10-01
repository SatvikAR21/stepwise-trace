"""Cheap automatic checks on a trace: plain code, no LLM, a second opinion next to the judge.

They reuse the grounding clues recorded since tracing began. They advise, they never decide:
some raise false alarms (a correctly computed number is not written in the document) and some
miss whole kinds of failure (an omission invents nothing). Each check names the step it points
at, or ``None`` when it concerns the document itself.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from app.pipeline.features import is_grounded, numbers_in, summary_numbers
from app.pipeline.models import StepName
from app.tracing.models import SpanStatus, Trace

_INSTRUCTION_RE = re.compile(
    r"ignore (?:all |any )?(?:previous|prior|above) instructions"
    r"|(?:note|message|instruction)s? (?:to|for) (?:the |an |any )?"
    r"(?:ai|assistant|automated|model|system)"
    r"|system note",
    re.IGNORECASE,
)


class CheckResult(BaseModel):
    """The outcome of one automatic check."""

    name: str
    step: StepName | None
    flagged: bool
    detail: str


def run_checks(trace: Trace) -> list[CheckResult]:
    """Every check that applies to the steps that ran, flagged or not."""
    spans = {StepName(span.name): span for span in trace.spans}
    intake = spans.get(StepName.INTAKE)
    document = (intake.input or {}).get("content", "") if intake else ""
    results = [_instruction_like_text(str(document))]
    for step, span in spans.items():
        if span.status is SpanStatus.ERROR:
            error = span.error.error_type if span.error else "error"
            results.append(CheckResult(name="step_failed", step=step, flagged=True, detail=error))
    if intake is not None and intake.status is SpanStatus.OK:
        results.append(_truncated(intake.output))
    extraction = spans.get(StepName.EXTRACTION)
    if extraction is not None and extraction.status is SpanStatus.OK:
        results.append(_ungrounded_entities(extraction.input, extraction.output))
    summary = spans.get(StepName.SUMMARIZATION)
    if summary is not None and summary.status is SpanStatus.OK:
        results.append(_summary_numbers_not_in_document(summary.input, summary.output))
        results.append(_mixed_currency_total(summary.input, summary.output))
    return results


def _instruction_like_text(content: str) -> CheckResult:
    found = sorted({match.group(0) for match in _INSTRUCTION_RE.finditer(content)})
    return CheckResult(
        name="instruction_like_text",
        step=None,
        flagged=bool(found),
        detail=f"the document contains: {found}" if found else "none found",
    )


def _truncated(output: dict[str, Any]) -> CheckResult:
    return CheckResult(
        name="input_truncated",
        step=StepName.INTAKE,
        flagged=bool(output.get("truncated")),
        detail=f"kept {output.get('char_count')} of {output.get('original_char_count')} characters",
    )


def _ungrounded_entities(step_input: dict[str, Any], output: dict[str, Any]) -> CheckResult:
    text = str(step_input.get("text", ""))
    mentions = [
        *(p["name"] for p in output.get("people", [])),
        *(o["name"] for o in output.get("organizations", [])),
        *(d["raw"] for d in output.get("dates", [])),
        *(a["raw"] for a in output.get("amounts", [])),
    ]
    missing = [m for m in mentions if not is_grounded(m, text)]
    return CheckResult(
        name="ungrounded_entities",
        step=StepName.EXTRACTION,
        flagged=bool(missing),
        detail=f"not found in the text: {missing}" if missing else f"all {len(mentions)} found",
    )


def _summary_numbers_not_in_document(
    step_input: dict[str, Any], output: dict[str, Any]
) -> CheckResult:
    in_document = numbers_in(str(step_input["document"]["text"]))
    missing = sorted(summary_numbers(output) - in_document)
    return CheckResult(
        name="summary_numbers_not_in_document",
        step=StepName.SUMMARIZATION,
        flagged=bool(missing),
        detail=f"not in the document: {[_plain(n) for n in missing]}" if missing else "none",
    )


def _mixed_currency_total(step_input: dict[str, Any], output: dict[str, Any]) -> CheckResult:
    currencies = sorted({a["currency"] for a in step_input["entities"].get("amounts", [])})
    total = output.get("total_amount")
    if total is None or len(currencies) < 2:
        return CheckResult(
            name="mixed_currency_total",
            step=StepName.SUMMARIZATION,
            flagged=False,
            detail="no single total over several currencies",
        )
    return CheckResult(
        name="mixed_currency_total",
        step=StepName.SUMMARIZATION,
        flagged=True,
        detail=f"one total ({total['raw']}) although the amounts are in {', '.join(currencies)}",
    )


def _plain(number: Decimal) -> str:
    return format(number.normalize(), "f")
