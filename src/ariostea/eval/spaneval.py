"""Span-level evaluation: run gold cases through a chunk-returning search
function and report retrieval quality at two granularities — note-level (did we
find the right note) and span-level (did a retrieved chunk actually contain the
answer) — overall and grouped by query type.
"""

from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.harness import SpanSearchFn, dedupe
from ariostea.eval.metrics import recall_at_k, reciprocal_rank
from ariostea.eval.span_metrics import span_recall_at_k, span_reciprocal_rank
from ariostea.eval.wiki_gold import WikiGoldCase


@dataclass(frozen=True)
class SpanScore:
    type: str
    n: int
    note_recall_at_k: float
    note_mrr: float
    span_recall_at_k: float
    span_mrr: float


@dataclass(frozen=True)
class SpanEvalReport:
    k: int
    overall: SpanScore
    by_type: tuple[SpanScore, ...]


# Each scored row is (note_recall, note_mrr, span_recall, span_mrr).
def _aggregate(type_: str, rows: list[tuple[float, float, float, float]]) -> SpanScore:
    n = len(rows)
    if n == 0:
        return SpanScore(type_, 0, 0.0, 0.0, 0.0, 0.0)
    return SpanScore(
        type=type_,
        n=n,
        note_recall_at_k=sum(r[0] for r in rows) / n,
        note_mrr=sum(r[1] for r in rows) / n,
        span_recall_at_k=sum(r[2] for r in rows) / n,
        span_mrr=sum(r[3] for r in rows) / n,
    )


def evaluate_spans(
    cases: list[WikiGoldCase], span_fn: SpanSearchFn, k: int, pool: int = 50
) -> SpanEvalReport:
    """Run every case through span_fn once and aggregate note-level and
    span-level recall@k / MRR, overall and grouped by query type.

    span_fn is asked for a generous `pool` of ranked chunks (pool >= k). Span
    metrics use the top-k chunks; note metrics dedupe the full pool to notes and
    take the top-k distinct notes — so note-level scoring counts k *notes*, not k
    *chunks*, and stays comparable to the note-level channels regardless of how
    many chunks a single note contributes.
    """
    scored: list[tuple[str, tuple[float, float, float, float]]] = []
    for case in cases:
        retrieved = span_fn(case.query, pool)
        notes = dedupe([note for note, _ in retrieved])
        expected = set(case.expected_notes)
        row = (
            recall_at_k(expected, notes, k),
            reciprocal_rank(expected, notes),
            span_recall_at_k(case.answer_spans, retrieved, k),
            span_reciprocal_rank(case.answer_spans, retrieved),
        )
        scored.append((case.type, row))

    overall = _aggregate("overall", [row for _, row in scored])
    types = sorted({t for t, _ in scored})
    by_type = tuple(_aggregate(t, [row for typ, row in scored if typ == t]) for t in types)
    return SpanEvalReport(k=k, overall=overall, by_type=by_type)


def format_span_report(report: SpanEvalReport) -> str:
    header = f"{'type':<14} {'n':>3}  note_r@{report.k:<2} note_mrr  span_r@{report.k:<2} span_mrr"
    lines = [header]
    for s in (*report.by_type, report.overall):
        lines.append(
            f"{s.type:<14} {s.n:>3}  "
            f"{s.note_recall_at_k:>7.3f} {s.note_mrr:>7.3f}  "
            f"{s.span_recall_at_k:>7.3f} {s.span_mrr:>7.3f}"
        )
    return "\n".join(lines)
