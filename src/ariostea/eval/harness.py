from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ariostea.eval.metrics import recall_at_k, reciprocal_rank

# A ranker: given (query, k), return up to k note paths in rank order (best first).
SearchFn = Callable[[str, int], list[str]]


@dataclass(frozen=True)
class GoldCase:
    query: str
    query_lang: str
    expected: tuple[str, ...]
    direction: str  # "en→it" | "it→en" | "same"


@dataclass(frozen=True)
class DirectionScore:
    direction: str
    n: int
    recall_at_k: float
    mrr: float


@dataclass(frozen=True)
class EvalReport:
    k: int
    overall: DirectionScore
    by_direction: tuple[DirectionScore, ...]


def load_gold(path: str | Path) -> list[GoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldCase(
            query=row["query"],
            query_lang=row["query_lang"],
            expected=tuple(row["expected"]),
            direction=row["direction"],
        )
        for row in rows
    ]


def dedupe(paths: list[str]) -> list[str]:
    """Collapse a chunk-level path list to one entry per note, preserving order."""
    seen: list[str] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _aggregate(direction: str, rows: list[tuple[float, float]]) -> DirectionScore:
    n = len(rows)
    if n == 0:
        return DirectionScore(direction=direction, n=0, recall_at_k=0.0, mrr=0.0)
    return DirectionScore(
        direction=direction,
        n=n,
        recall_at_k=sum(recall for recall, _ in rows) / n,
        mrr=sum(rr for _, rr in rows) / n,
    )


def evaluate(cases: list[GoldCase], search_fn: SearchFn, k: int) -> EvalReport:
    """Run every gold case through search_fn once and aggregate recall@k / MRR
    overall and per direction. search_fn must return deduped note paths."""
    scored: list[tuple[str, float, float]] = []
    for case in cases:
        ranked = search_fn(case.query, k)
        expected = set(case.expected)
        scored.append(
            (case.direction, recall_at_k(expected, ranked, k), reciprocal_rank(expected, ranked))
        )

    overall = _aggregate("overall", [(r, rr) for _, r, rr in scored])
    directions = sorted({direction for direction, _, _ in scored})
    by_direction = tuple(
        _aggregate(d, [(r, rr) for direction, r, rr in scored if direction == d])
        for d in directions
    )
    return EvalReport(k=k, overall=overall, by_direction=by_direction)


def format_report(report: EvalReport) -> str:
    header = f"{'direction':<10} {'n':>3}  recall@{report.k:<3}  mrr"
    lines = [header]
    for d in (*report.by_direction, report.overall):
        lines.append(f"{d.direction:<10} {d.n:>3}  {d.recall_at_k:>8.3f}  {d.mrr:.3f}")
    return "\n".join(lines)
