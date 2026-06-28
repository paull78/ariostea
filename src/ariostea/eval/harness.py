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
    scenario: str  # "same" | "en→it" | … | "accent" | "inflection"


@dataclass(frozen=True)
class ScenarioScore:
    scenario: str
    n: int
    recall_at_k: float
    mrr: float


@dataclass(frozen=True)
class EvalReport:
    k: int
    overall: ScenarioScore
    by_scenario: tuple[ScenarioScore, ...]


def load_gold(path: str | Path) -> list[GoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldCase(
            query=row["query"],
            query_lang=row["query_lang"],
            expected=tuple(row["expected"]),
            scenario=row["scenario"],
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


def _aggregate(scenario: str, rows: list[tuple[float, float]]) -> ScenarioScore:
    n = len(rows)
    if n == 0:
        return ScenarioScore(scenario=scenario, n=0, recall_at_k=0.0, mrr=0.0)
    return ScenarioScore(
        scenario=scenario,
        n=n,
        recall_at_k=sum(recall for recall, _ in rows) / n,
        mrr=sum(rr for _, rr in rows) / n,
    )


def evaluate(cases: list[GoldCase], search_fn: SearchFn, k: int) -> EvalReport:
    """Run every gold case through search_fn once and aggregate recall@k / MRR
    overall and per scenario. search_fn must return deduped note paths."""
    scored: list[tuple[str, float, float]] = []
    for case in cases:
        ranked = search_fn(case.query, k)
        expected = set(case.expected)
        scored.append(
            (case.scenario, recall_at_k(expected, ranked, k), reciprocal_rank(expected, ranked))
        )

    overall = _aggregate("overall", [(r, rr) for _, r, rr in scored])
    scenarios = sorted({scenario for scenario, _, _ in scored})
    by_scenario = tuple(
        _aggregate(s, [(r, rr) for scenario, r, rr in scored if scenario == s])
        for s in scenarios
    )
    return EvalReport(k=k, overall=overall, by_scenario=by_scenario)


def format_report(report: EvalReport) -> str:
    header = f"{'scenario':<12} {'n':>3}  recall@{report.k:<3}  mrr"
    lines = [header]
    for s in (*report.by_scenario, report.overall):
        lines.append(f"{s.scenario:<12} {s.n:>3}  {s.recall_at_k:>8.3f}  {s.mrr:.3f}")
    return "\n".join(lines)
