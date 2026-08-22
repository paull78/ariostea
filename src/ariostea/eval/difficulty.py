"""The difficulty guard: a dense-only baseline per cluster, and a flag for the
clusters too easy to measure anything on.

The corpus this replaces failed for exactly one reason -- with one plausible
target per query, every method scored near 1.0 and no improvement was visible.
Clusters are the unit because difficulty is a property of a cluster's internal
similarity, not of the corpus as a whole: `board-games` can be trivially easy
while `string-instruments` is hard, and an overall average hides that.

Dense-only is the baseline because it is the channel needing no lexical
overlap. If a plain embedding lookup already finds the answer nineteen times
in twenty, nothing downstream -- fusion, reranking, contextual blurbs -- has
room left to show a difference on that cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.harness import SpanSearchFn, dedupe
from ariostea.eval.metrics import recall_at_k
from ariostea.eval.span_metrics import span_recall_at_k
from ariostea.eval.wiki_gold import WikiGoldCase
from ariostea.eval.wiki_notes import cluster_of

# The design doc's "flag clusters scoring >= ~0.95 as too easy".
EASY_THRESHOLD = 0.95


@dataclass(frozen=True)
class ClusterBaseline:
    cluster: str
    n: int
    note_recall_at_k: float
    span_recall_at_k: float


def cluster_baselines(
    cases: list[WikiGoldCase], span_fn: SpanSearchFn, k: int, pool: int = 50
) -> tuple[ClusterBaseline, ...]:
    """One baseline per cluster, in cluster-name order.

    A case's cluster is the first path segment of its first expected note. A
    case with no expected notes raises rather than landing in a `""` bucket:
    `validate_wiki_gold` already rejects that shape, so reaching here means
    the gold was assembled by a path that skipped validation, and quietly
    inventing an empty cluster would hide it.

    Both recalls are reported. Note-level is what the flag reads -- it answers
    "can the dense channel find the right article at all", which is the sense
    in which the old corpus was too easy. Span-level is reported alongside
    because a cluster can be easy to find and still hard to answer, and that
    gap is worth seeing rather than averaging away.
    """
    grouped: dict[str, list[tuple[float, float]]] = {}
    for case in cases:
        if not case.expected_notes:
            raise ValueError(f"case {case.query!r} has no expected_notes to take a cluster from")
        retrieved = span_fn(case.query, pool)
        notes = dedupe([note for note, _ in retrieved])
        grouped.setdefault(cluster_of(case.expected_notes[0]), []).append(
            (
                recall_at_k(set(case.expected_notes), notes, k),
                span_recall_at_k(case.answer_spans, retrieved, k),
            )
        )
    return tuple(
        ClusterBaseline(
            cluster=cluster,
            n=len(rows),
            note_recall_at_k=sum(note for note, _ in rows) / len(rows),
            span_recall_at_k=sum(span for _, span in rows) / len(rows),
        )
        for cluster, rows in sorted(grouped.items())
    )


def flag_easy_clusters(
    baselines: list[ClusterBaseline] | tuple[ClusterBaseline, ...],
    threshold: float = EASY_THRESHOLD,
) -> tuple[str, ...]:
    """Clusters whose dense-only note recall reaches `threshold`.

    Inclusive at the threshold: a cluster sitting exactly on the line is the
    case the guard exists to surface, not one to wave through.
    """
    return tuple(b.cluster for b in baselines if b.note_recall_at_k >= threshold)


def format_baselines(
    baselines: list[ClusterBaseline] | tuple[ClusterBaseline, ...],
    threshold: float = EASY_THRESHOLD,
) -> str:
    flagged = set(flag_easy_clusters(baselines, threshold))
    lines = [f"{'cluster':<22} {'n':>3}  {'note_recall':>11}  {'span_recall':>11}"]
    for baseline in baselines:
        mark = "  <-- TOO EASY" if baseline.cluster in flagged else ""
        lines.append(
            f"{baseline.cluster:<22} {baseline.n:>3}  "
            f"{baseline.note_recall_at_k:>11.3f}  {baseline.span_recall_at_k:>11.3f}{mark}"
        )
    return "\n".join(lines)
