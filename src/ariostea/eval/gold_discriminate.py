"""Validation stage 3: drop the queries that carry no experimental signal.

A query every retrieval channel already answers at rank 1 cannot distinguish
two methods -- it reads 1.0 before a change and 1.0 after it, and its only
effect on the eval is to raise every number and shrink every visible
difference. That is precisely the flaw the old corpus had, so it is worth
spending model calls on candidates that then get thrown away here rather than
shipping a gold set that scores well and measures nothing.

Note what is *not* dropped: a query no channel answers. Hard is not useless --
an unanswered case is the one an improvement can actually move.
"""

from __future__ import annotations

from ariostea.eval.harness import SpanSearchFn
from ariostea.eval.span_metrics import span_recall_at_k
from ariostea.eval.wiki_gold import WikiGoldCase


def discrimination_filter(
    cases: list[WikiGoldCase], channels: dict[str, SpanSearchFn]
) -> tuple[list[WikiGoldCase], list[WikiGoldCase]]:
    """Split `cases` into `(kept, dropped)`, preserving order in both.

    "Answered" means the top-ranked chunk is in the span's own note *and*
    contains the span -- the same containment rule the span metrics use, so a
    case dropped here really is one the eval would score 1.0.

    Raises on an empty `channels` map: `all()` over an empty sequence is
    vacuously true, so every case would be classified "answered by every
    channel" and the run would write an empty gold file while reporting
    success.
    """
    if not channels:
        raise ValueError("discrimination_filter needs at least one channel to judge against")

    kept: list[WikiGoldCase] = []
    dropped: list[WikiGoldCase] = []
    for case in cases:
        answered_by_all = all(
            span_recall_at_k(case.answer_spans, search_fn(case.query, 1), 1) == 1.0
            for search_fn in channels.values()
        )
        (dropped if answered_by_all else kept).append(case)
    return kept, dropped
