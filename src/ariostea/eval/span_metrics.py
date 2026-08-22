"""Span-level retrieval metrics for the eval harness.

A retrieved chunk counts as a hit only if it is in the answer span's own note
*and* its text contains the span text, matching by normalized-whitespace,
case-insensitive **substring containment**. This ensures the same gold survives
re-chunking: a span may land in differently-sized chunks across policies.
"""

from __future__ import annotations

import math

from ariostea.eval.normalize import normalize_ws
from ariostea.eval.wiki_gold import AnswerSpan


def chunk_contains_span(chunk_text: str, span_text: str) -> bool:
    return normalize_ws(span_text) in normalize_ws(chunk_text)


def _is_hit(spans: tuple[AnswerSpan, ...], note_path: str, chunk_text: str) -> bool:
    return any(
        span.note == note_path and chunk_contains_span(chunk_text, span.text) for span in spans
    )


def span_recall_at_k(
    spans: tuple[AnswerSpan, ...], retrieved: list[tuple[str, str]], k: int
) -> float:
    """1.0 if any of the top-k retrieved (note_path, chunk_text) pairs contains
    an answer span in its own note, else 0.0."""
    return 1.0 if any(_is_hit(spans, note, text) for note, text in retrieved[:k]) else 0.0


def span_reciprocal_rank(spans: tuple[AnswerSpan, ...], retrieved: list[tuple[str, str]]) -> float:
    for index, (note, text) in enumerate(retrieved):
        if _is_hit(spans, note, text):
            return 1.0 / (index + 1)
    return 0.0


def span_ndcg_at_k(
    spans: tuple[AnswerSpan, ...], retrieved: list[tuple[str, str]], k: int
) -> float:
    """Binary-relevance nDCG@k over chunks, using the same containment rule as
    `span_recall_at_k`: ``1/log2(rank + 1)`` at the first top-k chunk that is
    in an answer span's own note and contains that span, else 0.0."""
    for index, (note, text) in enumerate(retrieved[:k]):
        if _is_hit(spans, note, text):
            return 1.0 / math.log2(index + 2)
    return 0.0
