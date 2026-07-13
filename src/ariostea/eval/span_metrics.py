from __future__ import annotations

import re

from ariostea.eval.wiki_gold import AnswerSpan

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Lowercase and collapse all runs of whitespace to single spaces."""
    return _WS.sub(" ", text).strip().lower()


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
