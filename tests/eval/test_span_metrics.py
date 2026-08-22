import math

from ariostea.eval.span_metrics import (
    chunk_contains_span,
    normalize_ws,
    span_ndcg_at_k,
    span_recall_at_k,
    span_reciprocal_rank,
)
from ariostea.eval.wiki_gold import AnswerSpan


def test_normalize_collapses_whitespace_and_case():
    assert normalize_ws("The  Violin\nis\tTuned") == "the violin is tuned"


def test_chunk_contains_span_ignores_whitespace_and_case():
    assert chunk_contains_span(
        "The violin is  tuned in\nperfect fifths.", "Tuned In Perfect Fifths"
    )
    assert not chunk_contains_span("A cello has four strings.", "tuned in perfect fifths")


def test_span_recall_requires_matching_note_and_text():
    spans = (AnswerSpan(note="violin.md", text="perfect fifths"),)
    retrieved = [("cello.md", "perfect fifths"), ("violin.md", "tuned in perfect fifths")]
    # right text but wrong note at rank 1; correct note at rank 2 → within k=2, not k=1
    assert span_recall_at_k(spans, retrieved, k=1) == 0.0
    assert span_recall_at_k(spans, retrieved, k=2) == 1.0


def test_span_reciprocal_rank_uses_first_true_hit():
    spans = (AnswerSpan(note="violin.md", text="perfect fifths"),)
    retrieved = [("cello.md", "perfect fifths"), ("violin.md", "tuned in perfect fifths")]
    assert span_reciprocal_rank(spans, retrieved) == 0.5


NDCG_SPANS = (AnswerSpan(note="a.md", text="perfect fifths"),)


def test_span_ndcg_is_one_when_the_first_chunk_contains_the_span():
    assert span_ndcg_at_k(NDCG_SPANS, [("a.md", "tuned in perfect fifths")], k=5) == 1.0


def test_span_ndcg_discounts_a_later_chunk():
    retrieved = [("a.md", "unrelated"), ("a.md", "tuned in perfect fifths")]
    assert span_ndcg_at_k(NDCG_SPANS, retrieved, k=5) == 1.0 / math.log2(3)


def test_span_ndcg_ignores_a_containing_chunk_from_the_wrong_note():
    assert span_ndcg_at_k(NDCG_SPANS, [("b.md", "tuned in perfect fifths")], k=5) == 0.0


def test_span_ndcg_respects_the_k_cutoff():
    retrieved = [("a.md", "unrelated"), ("a.md", "tuned in perfect fifths")]
    assert span_ndcg_at_k(NDCG_SPANS, retrieved, k=1) == 0.0


def test_span_ndcg_is_zero_with_no_hit():
    assert span_ndcg_at_k(NDCG_SPANS, [("a.md", "nothing here")], k=5) == 0.0
