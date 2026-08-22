import math

from ariostea.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_hit_within_k():
    assert recall_at_k({"a.md"}, ["x.md", "a.md", "y.md"], k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    # a.md is at index 2 (rank 3); with k=2 it is outside the window.
    assert recall_at_k({"a.md"}, ["x.md", "y.md", "a.md"], k=2) == 0.0


def test_recall_at_k_any_expected_counts():
    assert recall_at_k({"a.md", "b.md"}, ["b.md", "z.md"], k=1) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank({"a.md"}, ["a.md", "b.md"]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank({"a.md"}, ["b.md", "a.md"]) == 0.5


def test_reciprocal_rank_absent_is_zero():
    assert reciprocal_rank({"a.md"}, ["b.md", "c.md"]) == 0.0


def test_ndcg_is_one_at_rank_1():
    assert ndcg_at_k({"a.md"}, ["a.md", "b.md"], k=5) == 1.0


def test_ndcg_discounts_logarithmically():
    assert ndcg_at_k({"b.md"}, ["a.md", "b.md"], k=5) == 1.0 / math.log2(3)


def test_ndcg_is_zero_when_the_hit_falls_outside_k():
    assert ndcg_at_k({"b.md"}, ["a.md", "b.md"], k=1) == 0.0


def test_ndcg_is_zero_with_no_hit():
    assert ndcg_at_k({"z.md"}, ["a.md", "b.md"], k=5) == 0.0


def test_ndcg_is_zero_on_an_empty_ranking():
    assert ndcg_at_k({"a.md"}, [], k=5) == 0.0


def test_ndcg_discounts_more_gently_than_mrr():
    # The reason it is worth reporting alongside MRR: between rank 3 and rank
    # 5 MRR loses 40% of its value and nDCG only 22%, so mid-list movement
    # stays visible.
    ranked = ["x.md", "y.md", "a.md", "z.md", "a.md"]
    assert ndcg_at_k({"a.md"}, ranked, k=5) > reciprocal_rank({"a.md"}, ranked)
