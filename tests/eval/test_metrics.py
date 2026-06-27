from ariostea.eval.metrics import recall_at_k, reciprocal_rank


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
