import pytest

from ariostea.eval.difficulty import (
    ClusterBaseline,
    cluster_baselines,
    flag_easy_clusters,
    format_baselines,
)
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase


def _case(cluster: str, query: str) -> WikiGoldCase:
    note = f"{cluster}/note.md"
    return WikiGoldCase(
        query=query,
        query_lang="en",
        type="paraphrase",
        scenario="paraphrase",
        expected_notes=(note,),
        answer_spans=(AnswerSpan(note=note, text="perfect fifths"),),
    )


def _always_finds_strings(query, k):
    return [("strings/note.md", "tuned in perfect fifths")]


def test_baselines_are_grouped_by_cluster_in_name_order():
    cases = [_case("strings", "a"), _case("cheese", "b")]
    baselines = cluster_baselines(cases, _always_finds_strings, k=5)
    assert [b.cluster for b in baselines] == ["cheese", "strings"]
    assert [b.n for b in baselines] == [1, 1]


def test_a_cluster_the_dense_channel_always_answers_scores_one():
    (baseline,) = cluster_baselines([_case("strings", "a")], _always_finds_strings, k=5)
    assert baseline.note_recall_at_k == 1.0
    assert baseline.span_recall_at_k == 1.0


def test_a_cluster_the_dense_channel_never_answers_scores_zero():
    (baseline,) = cluster_baselines([_case("cheese", "a")], _always_finds_strings, k=5)
    assert baseline.note_recall_at_k == 0.0
    assert baseline.span_recall_at_k == 0.0


def test_the_right_note_with_the_wrong_chunk_scores_note_but_not_span():
    # The gap worth seeing: a cluster can be easy to find and hard to answer.
    def right_note_wrong_chunk(query, k):
        return [("strings/note.md", "a paragraph about the varnish")]

    (baseline,) = cluster_baselines([_case("strings", "a")], right_note_wrong_chunk, k=5)
    assert baseline.note_recall_at_k == 1.0
    assert baseline.span_recall_at_k == 0.0


def test_scores_average_over_the_cases_in_a_cluster():
    cases = [_case("strings", "a"), _case("cheese", "b"), _case("strings", "c")]
    baselines = {b.cluster: b for b in cluster_baselines(cases, _always_finds_strings, k=5)}
    assert baselines["strings"].n == 2 and baselines["strings"].note_recall_at_k == 1.0
    assert baselines["cheese"].n == 1 and baselines["cheese"].note_recall_at_k == 0.0


def test_cluster_baselines_rejects_a_case_with_no_expected_notes():
    empty = WikiGoldCase(
        query="q",
        query_lang="en",
        type="paraphrase",
        scenario="paraphrase",
        expected_notes=(),
        answer_spans=(),
    )
    with pytest.raises(ValueError, match="no expected_notes"):
        cluster_baselines([empty], lambda q, k: [], k=5)


def test_flag_easy_clusters_uses_note_recall_and_the_threshold():
    easy = ClusterBaseline("cheese", 10, 0.96, 0.90)
    hard = ClusterBaseline("strings", 10, 0.60, 0.40)
    assert flag_easy_clusters([easy, hard], threshold=0.95) == ("cheese",)


def test_flag_easy_clusters_is_inclusive_at_the_threshold():
    borderline = ClusterBaseline("cheese", 10, 0.95, 0.90)
    assert flag_easy_clusters([borderline], threshold=0.95) == ("cheese",)


def test_a_high_span_recall_alone_does_not_flag_a_cluster():
    # The flag reads note recall: "can dense find the right article at all",
    # which is the sense in which the old corpus was too easy.
    assert flag_easy_clusters([ClusterBaseline("cheese", 10, 0.40, 0.99)]) == ()


def test_format_baselines_marks_the_flagged_clusters():
    text = format_baselines([ClusterBaseline("cheese", 10, 0.96, 0.90)], threshold=0.95)
    assert "cheese" in text and "TOO EASY" in text


def test_format_baselines_says_nothing_when_nothing_is_flagged():
    text = format_baselines([ClusterBaseline("strings", 10, 0.60, 0.40)], threshold=0.95)
    assert "TOO EASY" not in text
