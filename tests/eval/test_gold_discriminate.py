import pytest

from ariostea.eval.gold_discriminate import discrimination_filter
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase

CASE = WikiGoldCase(
    query="how is a violin tuned",
    query_lang="en",
    type="paraphrase",
    scenario="paraphrase",
    expected_notes=("strings/violin.md",),
    answer_spans=(AnswerSpan(note="strings/violin.md", text="perfect fifths"),),
)


def _hit(query, k):
    return [("strings/violin.md", "It is tuned in perfect fifths.")]


def _miss(query, k):
    return [("strings/cello.md", "Something else entirely.")]


def test_a_case_every_channel_answers_at_rank_1_is_dropped():
    kept, dropped = discrimination_filter([CASE], {"DENSE": _hit, "SPARSE": _hit})
    assert kept == [] and dropped == [CASE]


def test_a_case_one_channel_misses_is_kept():
    kept, dropped = discrimination_filter([CASE], {"DENSE": _hit, "SPARSE": _miss})
    assert kept == [CASE] and dropped == []


def test_a_case_every_channel_misses_is_kept():
    # Hard is not the same as useless: a case nothing answers is exactly the
    # kind an improvement should later be able to move.
    kept, _ = discrimination_filter([CASE], {"DENSE": _miss, "SPARSE": _miss})
    assert kept == [CASE]


def test_a_right_note_with_the_wrong_chunk_does_not_count_as_answered():
    def right_note_wrong_chunk(query, k):
        return [("strings/violin.md", "A paragraph about the varnish.")]

    kept, _ = discrimination_filter([CASE], {"DENSE": right_note_wrong_chunk, "SPARSE": _hit})
    assert kept == [CASE]


def test_rank_2_does_not_count_as_answered():
    def rank_two(query, k):
        return [("strings/cello.md", "Wrong."), ("strings/violin.md", "perfect fifths")]

    kept, _ = discrimination_filter([CASE], {"DENSE": rank_two, "SPARSE": _hit})
    assert kept == [CASE]


def test_an_empty_result_list_does_not_count_as_answered():
    kept, _ = discrimination_filter([CASE], {"DENSE": lambda q, k: [], "SPARSE": _hit})
    assert kept == [CASE]


def test_an_empty_channel_map_raises():
    # `all()` over no channels is vacuously true, which would silently drop
    # every case as "too easy" and write an empty gold file while reporting
    # success.
    with pytest.raises(ValueError, match="at least one channel"):
        discrimination_filter([CASE], {})


def test_each_channel_is_asked_for_exactly_one_result_per_case():
    seen: list[int] = []

    def recording(query, k):
        seen.append(k)
        return _hit(query, k)

    discrimination_filter([CASE], {"DENSE": recording})
    assert seen == [1]


def test_case_order_is_preserved_in_both_partitions():
    easy = CASE
    hard = WikiGoldCase(
        query="something nothing finds",
        query_lang="en",
        type="buried",
        scenario="buried",
        expected_notes=("strings/cello.md",),
        answer_spans=(AnswerSpan(note="strings/cello.md", text="a span nobody retrieves"),),
    )
    kept, dropped = discrimination_filter([easy, hard, easy], {"DENSE": _hit})
    assert kept == [hard] and dropped == [easy, easy]
