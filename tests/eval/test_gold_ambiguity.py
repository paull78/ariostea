import pytest

from ariostea.adapters.chat.openai_compat import ChatError
from ariostea.eval.gold_ambiguity import (
    UNREACHABLE_PREFIX,
    ambiguity_filter,
    ambiguity_gate,
    collect_competitors,
    competing_passages,
)
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase

CASE = WikiGoldCase(
    query="how is a violin tuned",
    query_lang="en",
    type="paraphrase",
    scenario="paraphrase",
    expected_notes=("strings/violin.md",),
    answer_spans=(AnswerSpan(note="strings/violin.md", text="perfect fifths"),),
)

HIT = ("strings/violin.md", "The violin is tuned in perfect fifths.")
OTHER_NOTE = ("strings/cello.md", "The cello is also tuned in fifths.")
SAME_NOTE_MISS = ("strings/violin.md", "The violin has four strings and a curved bridge.")


class FakeJudge:
    """Records the prompts it is asked, replies with a canned string."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.users: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.users.append(user)
        return self.reply


def _channel(*rows):
    def search_fn(query, k):
        return list(rows)[:k]

    return search_fn


def test_the_chunk_containing_the_span_is_not_a_competitor():
    # It is the hit itself. Counting it would make every case look ambiguous.
    assert competing_passages(CASE, {"DENSE": _channel(HIT)}, limit=5) == ()


def test_a_chunk_in_another_note_competes():
    assert competing_passages(CASE, {"DENSE": _channel(HIT, OTHER_NOTE)}, limit=5) == (
        OTHER_NOTE[1],
    )


def test_a_chunk_in_the_expected_note_that_lacks_the_span_still_competes():
    # The span metric requires the retrieved chunk to *contain the span*, so a
    # sibling chunk of the correct note scores as a miss exactly like a chunk
    # of some unrelated note. Same-note is not the same as same-chunk, and
    # filtering by note alone would miss this whole class of ambiguity.
    assert competing_passages(CASE, {"DENSE": _channel(SAME_NOTE_MISS)}, limit=5) == (
        SAME_NOTE_MISS[1],
    )


def test_competitors_are_deduped_across_channels():
    channels = {"DENSE": _channel(OTHER_NOTE), "SPARSE": _channel(OTHER_NOTE, SAME_NOTE_MISS)}
    assert len(competing_passages(CASE, channels, limit=5)) == 2


def test_competitors_are_capped_even_when_channels_overrun_the_limit():
    # `limit` is passed to each channel as k, but a channel is free to return
    # more than it was asked for, and several channels together certainly can.
    # The cap has to be applied to the merged list, not trusted to the inputs.
    def greedy(query, k):
        return [("a.md", "first"), ("b.md", "second"), ("c.md", "third")]

    assert competing_passages(CASE, {"DENSE": greedy}, limit=2) == ("first", "second")


def test_competing_passages_needs_at_least_one_channel():
    with pytest.raises(ValueError, match="at least one channel"):
        competing_passages(CASE, {}, limit=5)


def test_no_competitors_approves_without_asking_the_judge():
    # Nothing to be ambiguous against, so spending a model call would only
    # invent a chance to be wrong.
    judge = FakeJudge('{"competes": true, "reason": "should never be consulted"}')
    assert ambiguity_gate(judge, CASE, ()) is None
    assert judge.users == []


def test_a_competitor_that_also_answers_is_rejected():
    judge = FakeJudge('{"competes": true, "which": 1, "reason": "cello tuned in fifths too"}')
    reason = ambiguity_gate(judge, CASE, ("The cello is also tuned in fifths.",))
    assert reason is not None
    assert "cello tuned in fifths too" in reason


def test_a_competitor_that_does_not_answer_is_approved():
    judge = FakeJudge('{"competes": false, "reason": "none of them mention tuning"}')
    assert ambiguity_gate(judge, CASE, ("The violin has four strings.",)) is None


def test_an_unreadable_verdict_is_a_rejection():
    # Same rule as the adversarial gate: a verdict nobody can read is not
    # evidence the case is sound, and defaulting to approval would wave
    # through precisely the responses the judge struggled with.
    judge = FakeJudge("I think, on balance, probably not?")
    assert ambiguity_gate(judge, CASE, ("something",)) is not None


def test_a_verdict_missing_the_key_is_a_rejection():
    judge = FakeJudge('{"reason": "forgot the verdict"}')
    assert ambiguity_gate(judge, CASE, ("something",)) is not None


def test_the_judge_is_never_shown_which_passage_is_the_labelled_one():
    # It must decide whether a passage answers the query on the passage's own
    # merits. Telling it which one is gold invites it to rubber-stamp.
    judge = FakeJudge('{"competes": false, "reason": "no"}')
    ambiguity_gate(judge, CASE, ("The cello is also tuned in fifths.",))
    assert "perfect fifths" not in judge.users[0]
    assert "strings/violin.md" not in judge.users[0]


def test_collect_competitors_pairs_each_case_with_its_rivals():
    collected = collect_competitors([CASE], {"DENSE": _channel(HIT, OTHER_NOTE)}, limit=5)
    assert collected == [(CASE, (OTHER_NOTE[1],))]


def test_collect_competitors_makes_no_model_calls():
    # The whole point of the split: retrieval must finish and release the
    # index before the judge model is ever loaded.
    judge = FakeJudge('{"competes": true, "reason": "x"}')
    collect_competitors([CASE], {"DENSE": _channel(HIT, OTHER_NOTE)}, limit=5)
    assert judge.users == []


def test_ambiguity_filter_splits_and_reports_reasons():
    rejecting = FakeJudge('{"competes": true, "reason": "another passage answers"}')
    collected = collect_competitors([CASE], {"DENSE": _channel(HIT, OTHER_NOTE)}, limit=5)
    kept, dropped = ambiguity_filter(collected, rejecting)
    assert kept == []
    assert [case for case, _ in dropped] == [CASE]
    assert "another passage answers" in dropped[0][1]


def test_ambiguity_filter_keeps_a_case_with_no_competitors():
    judge = FakeJudge('{"competes": true, "reason": "never asked"}')
    collected = collect_competitors([CASE], {"DENSE": _channel(HIT)}, limit=5)
    kept, dropped = ambiguity_filter(collected, judge)
    assert kept == [CASE] and dropped == []
    assert judge.users == []


def test_a_judge_outage_is_marked_unreachable_not_ambiguous():
    # An endpoint that cannot load the model says nothing about the case.
    # Recording it as ambiguity would shrink the gold set and call an outage
    # a quality signal -- this pipeline has been bitten by that once already.
    class DeadJudge:
        def complete(self, system, user):
            raise ChatError("model failed to load: insufficient system resources")

    collected = collect_competitors([CASE], {"DENSE": _channel(HIT, OTHER_NOTE)}, limit=5)
    kept, dropped = ambiguity_filter(collected, DeadJudge())
    assert kept == []
    assert dropped[0][1].startswith(UNREACHABLE_PREFIX)


def test_progress_is_reported_after_every_verdict():
    # A stage that prints nothing for half an hour of model calls is
    # indistinguishable from one that has hung; two real runs were stopped
    # mid-judging while they were in fact progressing normally.
    judge = FakeJudge('{"competes": false, "reason": "no"}')
    collected = collect_competitors([CASE, CASE], {"DENSE": _channel(HIT, OTHER_NOTE)}, limit=5)
    seen: list[tuple[int, int, int]] = []
    ambiguity_filter(collected, judge, on_progress=lambda *args: seen.append(args))
    assert seen == [(1, 2, 1), (2, 2, 2)]


def test_progress_is_optional():
    judge = FakeJudge('{"competes": false, "reason": "no"}')
    collected = collect_competitors([CASE], {"DENSE": _channel(HIT, OTHER_NOTE)}, limit=5)
    assert ambiguity_filter(collected, judge)[0] == [CASE]
