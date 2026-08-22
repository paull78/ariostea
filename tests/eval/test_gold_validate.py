from dataclasses import replace

from ariostea.eval.gold_generate import Candidate
from ariostea.eval.gold_validate import automatic_gate

NOTES = {
    "strings/violin.md": (
        "# Violin\n\nThe violin is tuned in perfect fifths: G, D, A, E. "
        "Players sometimes bow sul ponticello, near the bridge.\n"
    )
}
TITLES = {"strings/violin.md": "Violin"}

GOOD = Candidate(
    query="what note does the lowest string sound",
    query_lang="en",
    type="paraphrase",
    note="strings/violin.md",
    passage="The violin is tuned in perfect fifths: G, D, A, E.",
    span="tuned in perfect fifths: G, D, A, E",
)


def test_a_well_formed_candidate_passes():
    assert automatic_gate(GOOD, NOTES, TITLES) is None


def test_a_span_absent_from_the_passage_is_rejected():
    bad = replace(GOOD, span="tuned in perfect fourths")
    assert "verbatim in the passage" in automatic_gate(bad, NOTES, TITLES)


def test_a_span_present_in_the_passage_but_not_the_note_is_rejected():
    # Catches a passage that was hand-built or has drifted from the corpus.
    bad = replace(GOOD, passage="Invented text about the bridge.", span="Invented text")
    assert "verbatim in the cited note" in automatic_gate(bad, NOTES, TITLES)


def test_matching_is_whitespace_and_case_insensitive():
    ok = replace(GOOD, span="Tuned  in\nperfect fifths: g, d, a, e")
    assert automatic_gate(ok, NOTES, TITLES) is None


def test_a_span_shorter_than_the_minimum_is_rejected():
    bad = replace(GOOD, span="G, D")
    assert "shorter" in automatic_gate(bad, NOTES, TITLES)


def test_a_span_longer_than_the_maximum_is_rejected():
    notes = {"a/x.md": "# X\n\n" + "word " * 200}
    bad = Candidate(
        query="a query about the many repeated words",
        query_lang="en",
        type="paraphrase",
        note="a/x.md",
        passage="word " * 200,
        span="word " * 100,
    )
    assert "longer" in automatic_gate(bad, notes, {"a/x.md": "X"})


def test_a_query_that_restates_the_title_is_rejected():
    bad = replace(GOOD, query="violin")
    assert "restates the article title" in automatic_gate(bad, NOTES, TITLES)


def test_a_query_merely_mentioning_the_title_is_allowed():
    # The gate rejects restatement, not any use of the subject word -- a real
    # query about a violin usually says "violin".
    ok = replace(GOOD, query="which four notes does a violin sound when played open")
    assert automatic_gate(ok, NOTES, TITLES) is None


def test_a_span_adding_nothing_beyond_the_title_is_rejected():
    notes = {"a/x.md": "# Violin family\n\nThe violin family violin family violin.\n"}
    bad = Candidate(
        query="which group of instruments is discussed",
        query_lang="en",
        type="paraphrase",
        note="a/x.md",
        passage="The violin family violin family violin.",
        span="violin family violin",
    )
    assert "beyond the article title" in automatic_gate(bad, notes, {"a/x.md": "Violin family"})


def test_a_query_with_no_content_words_is_rejected():
    bad = replace(GOOD, query="?? 42 ??")
    assert "no content words" in automatic_gate(bad, NOTES, TITLES)


def test_a_note_outside_the_corpus_is_rejected():
    bad = replace(GOOD, note="strings/ghost.md")
    assert "not in corpus" in automatic_gate(bad, NOTES, TITLES)


def test_a_cross_lingual_query_written_in_english_is_rejected():
    bad = replace(GOOD, type="cross_lingual", query_lang="it", query="tuned in perfect fifths")
    assert "not in another language" in automatic_gate(bad, NOTES, TITLES)


def test_a_genuine_italian_cross_lingual_query_passes():
    ok = replace(GOOD, type="cross_lingual", query_lang="it", query="come si accorda uno strumento")
    assert automatic_gate(ok, NOTES, TITLES) is None


def test_the_language_check_only_applies_to_cross_lingual_cases():
    # An English query whose words all appear in the passage is fine for a
    # paraphrase case; it is only evidence of failure for a cross-lingual one.
    ok = replace(GOOD, query="tuned in perfect fifths")
    assert automatic_gate(ok, NOTES, TITLES) is None
