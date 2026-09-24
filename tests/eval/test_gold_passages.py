from collections import Counter

import pytest

from ariostea.eval.gold_passages import (
    BURIED_MIN_NOTE_CHARS,
    BURIED_MIN_OFFSET,
    MAX_CHARS,
    RARE_MIN_TFIDF,
    CorpusStats,
    Passage,
    best_term_score,
    corpus_stats,
    document_frequency,
    rare_terms,
    select_passages,
    split_passages,
    term_counts,
)

BODY = (
    "# Violin\n\n"
    + "The violin is a wooden chordophone in the violin family. " * 6
    + "\n\n## Construction\n\n"
    + "Most violins have a hollow wooden body with a spruce top. " * 6
    + "\n\n- a list item that is not prose\n"
)


def test_split_passages_attributes_a_heading_to_each_passage():
    passages = split_passages("a/violin.md", BODY)
    assert {p.heading for p in passages} == {"", "Construction"}


def test_split_passages_records_the_offset_within_the_note():
    passages = split_passages("a/violin.md", BODY)
    construction = next(p for p in passages if p.heading == "Construction")
    assert BODY[construction.offset :].startswith(construction.text[:40])
    assert construction.offset > 0


def test_split_passages_emits_text_that_is_verbatim_in_the_note():
    # The whole pipeline rests on this: a span copied out of a passage must be
    # findable in the note, so a passage must be a substring of the note.
    for passage in split_passages("a/violin.md", BODY):
        assert passage.text in BODY


def test_split_passages_skips_list_items():
    joined = " ".join(p.text for p in split_passages("a/violin.md", BODY))
    assert "a list item that is not prose" not in joined


def test_split_passages_skips_the_h1_heading_line():
    joined = " ".join(p.text for p in split_passages("a/violin.md", BODY))
    assert "# Violin" not in joined


def test_split_passages_never_exceeds_the_maximum_length():
    for passage in split_passages("a/violin.md", BODY):
        assert len(passage.text) <= MAX_CHARS


def test_split_passages_drops_a_paragraph_shorter_than_the_minimum():
    assert split_passages("a/x.md", "# X\n\nToo short.\n") == []


def test_split_passages_drops_a_paragraph_longer_than_the_maximum():
    # Truncating instead would end the passage mid-sentence, and a span copied
    # out of it would read as a fragment back in the note.
    huge = "This sentence is repeated many times to overflow the limit. " * 40
    assert split_passages("a/x.md", f"# X\n\n{huge}\n") == []


def test_split_passages_records_the_note_length():
    for passage in split_passages("a/violin.md", BODY):
        assert passage.note_chars == len(BODY)


def test_document_frequency_counts_notes_not_occurrences():
    df = document_frequency({"a.md": "violin violin violin", "b.md": "cello"})
    assert df["violin"] == 1
    assert df["cello"] == 1


def test_document_frequency_ignores_short_tokens_and_digits():
    df = document_frequency({"a.md": "the 1737 sul ponticello"})
    assert "the" not in df and "1737" not in df
    assert df["ponticello"] == 1


def test_corpus_stats_carries_the_note_count():
    stats = corpus_stats({"a.md": "violin", "b.md": "cello"})
    assert stats.note_count == 2
    assert stats.document_frequency["violin"] == 1


def test_idf_is_zero_for_a_term_in_every_note():
    stats = CorpusStats(document_frequency=Counter({"violin": 4}), note_count=4)
    assert stats.idf("violin") == 0.0


def test_idf_is_zero_for_an_unknown_term():
    stats = CorpusStats(document_frequency=Counter({"violin": 1}), note_count=4)
    assert stats.idf("unseen") == 0.0


def test_idf_grows_as_a_term_gets_rarer():
    stats = CorpusStats(document_frequency=Counter({"rare": 1, "common": 8}), note_count=16)
    assert stats.idf("rare") > stats.idf("common") > 0


def test_term_counts_counts_occurrences_case_insensitively():
    assert term_counts("Violin violin cello")["violin"] == 2


def test_rare_terms_keeps_only_tokens_in_few_notes():
    stats = CorpusStats(document_frequency=Counter({"violin": 40, "ponticello": 1}), note_count=79)
    counts = term_counts("Bow sul ponticello on the violin")
    assert rare_terms("Bow sul ponticello on the violin", stats, counts) == ("ponticello",)


def test_rare_terms_ranks_a_repeated_technical_term_above_incidental_prose():
    # The point of the ranking. Both qualify on document frequency in a small
    # corpus, but only the one that recurs in its own article is a term a
    # lexical channel can win on -- measured on the real corpus, "conveys" is
    # as "rare" as "annatto" and worth nothing as a query anchor.
    stats = CorpusStats(document_frequency=Counter({"annatto": 1, "conveys": 1}), note_count=79)
    note_counts = Counter({"annatto": 5, "conveys": 1})
    assert rare_terms("annatto conveys", stats, note_counts) == ("annatto", "conveys")


def test_rare_terms_prefers_the_rarer_term_when_note_frequency_ties():
    stats = CorpusStats(document_frequency=Counter({"alpha": 1, "bravo": 2}), note_count=79)
    note_counts = Counter({"alpha": 3, "bravo": 3})
    assert rare_terms("bravo alpha", stats, note_counts) == ("alpha", "bravo")


def test_rare_terms_breaks_ties_alphabetically_so_selection_is_deterministic():
    stats = CorpusStats(document_frequency=Counter({"alpha": 1, "bravo": 1}), note_count=79)
    note_counts = Counter({"alpha": 2, "bravo": 2})
    assert rare_terms("bravo alpha", stats, note_counts) == ("alpha", "bravo")


def test_rare_terms_deduplicates():
    stats = CorpusStats(document_frequency=Counter({"tasto": 2}), note_count=79)
    assert rare_terms("tasto tasto tasto", stats, term_counts("tasto tasto tasto")) == ("tasto",)


def test_rare_terms_excludes_a_token_absent_from_the_frequencies():
    # Count 0 means the token came from outside the corpus the frequencies
    # were built over, so nothing is known about how rare it is there.
    stats = CorpusStats(document_frequency=Counter({"violin": 1}), note_count=79)
    assert rare_terms("unseen", stats, term_counts("unseen")) == ()


def test_best_term_score_reports_the_top_terms_tf_idf():
    stats = CorpusStats(document_frequency=Counter({"annatto": 1}), note_count=79)
    expected = 5 * stats.idf("annatto")
    assert best_term_score("annatto", stats, Counter({"annatto": 5})) == expected


def test_best_term_score_is_zero_without_a_rare_term():
    stats = CorpusStats(document_frequency=Counter({"violin": 79}), note_count=79)
    assert best_term_score("violin", stats, Counter({"violin": 9})) == 0.0


def test_passage_cluster_comes_from_the_note_path():
    passage = Passage(note="cheese/brie.md", heading="", text="x", offset=0, note_chars=1)
    assert passage.cluster == "cheese"


LONG_PROSE = "The instrument has a hollow wooden body with a carved spruce top. " * 8


def _corpus() -> dict[str, str]:
    """A fixture that can actually reach RARE_MIN_TFIDF.

    The threshold is calibrated for the 79-note corpus, where idf tops out at
    log(79) = 4.4. A three-note fixture caps idf at log(3) = 1.1, so the rare
    term has to recur often enough to clear 12 on term frequency alone --
    hence "ponticello" twelve times, in one note only.
    """
    filler = "Filler prose that keeps the note long enough to bury a fact. " * 200
    ponticello = "The player may bow sul ponticello. " * 12
    return {
        "strings/violin.md": (
            f"# Violin\n\n{LONG_PROSE}\n\n## History\n\n{filler}\n\n{LONG_PROSE}\n"
        ),
        "strings/cello.md": f"# Cello\n\n{ponticello}{LONG_PROSE}\n",
        "strings/violino-it.md": f"# Violino\n\n{LONG_PROSE}\n",
    }


def test_select_passages_respects_the_per_type_budget():
    chosen = select_passages(_corpus(), {"paraphrase": 2})
    assert len(chosen) == 2
    assert {t for t, _ in chosen} == {"paraphrase"}


def test_select_passages_spreads_across_notes_before_repeating_one():
    chosen = select_passages(_corpus(), {"paraphrase": 2})
    assert len({p.note for _, p in chosen}) == 2


def test_select_passages_never_reuses_a_passage_across_types():
    # Two query types over the same fact are not independent samples.
    chosen = select_passages(_corpus(), {"paraphrase": 3, "buried": 1})
    keys = [(p.note, p.offset) for _, p in chosen]
    assert len(keys) == len(set(keys))


def test_buried_selection_requires_a_long_note_and_a_late_passage():
    chosen = select_passages(_corpus(), {"buried": 5})
    assert chosen
    for _, passage in chosen:
        assert passage.note_chars >= BURIED_MIN_NOTE_CHARS
        assert passage.offset >= BURIED_MIN_OFFSET * passage.note_chars


def test_exact_term_selection_only_picks_passages_with_a_strong_rare_term():
    chosen = select_passages(_corpus(), {"exact_term": 5})
    assert chosen, "fixture must offer a rare-term passage or this proves nothing"
    stats = corpus_stats(_corpus())
    for _, passage in chosen:
        assert passage.rare_terms
        counts = term_counts(_corpus()[passage.note])
        assert best_term_score(passage.text, stats, counts) >= RARE_MIN_TFIDF


def test_selected_passages_carry_their_ranked_rare_terms():
    chosen = select_passages(_corpus(), {"exact_term": 1})
    assert "ponticello" in chosen[0][1].rare_terms


def test_cross_lingual_selection_only_picks_english_notes():
    chosen = select_passages(_corpus(), {"cross_lingual": 5})
    assert chosen
    assert all(not p.note.endswith(("-it.md", "-es.md")) for _, p in chosen)


def test_select_passages_returns_fewer_than_asked_when_the_corpus_runs_out():
    # Silent truncation would read as "the corpus supports 150 queries" when it
    # does not; the runner compares the returned count against the budget.
    chosen = select_passages({"a/x.md": f"# X\n\n{LONG_PROSE}\n"}, {"paraphrase": 99})
    assert 0 < len(chosen) < 99


def test_select_passages_is_deterministic():
    assert select_passages(_corpus(), {"paraphrase": 3}) == select_passages(
        _corpus(), {"paraphrase": 3}
    )


def test_select_passages_rejects_an_unknown_query_type():
    with pytest.raises(ValueError, match="unknown query type"):
        select_passages(_corpus(), {"nonsense": 1})


def test_select_passages_rejects_an_unknown_type_before_doing_any_work():
    # The probe runs first so a typo fails instantly rather than after
    # splitting every note in an 80-article corpus.
    with pytest.raises(ValueError, match="unknown query type"):
        select_passages(_corpus(), {"paraphrase": 1, "nonsense": 1})


def test_selection_spreads_across_clusters_not_just_notes():
    # Round-robin over notes in sorted path order would let a budget smaller
    # than the corpus fall entirely to the alphabetically-first clusters,
    # leaving later ones with too few cases for their difficulty baseline to
    # mean anything.
    corpus = {
        f"{cluster}/note{i}.md": f"# N\n\n{LONG_PROSE}\n"
        for cluster in ("aaa", "zzz")
        for i in range(5)
    }
    chosen = select_passages(corpus, {"paraphrase": 4})
    clusters = Counter(p.cluster for _, p in chosen)
    assert clusters["aaa"] == clusters["zzz"] == 2
