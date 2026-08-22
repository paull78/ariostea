from collections import Counter

from ariostea.eval.gold_passages import (
    MAX_CHARS,
    CorpusStats,
    Passage,
    best_term_score,
    corpus_stats,
    document_frequency,
    rare_terms,
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
