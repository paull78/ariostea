"""Choose the passages gold queries are generated from.

This module, not the prompt, is where "a query that stresses BM25" is
decided. A prompt can ask for a rare-term query, but only passage selection
can guarantee a rare term is present to build one from; a prompt can ask for
a buried fact, but only selection knows whether the passage is actually
buried. The prompt phrases the request, this module makes it satisfiable.

Selection is fully deterministic -- notes in sorted path order, passages in
document order -- so the same corpus always offers the same candidates. The
LLM is the only nondeterminism in the pipeline, which is as much as one
generation run can reasonably contain.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace

from ariostea.eval.wiki_notes import cluster_of

_H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)
# Letters only, four or more: skips digits, punctuation and short function
# words without needing a stopword list per language.
_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)

# A passage has to state enough to hold an answer, and stay short enough that
# a model copying a span out of it is not searching a page of text.
MIN_CHARS = 300
MAX_CHARS = 1500
# A note has to be long before a fact inside it counts as "buried"...
BURIED_MIN_NOTE_CHARS = 8000
# ...and the passage has to sit past this fraction of the way into it.
BURIED_MIN_OFFSET = 0.4
# A token appearing in at most this many notes is rare enough that a lexical
# channel should find it and a dense channel may not.
RARE_MAX_NOTES = 2
# ...but in a 79-note corpus that admits ordinary prose too, so an
# `exact_term` passage must also carry a term scoring at least this in tf-idf.
# Calibrated on the real corpus rather than guessed: 12 sits just below the
# median best-term score (14.7) and leaves 72 of 79 notes eligible with 1214
# candidate passages, which is 30x the query budget with the round-robin
# spread across clusters intact. Raising it to 20 would cut note coverage to
# 59 and start starving whole clusters of `exact_term` cases.
RARE_MIN_TFIDF = 12.0


@dataclass(frozen=True)
class Passage:
    note: str
    heading: str
    text: str
    offset: int  # character offset of `text` within the note body
    note_chars: int
    rare_terms: tuple[str, ...] = ()
    # tf-idf of the strongest term in `rare_terms`; 0.0 when there is none.
    rare_score: float = 0.0

    @property
    def cluster(self) -> str:
        return cluster_of(self.note)


def _sections(text: str) -> list[tuple[str, int, str]]:
    """`(heading, offset of the body, body)` for each `##` section.

    Text before the first `##` is returned under an empty heading -- that is
    the article lead, which is prose worth generating from.
    """
    matches = list(_H2.finditer(text))
    if not matches:
        return [("", 0, text)]
    sections = [("", 0, text[: matches[0].start()])]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), match.end(), text[match.end() : end]))
    return sections


def _is_prose(paragraph: str) -> bool:
    """Whether a paragraph is worth generating a query from: not a heading,
    list, quote or table row, and ending like a sentence."""
    stripped = paragraph.strip()
    if not stripped or stripped[0] in "#-*>|":
        return False
    return stripped.endswith((".", ".)", '."', "!", "?"))


def split_passages(note: str, text: str) -> list[Passage]:
    """Candidate passages from one note body, in document order.

    Consecutive prose paragraphs accumulate until they reach `MIN_CHARS` and
    are emitted if they still fit in `MAX_CHARS`. A run that overshoots
    `MAX_CHARS` is discarded rather than truncated: a truncated passage can
    end mid-sentence, and a span copied out of it would be verbatim in the
    passage but read as a fragment in the note. Discarding candidates is free
    -- this is selection, not conversion, and the corpus offers far more
    passages than ~150 queries need.

    `offset` is where the passage's first paragraph starts within `text`,
    which is what `_eligible` reads to decide whether a fact is buried.
    """
    passages: list[Passage] = []
    for heading, base, body in _sections(text):
        cursor = base
        buffer: list[str] = []
        start = base
        for paragraph in body.split("\n\n"):
            if _is_prose(paragraph):
                if not buffer:
                    start = cursor
                buffer.append(paragraph.strip())
                joined = "\n\n".join(buffer)
                if len(joined) >= MIN_CHARS:
                    if len(joined) <= MAX_CHARS:
                        passages.append(
                            Passage(
                                note=note,
                                heading=heading,
                                text=joined,
                                offset=start,
                                note_chars=len(text),
                            )
                        )
                    buffer = []
            else:
                buffer = []
            cursor += len(paragraph) + 2  # the "\n\n" that split() consumed
    return passages


def document_frequency(notes: dict[str, str]) -> Counter[str]:
    """How many notes each lowercased word appears in -- note count, not
    occurrence count, which is what makes a term *rare in the corpus* rather
    than merely infrequent inside one article."""
    frequencies: Counter[str] = Counter()
    for text in notes.values():
        frequencies.update({word.lower() for word in _WORD.findall(text)})
    return frequencies


def term_counts(text: str) -> Counter[str]:
    """How often each lowercased word occurs in one piece of text."""
    return Counter(word.lower() for word in _WORD.findall(text))


@dataclass(frozen=True)
class CorpusStats:
    """Corpus-wide term statistics, carried together because a document
    frequency means nothing without the note count it is out of."""

    document_frequency: Counter[str]
    note_count: int

    def idf(self, term: str) -> float:
        """Inverse document frequency, `log(note_count / df)`. Zero for a term
        in every note, and undefined -- returned as 0.0 -- for one in none."""
        df = self.document_frequency[term]
        return math.log(self.note_count / df) if df else 0.0


def corpus_stats(notes: dict[str, str]) -> CorpusStats:
    return CorpusStats(document_frequency=document_frequency(notes), note_count=len(notes))


def rare_terms(
    text: str,
    stats: CorpusStats,
    note_counts: Counter[str],
    max_notes: int = RARE_MAX_NOTES,
) -> tuple[str, ...]:
    """Tokens of `text` in at most `max_notes` corpus notes, best first.

    "Best" is plain tf-idf: the term's frequency *in its own note* times its
    inverse document frequency across the corpus. This is the one place in the
    project where tf-idf is the right tool -- retrieval uses BM25, whose term
    saturation is what you want for *scoring documents*, while here the job is
    only to *rank a handful of candidate terms* and the unsaturated product is
    both adequate and easier to reason about.

    The ranking matters more than the filter. In a 79-note corpus almost every
    passage contains some token appearing in one or two notes, so the `df`
    threshold alone admits ordinary prose -- measured on the real corpus,
    "conveys", "speculate" and "capita" all qualify as rare. What separates a
    technical term from incidental prose is that the technical term *recurs in
    its own article*: "annatto" appears five times in the cheddar note,
    "conveys" once in the espresso note. tf-idf reads exactly that difference,
    so the terms handed to the prompt are the ones a lexical channel can
    actually win on.

    Ties break alphabetically, so selection stays deterministic.
    """
    candidates = {
        word.lower()
        for word in _WORD.findall(text)
        if 0 < stats.document_frequency[word.lower()] <= max_notes
    }
    return tuple(sorted(candidates, key=lambda term: (-note_counts[term] * stats.idf(term), term)))


def best_term_score(
    text: str, stats: CorpusStats, note_counts: Counter[str], max_notes: int = RARE_MAX_NOTES
) -> float:
    """The tf-idf score of the strongest rare term in `text`, or 0.0 if there
    is none. This is what `_eligible` thresholds on for the `exact_term`
    track."""
    terms = rare_terms(text, stats, note_counts, max_notes)
    return note_counts[terms[0]] * stats.idf(terms[0]) if terms else 0.0


def _cluster_interleaved(notes: list[str]) -> list[str]:
    """Note visit order that cycles across clusters: the first note of every
    cluster, then the second of every cluster, and so on.

    Plain sorted order would be wrong here. Selection takes one passage per
    note per round, and a 40-query budget against 79 notes is satisfied
    entirely in the first round -- so with paths sorted alphabetically the
    first forty notes take everything and the last clusters get nothing.
    Measured on the real corpus that skewed `board-games` to 28 cases and
    `string-instruments` to 16, which is backwards: string-instruments is the
    largest cluster and the one the cross-lingual track depends on. Cycling by
    cluster makes the budget land evenly no matter how the clusters are named.
    """
    by_cluster: dict[str, list[str]] = {}
    for note in notes:
        by_cluster.setdefault(cluster_of(note), []).append(note)
    if not by_cluster:
        return []
    order: list[str] = []
    for depth in range(max(len(members) for members in by_cluster.values())):
        for cluster in sorted(by_cluster):
            members = by_cluster[cluster]
            if depth < len(members):
                order.append(members[depth])
    return order


def _eligible(passage: Passage, query_type: str) -> bool:
    """Whether `passage` can support a query of `query_type`.

    Each branch encodes what the corresponding retrieval track needs:

    - `exact_term` needs a term rare *and* distinctive enough that a lexical
      channel has a real advantage -- see `rare_terms` for why the document
      frequency threshold alone is not enough in a corpus this size.
    - `buried` needs a long note *and* a passage late in it. Both matter: a
      late passage in a short note is not buried, and an early passage in a
      long one is what the lead paragraph already said.
    - `cross_lingual` needs an English note, because the design is a query in
      Italian or Spanish whose answer span is English. A passage from the
      it/es notes would make the query same-language and test nothing; those
      notes stay in the corpus as the distractors that make the track hard.
    - `paraphrase` needs nothing beyond being prose.
    """
    if query_type == "paraphrase":
        return True
    if query_type == "exact_term":
        return passage.rare_score >= RARE_MIN_TFIDF
    if query_type == "buried":
        return (
            passage.note_chars >= BURIED_MIN_NOTE_CHARS
            and passage.offset >= BURIED_MIN_OFFSET * passage.note_chars
        )
    if query_type == "cross_lingual":
        return not passage.note.endswith(("-it.md", "-es.md"))
    raise ValueError(f"unknown query type {query_type!r}")


def select_passages(notes: dict[str, str], per_type: dict[str, int]) -> list[tuple[str, Passage]]:
    """`(query_type, passage)` pairs, at most `per_type[type]` of each.

    Passages are taken round-robin across notes -- one from every eligible
    note, then a second from every eligible note, and so on -- so a single
    long article cannot supply a whole query type. A passage is used at most
    once across *all* types: two queries about the same sentence are not
    independent samples, and having a `paraphrase` and an `exact_term` case
    resolve to the same chunk would make those two tracks correlate for
    reasons that have nothing to do with retrieval.

    Returns fewer pairs than asked when the corpus runs out of eligible
    passages. The caller must compare the counts and report the shortfall:
    silently returning 30 `buried` cases against a budget of 40 would read as
    a corpus that supports the requested design when it does not.
    """
    probe = Passage(note="probe/probe.md", heading="", text="", offset=0, note_chars=0)
    for query_type in per_type:
        _eligible(probe, query_type)  # raises on a typo before any work happens

    stats = corpus_stats(notes)
    by_note: dict[str, list[Passage]] = {}
    for note in _cluster_interleaved(sorted(notes)):
        text = notes[note]
        counts = term_counts(text)
        by_note[note] = [
            replace(
                passage,
                rare_terms=rare_terms(passage.text, stats, counts),
                rare_score=best_term_score(passage.text, stats, counts),
            )
            for passage in split_passages(note, text)
        ]

    chosen: list[tuple[str, Passage]] = []
    used: set[tuple[str, int]] = set()
    for query_type in sorted(per_type):
        budget = per_type[query_type]
        picked = 0
        depth = 0
        while picked < budget:
            any_note_reached_this_depth = False
            for note, passages in by_note.items():
                pool = [p for p in passages if _eligible(p, query_type)]
                if depth >= len(pool):
                    continue
                any_note_reached_this_depth = True
                passage = pool[depth]
                if (note, passage.offset) in used:
                    continue
                used.add((note, passage.offset))
                chosen.append((query_type, passage))
                picked += 1
                if picked == budget:
                    break
            if not any_note_reached_this_depth:
                break  # every note is exhausted for this type
            depth += 1
    return chosen
