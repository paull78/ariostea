"""Span-anchored gold schema for the eval corpus: dataclasses, a JSON loader,
and a validator that checks each case's notes and answer spans against the
corpus before it's trusted for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ariostea.eval.normalize import normalize_ws

# Query types that stress a specific retrieval track (see the eval-corpus design doc).
SPAN_TYPES = ("paraphrase", "exact_term", "buried", "cross_lingual")


@dataclass(frozen=True)
class AnswerSpan:
    note: str
    text: str


@dataclass(frozen=True)
class WikiGoldCase:
    query: str
    query_lang: str
    type: str
    scenario: str
    expected_notes: tuple[str, ...]
    answer_spans: tuple[AnswerSpan, ...]


def load_wiki_gold(path: str | Path) -> list[WikiGoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        WikiGoldCase(
            query=row["query"],
            query_lang=row["query_lang"],
            type=row["type"],
            scenario=row["scenario"],
            expected_notes=tuple(row["expected_notes"]),
            answer_spans=tuple(
                AnswerSpan(note=span["note"], text=span["text"]) for span in row["answer_spans"]
            ),
        )
        for row in rows
    ]


def validate_wiki_gold(cases: list[WikiGoldCase], notes: dict[str, str]) -> list[str]:
    """Return a list of human-readable errors; an empty list means valid.

    `notes` maps a note path to its full text. A case is valid when it names at
    least one expected note, uses a known query type, and every answer span both
    references a corpus note and appears verbatim (whitespace/case-insensitive)
    in that note.
    """
    errors: list[str] = []
    for i, case in enumerate(cases):
        if not case.expected_notes:
            errors.append(f"case {i}: expected_notes is empty")
        if case.type not in SPAN_TYPES:
            errors.append(f"case {i}: unknown type {case.type!r}")
        if not case.answer_spans:
            errors.append(f"case {i}: no answer_spans")
        for note_path in case.expected_notes:
            if note_path not in notes:
                errors.append(f"case {i}: expected note {note_path!r} not in corpus")
        for span in case.answer_spans:
            if span.note not in notes:
                errors.append(f"case {i}: span note {span.note!r} not in corpus")
            elif normalize_ws(span.text) not in normalize_ws(notes[span.note]):
                errors.append(f"case {i}: span text not found in {span.note!r}")
            if span.note not in case.expected_notes:
                errors.append(f"case {i}: span note {span.note!r} not in expected_notes")
    return errors
