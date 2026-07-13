from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
