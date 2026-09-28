"""Pieces of the chunking sweep that are worth testing on their own.

`eval/run_chunk_sweep.py` indexes the corpus once per chunking spec and logs
each result; everything here is the bookkeeping around that -- reading specs,
naming runs, counting how many gold spans a policy can reach at all, and
loading the cases the discrimination gate dropped so a policy that loses them
is caught.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ariostea.config.schema import ChunkingCfg
from ariostea.domain.models import Note
from ariostea.eval.span_metrics import chunk_contains_span
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase
from ariostea.ports.pipeline import Chunker

_SPEC = re.compile(r"^(\d+)([wt])(?:\+(\d+))?$")
_UNITS = {"w": "words", "t": "model_tokens"}


def parse_spec(spec: str) -> ChunkingCfg:
    """`512w`, `128t`, `128t+32`: a size, a unit (w words, t model tokens), and
    an optional overlap in the same unit."""
    match = _SPEC.match(spec)
    if match is None:
        raise ValueError(f"bad chunking spec {spec!r}; expected e.g. 512w, 128t or 128t+32")
    size, unit, overlap = match.groups()
    return ChunkingCfg(max_tokens=int(size), overlap=int(overlap or 0), unit=_UNITS[unit])


def describe(cfg: ChunkingCfg) -> str:
    unit = "tokens" if cfg.unit == "model_tokens" else "words"
    base = f"{cfg.max_tokens} {unit}"
    return f"{base}, overlap {cfg.overlap}" if cfg.overlap else base


def sweep_run_id(date: str, spec: str, channels: list[str]) -> str:
    """Stable per spec and channel set, so a cheap pass and a full pass of the
    same spec are separate log entries rather than a refused duplicate."""
    return f"{date}-chunk-{spec}-{'-'.join(sorted(c.lower() for c in channels))}"


def reachable_spans(cases: list[WikiGoldCase], notes: dict[str, str], chunker: Chunker) -> int:
    """Cases whose first span sits whole inside some chunk of its note.

    The ceiling on span recall for every channel under this chunking: a span
    split across a boundary scores zero no matter how good retrieval is.
    """
    chunks_by_note: dict[str, list[str]] = {}
    for path, body in notes.items():
        note = Note(
            path=path, title="", frontmatter={}, tags=(), wikilinks=(), content_hash="", mtime=0.0
        )
        chunks_by_note[path] = [c.text for c in chunker.chunk(note, body)]
    return sum(
        any(
            chunk_contains_span(text, case.answer_spans[0].text)
            for text in chunks_by_note.get(case.answer_spans[0].note, [])
        )
        for case in cases
    )


def load_discriminated(rejected_path: Path) -> list[WikiGoldCase]:
    """The cases the discrimination gate dropped because every channel already
    answered them at rank 1 on the original chunking.

    The committed gold set is biased toward what that chunking finds hard, which
    flatters any change. Scoring these alongside it shows whether a new policy
    buys its gains by losing the easy cases.
    """
    rows = json.loads(rejected_path.read_text(encoding="utf-8"))
    return [
        WikiGoldCase(
            query=row["query"],
            query_lang="und",  # not recorded for rejected candidates
            type=row["type"],
            scenario=row["type"],
            expected_notes=(row["note"],),
            answer_spans=(AnswerSpan(note=row["note"], text=row["span"]),),
        )
        for row in rows
        if row["stage"] == "discrimination"
    ]
