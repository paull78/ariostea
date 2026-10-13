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
import statistics
import time
from collections.abc import Callable, Sequence
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


def model_slug(model: str) -> str:
    """`BAAI/bge-m3` -> `bge-m3`: the short name a run id carries."""
    return model.rsplit("/", 1)[-1].lower()


def sweep_run_id(
    date: str,
    spec: str,
    channels: list[str],
    model: str | None = None,
    normalize: bool = False,
) -> str:
    """Stable per spec and channel set, so a cheap pass and a full pass of the
    same spec are separate log entries rather than a refused duplicate.

    `model` is set only for a model other than the pinned one, so control runs
    keep the id shape every earlier run has. `normalize` adds `norm` after the
    spec and model, for runs on unit-length vectors."""
    parts = [f"{date}-chunk-{spec}"]
    if model is not None:
        parts.append(model_slug(model))
    if normalize:
        parts.append("norm")
    parts.append("-".join(sorted(c.lower() for c in channels)))
    return "-".join(parts)


def run_config(k: int, cfg: ChunkingCfg, model: str, normalize: bool = False) -> dict:
    """The `config` a sweep run logs: what was measured, with which model.
    `normalize` appears only when on, so earlier runs' configs stay as logged."""
    config = {"k": k, "embedding": model, "chunking": cfg.model_dump()}
    if normalize:
        config["normalize"] = True
    return config


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


def median_query_ms(
    embed_query: Callable[[str], object],
    queries: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Median milliseconds to embed one query, after one untimed warm-up call.

    The warm-up absorbs the model's first-run cost, which a server pays once
    at start, not per search."""
    embed_query(queries[0])
    timings = []
    for query in queries:
        start = clock()
        embed_query(query)
        timings.append((clock() - start) * 1000)
    return statistics.median(timings)
