from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Note:
    path: str
    title: str
    frontmatter: dict[str, Any]
    tags: tuple[str, ...]
    wikilinks: tuple[str, ...]
    content_hash: str
    mtime: float


@dataclass(frozen=True)
class Chunk:
    note_path: str
    ordinal: int
    heading_path: tuple[str, ...]
    text: str
    token_count: int


@dataclass(frozen=True)
class ContextualizedChunk:
    chunk: Chunk
    context_blurb: str | None
    embedding_text: str


def contextual_text(blurb: str | None, text: str) -> str:
    """A chunk as contextual indexing embeds it: the note's blurb, a blank
    line, then the chunk. Without a blurb, the chunk alone. Indexing and the
    blurb-aware reranker both call this so they judge the same text."""
    return f"{blurb}\n\n{text}" if blurb else text


@dataclass(frozen=True)
class QueryFilters:
    tags: tuple[str, ...] = ()
    path_globs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Query:
    text: str
    k: int = 10
    filters: QueryFilters | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    # The note-level blurb from contextual indexing, for a reranker that
    # should judge the chunk with the context the first stages saw.
    context_blurb: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunks: tuple[RetrievedChunk, ...]


@dataclass(frozen=True)
class SourceHit:
    note_path: str
    title: str
    hit_count: int
    best_score: float
    snippets: tuple[str, ...]


@dataclass(frozen=True)
class NoteText:
    """A note's file text as written, with its indexed title."""

    path: str
    title: str
    text: str


@dataclass(frozen=True)
class NoteMissing:
    """No note to return: not in the index (`reason` None), or indexed but
    its file cannot be read now (`reason` says why)."""

    path: str
    reason: str | None


@dataclass(frozen=True)
class IndexStats:
    notes: int
    chunks: int


@dataclass(frozen=True)
class RunRecord:
    """How an index run ended, as the index stores it. `error` is None for a
    run that completed; `notes` and `chunks` count the whole index after it."""

    finished_at: datetime
    fingerprint: str
    notes: int
    chunks: int
    skipped: int
    error: str | None


@dataclass(frozen=True)
class StoredNote:
    """What the index knows of a note: the hash of the text it embedded and
    the fingerprint (model, contextualizer, chunking) it embedded it with."""

    content_hash: str
    fingerprint: str


@dataclass(frozen=True)
class Progress:
    """Where an indexing pass stands, reported after each note it lists."""

    done: int
    total: int
    embedded: int
    skipped: int


@dataclass(frozen=True)
class IndexReport:
    """What one indexing pass did. `notes` and `chunks` count the whole index
    afterwards; `skipped` pairs each unreadable path with the reason."""

    notes: int
    chunks: int
    embedded: tuple[str, ...]
    deleted: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
