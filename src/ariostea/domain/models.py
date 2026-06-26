from __future__ import annotations

from dataclasses import dataclass
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
class NoteDocument:
    note_path: str
    title: str
    text: str


@dataclass(frozen=True)
class IndexStats:
    notes: int
    chunks: int
    last_indexed: float
    config_fingerprint: str
