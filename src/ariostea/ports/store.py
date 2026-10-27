from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ariostea.domain.models import (
    ContextualizedChunk,
    IndexStats,
    Note,
    QueryFilters,
    RetrievedChunk,
    StoredNote,
)


@runtime_checkable
class DocumentWriter(Protocol):
    def upsert_note(
        self,
        note: Note,
        chunks: Sequence[ContextualizedChunk],
        embeddings: Sequence[list[float]],
        fingerprint: str,
    ) -> None: ...
    def delete_note(self, path: str) -> None: ...


@runtime_checkable
class ChunkRetriever(Protocol):
    def dense(
        self, vec: list[float], k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]: ...
    def sparse(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class DocumentReader(Protocol):
    def note_titles(self, paths: Sequence[str]) -> dict[str, str]: ...


@runtime_checkable
class IndexAdmin(Protocol):
    def known_notes(self) -> dict[str, StoredNote]: ...
    def stats(self) -> IndexStats: ...


@runtime_checkable
class IndexStore(DocumentWriter, IndexAdmin, Protocol):
    """Composite role for the indexing use case: it both writes notes and
    administers the index (what each note was embedded from, stats). Python has no
    intersection type, so we name the combination as one Protocol."""
