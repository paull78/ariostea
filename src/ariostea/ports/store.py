from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ariostea.domain.models import (
    ContextualizedChunk,
    IndexStats,
    Note,
    QueryFilters,
    RetrievedChunk,
)


@runtime_checkable
class DocumentWriter(Protocol):
    def upsert_note(
        self,
        note: Note,
        chunks: Sequence[ContextualizedChunk],
        embeddings: Sequence[list[float]],
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
class IndexAdmin(Protocol):
    def known_hashes(self) -> dict[str, str]: ...
    def stats(self) -> IndexStats: ...
    def fingerprint(self) -> str: ...
    def set_fingerprint(self, value: str) -> None: ...


@runtime_checkable
class IndexStore(DocumentWriter, IndexAdmin, Protocol):
    """Composite role for the indexing use case: it both writes notes and
    administers the index (hashes, fingerprint, stats). Python has no
    intersection type, so we name the combination as one Protocol."""
