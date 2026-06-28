from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ariostea.domain.models import Chunk, ContextualizedChunk, Note


@runtime_checkable
class MarkdownParser(Protocol):
    def parse(self, path: str, raw: str, mtime: float) -> tuple[Note, str]:
        """Return (note metadata, body-without-frontmatter)."""
        ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, note: Note, body: str) -> list[Chunk]: ...


@runtime_checkable
class Contextualizer(Protocol):
    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]: ...

    @property
    def fingerprint(self) -> str: ...
