from __future__ import annotations

from typing import Protocol, runtime_checkable

from ariostea.domain.models import Note, Chunk


@runtime_checkable
class MarkdownParser(Protocol):
    def parse(self, path: str, raw: str, mtime: float) -> tuple[Note, str]:
        """Return (note metadata, body-without-frontmatter)."""
        ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, note: Note, body: str) -> list[Chunk]: ...
