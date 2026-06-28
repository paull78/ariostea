from __future__ import annotations

from collections.abc import Sequence

from ariostea.domain.models import Chunk, ContextualizedChunk, Note
from ariostea.ports.pipeline import Contextualizer


class NoopContextualizer(Contextualizer):
    """No LLM: every chunk is embedded/indexed as its bare text."""

    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]:
        return [
            ContextualizedChunk(chunk=c, context_blurb=None, embedding_text=c.text) for c in chunks
        ]

    @property
    def fingerprint(self) -> str:
        return "noop"
