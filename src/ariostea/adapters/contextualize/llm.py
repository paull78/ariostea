from __future__ import annotations

import logging
from collections.abc import Sequence

from ariostea.domain.models import Chunk, ContextualizedChunk, Note
from ariostea.ports.chat import ChatProvider
from ariostea.ports.pipeline import Contextualizer

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "You write a short context blurb (one or two sentences, about 50 words) that situates "
    "the following note for search retrieval: state its main topic and key entities so a "
    "fragment of it can be found out of context. Output only the blurb, with no preamble."
)


class LLMContextualizer(Contextualizer):
    """Generate one note-level blurb via a ChatProvider and prepend it to every
    chunk. Any failure (or an empty blurb) degrades the whole note to plain text
    so indexing is never blocked."""

    def __init__(self, chat: ChatProvider, model_name: str) -> None:
        self._chat = chat
        self._model_name = model_name

    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]:
        try:
            blurb = self._chat.complete(system=_INSTRUCTIONS, user=full_doc).strip()
        except Exception as exc:  # provider down / timeout / bad response
            logger.warning(
                "contextualization failed for %s; indexing plain", note.path, exc_info=exc
            )
            blurb = ""
        if not blurb:
            return [
                ContextualizedChunk(chunk=c, context_blurb=None, embedding_text=c.text)
                for c in chunks
            ]
        return [
            ContextualizedChunk(
                chunk=c, context_blurb=blurb, embedding_text=f"{blurb}\n\n{c.text}"
            )
            for c in chunks
        ]

    @property
    def fingerprint(self) -> str:
        return f"llm:{self._model_name}"
