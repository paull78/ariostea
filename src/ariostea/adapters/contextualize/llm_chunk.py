from __future__ import annotations

import logging
from collections.abc import Sequence

from ariostea.domain.models import Chunk, ContextualizedChunk, Note, contextual_text
from ariostea.ports.chat import ChatProvider
from ariostea.ports.pipeline import Contextualizer

logger = logging.getLogger(__name__)

_SYSTEM = "You situate chunks of a document for search retrieval.\n\n<document>\n{doc}\n</document>"
_USER = (
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. Write it in "
    "the same language as the chunk. Answer only with the succinct context and nothing else."
)


class LLMChunkContextualizer(Contextualizer):
    """Anthropic-style Contextual Retrieval: one LLM call per chunk, each seeing
    the whole note. The note goes in the system message, identical for every
    chunk, so a server with prompt caching processes it once per note. A failed
    call degrades only its own chunk to plain text."""

    def __init__(self, chat: ChatProvider, model_name: str) -> None:
        self._chat = chat
        self._model_name = model_name

    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]:
        system = _SYSTEM.format(doc=full_doc)
        out = []
        for chunk in chunks:
            failed = False
            try:
                context = self._chat.complete(
                    system=system, user=_USER.format(chunk=chunk.text)
                ).strip()
            except Exception as exc:  # provider down / timeout / bad response
                logger.warning(
                    "contextualization failed for %s chunk %d; indexing plain",
                    note.path,
                    chunk.ordinal,
                    exc_info=exc,
                )
                context = ""
                failed = True
            if not context:
                if not failed:
                    logger.warning(
                        "empty context for %s chunk %d; indexing plain", note.path, chunk.ordinal
                    )
                out.append(
                    ContextualizedChunk(chunk=chunk, context_blurb=None, embedding_text=chunk.text)
                )
                continue
            out.append(
                ContextualizedChunk(
                    chunk=chunk,
                    context_blurb=context,
                    embedding_text=contextual_text(context, chunk.text),
                )
            )
        return out

    @property
    def fingerprint(self) -> str:
        return f"llm-chunk:{self._model_name}"
