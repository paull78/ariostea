from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ariostea.domain.models import ContextualizedChunk, IndexStats
from ariostea.indexing.scanner import scan_vault
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import Chunker, MarkdownParser
from ariostea.ports.store import DocumentWriter, IndexAdmin


class IndexVault:
    def __init__(
        self,
        parser: MarkdownParser,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        store: DocumentWriter | IndexAdmin,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    def index(self, root: str | Path, ignore: Sequence[str] = ()) -> IndexStats:
        for scanned in scan_vault(root, ignore=ignore):
            note, body = self._parser.parse(scanned.rel_path, scanned.raw, scanned.mtime)
            chunks = self._chunker.chunk(note, body)
            if not chunks:
                continue
            cchunks = [
                ContextualizedChunk(chunk=c, context_blurb=None, embedding_text=c.text)
                for c in chunks
            ]
            vectors = self._embeddings.embed_documents([cc.embedding_text for cc in cchunks])
            self._store.upsert_note(note, cchunks, vectors)
        self._store.set_fingerprint(self._embeddings.fingerprint)
        return self._store.stats()
