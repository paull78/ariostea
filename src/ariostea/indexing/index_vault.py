from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ariostea.domain.models import IndexStats
from ariostea.indexing.scanner import scan_vault
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import Chunker, Contextualizer, MarkdownParser
from ariostea.ports.store import IndexStore


class IndexVault:
    def __init__(
        self,
        parser: MarkdownParser,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        store: IndexStore,
        contextualizer: Contextualizer,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store
        self._contextualizer = contextualizer

    def _fingerprint(self) -> str:
        # Both the embedding model AND the contextualization change embedding_text,
        # so a change in either must invalidate every stored vector.
        return f"{self._embeddings.fingerprint}|{self._contextualizer.fingerprint}"

    def index(self, root: str | Path, ignore: Sequence[str] = ()) -> IndexStats:
        seen: set[str] = set()
        known = self._store.known_hashes()
        fingerprint_changed = self._store.fingerprint() != self._fingerprint()

        for scanned in scan_vault(root, ignore=ignore):
            if not fingerprint_changed and known.get(scanned.rel_path) == scanned.content_hash:
                seen.add(scanned.rel_path)  # unchanged & already indexed — keep it
                continue
            note, body = self._parser.parse(scanned.rel_path, scanned.raw, scanned.mtime)
            chunks = self._chunker.chunk(note, body)
            if not chunks:
                continue
            cchunks = self._contextualizer.contextualize(note, body, chunks)
            vectors = self._embeddings.embed_documents([cc.embedding_text for cc in cchunks])
            self._store.upsert_note(note, cchunks, vectors)
            seen.add(note.path)
        for path in list(self._store.known_hashes()):
            if path not in seen:
                self._store.delete_note(path)
        self._store.set_fingerprint(self._fingerprint())
        return self._store.stats()
