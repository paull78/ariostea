from __future__ import annotations

import hashlib

from ariostea.domain.models import IndexReport
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import Chunker, Contextualizer, MarkdownParser
from ariostea.ports.store import IndexStore
from ariostea.ports.vault import NoteUnreadable, Vault, VaultUnavailable


class IndexVault:
    def __init__(
        self,
        parser: MarkdownParser,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        store: IndexStore,
        contextualizer: Contextualizer,
        vault: Vault,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store
        self._contextualizer = contextualizer
        self._vault = vault

    def _fingerprint(self) -> str:
        # The embedding model and the contextualization both change embedding_text,
        # and the chunking policy changes which texts exist at all, so a change in
        # any of them must rebuild every note. Empty parts are skipped: the default
        # chunker's fingerprint is "" so indexes built before chunking was
        # configurable keep the "emb|ctx" string they stored, and are not forced
        # into a full reindex on upgrade.
        parts = (
            self._embeddings.fingerprint,
            self._contextualizer.fingerprint,
            self._chunker.fingerprint,
        )
        return "|".join(part for part in parts if part)

    def index(self) -> IndexReport:
        """Bring the index in line with the vault. Raises VaultUnavailable,
        before touching the index, when the vault is missing or holds no notes
        while the index does: either would otherwise delete every note."""
        paths = self._vault.notes()
        known = self._store.known_hashes()
        if not paths and known:
            raise VaultUnavailable(
                self._vault.location, f"no notes found while the index holds {len(known)}"
            )
        fingerprint_changed = self._store.fingerprint() != self._fingerprint()
        seen: set[str] = set()
        embedded: list[str] = []
        skipped: list[tuple[str, str]] = []

        for path in paths:
            try:
                file = self._vault.read(path)
            except NoteUnreadable as exc:
                skipped.append((path, exc.reason))
                seen.add(path)  # keep what it had: a bad read is not a deletion
                continue
            content_hash = hashlib.sha256(file.text.encode("utf-8")).hexdigest()
            if not fingerprint_changed and known.get(path) == content_hash:
                seen.add(path)  # unchanged & already indexed — keep it
                continue
            note, body = self._parser.parse(path, file.text, file.mtime)
            chunks = self._chunker.chunk(note, body)
            if not chunks:
                continue
            cchunks = self._contextualizer.contextualize(note, body, chunks)
            vectors = self._embeddings.embed_documents([cc.embedding_text for cc in cchunks])
            self._store.upsert_note(note, cchunks, vectors)
            seen.add(path)
            embedded.append(path)

        # A note left out of `seen` is gone, excluded, or has no chunks left.
        deleted = [path for path in self._store.known_hashes() if path not in seen]
        for path in deleted:
            self._store.delete_note(path)
        self._store.set_fingerprint(self._fingerprint())
        stats = self._store.stats()
        return IndexReport(
            notes=stats.notes,
            chunks=stats.chunks,
            embedded=tuple(embedded),
            deleted=tuple(sorted(deleted)),
            skipped=tuple(skipped),
        )
