from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ariostea.domain.models import Query
from ariostea.ports.store import IndexAdmin

if TYPE_CHECKING:
    from ariostea.config.container import Container


def status_payload(admin: IndexAdmin) -> dict:
    s = admin.stats()
    return {
        "notes": s.notes,
        "chunks": s.chunks,
        "last_indexed": s.last_indexed,
        "config_fingerprint": s.config_fingerprint,
    }


def reindex_payload(container: "Container") -> dict:
    vault_path = os.path.expanduser(container.config.vault.path)
    stats = container.indexer.index(vault_path, ignore=container.config.vault.ignore)
    return {"notes": stats.notes, "chunks": stats.chunks}


def search_payload(container: "Container", query: str, k: int = 10) -> dict:
    result = container.searcher.search(Query(text=query, k=k))
    return {
        "results": [
            {
                "note_path": rc.chunk.note_path,
                "heading_path": list(rc.chunk.heading_path),
                "text": rc.chunk.text,
                "score": rc.score,
            }
            for rc in result.chunks
        ]
    }


def search_sources_payload(container: "Container", query: str, k: int = 10) -> dict:
    hits = container.sources.search(Query(text=query, k=k))
    return {
        "sources": [
            {
                "note_path": h.note_path,
                "title": h.title,
                "hit_count": h.hit_count,
                "best_score": h.best_score,
                "snippets": list(h.snippets),
            }
            for h in hits
        ]
    }


def get_note_payload(container: "Container", path: str) -> dict:
    doc = container.reader.read_note(path)
    if doc is None:
        return {"found": False, "note_path": path}
    return {"found": True, "note_path": doc.note_path, "title": doc.title, "text": doc.text}
