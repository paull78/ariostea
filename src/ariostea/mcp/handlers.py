from __future__ import annotations

from ariostea.ports.store import IndexAdmin


def status_payload(admin: IndexAdmin) -> dict:
    s = admin.stats()
    return {
        "notes": s.notes,
        "chunks": s.chunks,
        "last_indexed": s.last_indexed,
        "config_fingerprint": s.config_fingerprint,
    }
