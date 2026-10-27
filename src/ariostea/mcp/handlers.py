from __future__ import annotations

from collections.abc import Callable
from datetime import timezone
from typing import TYPE_CHECKING

from ariostea.domain.models import NoteMissing, Progress, Query, RunRecord
from ariostea.ports.vault import VaultUnavailable

if TYPE_CHECKING:
    from ariostea.config.container import Container
    from ariostea.mcp.background import BackgroundRuns


def status_payload(container: "Container", background: "BackgroundRuns | None" = None) -> dict:
    """Index health, read from the index, so it holds whichever process ran:
    `last_indexed` and `config_fingerprint` come from the last run that
    completed, `last_run` is the last that ended, `running` says whether any
    process is indexing, and `progress` appears only for a run this server
    started."""
    s = container.admin.stats()
    state = container.runs.state()
    completed = state.last_completed
    payload: dict = {
        "notes": s.notes,
        "chunks": s.chunks,
        "running": state.running,
        "last_indexed": _utc(completed) if completed else None,
        "config_fingerprint": completed.fingerprint if completed else None,
        "last_run": _run_payload(state.last_run) if state.last_run else None,
    }
    progress = background.progress() if background is not None else None
    if progress is not None:
        payload["progress"] = {
            "done": progress.done,
            "total": progress.total,
            "embedded": progress.embedded,
            "skipped": progress.skipped,
        }
    return payload


def _utc(record: RunRecord) -> str:
    return record.finished_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_payload(record: RunRecord) -> dict:
    return {
        "finished_at": _utc(record),
        "notes": record.notes,
        "chunks": record.chunks,
        "skipped": record.skipped,
        "error": record.error,
    }


def start_reindex_payload(background: "BackgroundRuns") -> dict:
    """Start an index run in the background, or say why not."""
    if background.start():
        return {"started": True}
    return {"started": False, "reason": "an index run is in progress"}


def _nothing() -> None:
    pass


def reindex_payload(
    container: "Container",
    progress: Callable[[Progress], None] | None = None,
    on_wait: Callable[[], None] = _nothing,
) -> dict:
    """Index the vault, waiting for a run in progress to end first (`on_wait`
    hears when that happens). A vault that cannot be trusted comes back as
    `{"error": ...}` with the index left as it was."""
    run = container.runs.begin(wait=True, on_wait=on_wait)
    try:
        report = run.execute(progress=progress)
    except VaultUnavailable as exc:
        return {"error": str(exc)}
    return {
        "notes": report.notes,
        "chunks": report.chunks,
        "skipped": [{"path": path, "reason": reason} for path, reason in report.skipped],
    }


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
    result = container.notes.get(path)
    if isinstance(result, NoteMissing):
        payload: dict = {"found": False, "note_path": path}
        if result.reason is not None:
            payload["reason"] = result.reason
        return payload
    return {"found": True, "note_path": result.path, "title": result.title, "text": result.text}
