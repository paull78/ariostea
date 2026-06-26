from __future__ import annotations

from ariostea.domain.models import Query, RetrievedChunk, SourceHit
from ariostea.ports.store import DocumentReader
from ariostea.search.search_knowledge import SearchKnowledge

_SNIPPET_CHARS = 160
_MAX_SNIPPETS = 3


class SearchSources:
    def __init__(self, searcher: SearchKnowledge, reader: DocumentReader, pool: int = 50) -> None:
        self._searcher = searcher
        self._reader = reader
        self._pool = pool

    def search(self, query: Query) -> list[SourceHit]:
        broad = Query(text=query.text, k=self._pool, filters=query.filters)
        chunks = self._searcher.search(broad).chunks

        groups: dict[str, list[RetrievedChunk]] = {}
        for rc in chunks:
            groups.setdefault(rc.chunk.note_path, []).append(rc)

        titles = self._reader.note_titles(list(groups))
        hits = [
            SourceHit(
                note_path=path,
                title=titles.get(path, path),
                hit_count=len(rcs),
                best_score=max(rc.score for rc in rcs),
                snippets=tuple(rc.chunk.text[:_SNIPPET_CHARS] for rc in rcs[:_MAX_SNIPPETS]),
            )
            for path, rcs in groups.items()
        ]
        hits.sort(key=lambda h: h.best_score, reverse=True)
        return hits[: query.k]
