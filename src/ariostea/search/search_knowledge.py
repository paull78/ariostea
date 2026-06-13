from __future__ import annotations

from ariostea.domain.models import Query, SearchResult
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.store import ChunkRetriever


class SearchKnowledge:
    def __init__(self, embeddings: EmbeddingProvider, retriever: ChunkRetriever) -> None:
        self._embeddings = embeddings
        self._retriever = retriever

    def search(self, query: Query) -> SearchResult:
        vec = self._embeddings.embed_query(query.text)
        hits = self._retriever.dense(vec, k=query.k, filters=query.filters)
        return SearchResult(chunks=tuple(hits))
