from __future__ import annotations

from ariostea.domain.models import Query, SearchResult
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.fusion import Fuser
from ariostea.ports.rerank import Reranker
from ariostea.ports.store import ChunkRetriever


class SearchKnowledge:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        retriever: ChunkRetriever,
        fuser: Fuser,
        reranker: Reranker,
        k_dense: int = 50,
        k_sparse: int = 50,
        pool: int = 100,
    ) -> None:
        self._embeddings = embeddings
        self._retriever = retriever
        self._fuser = fuser
        self._reranker = reranker
        self._k_dense = k_dense
        self._k_sparse = k_sparse
        self._pool = pool

    def search(self, query: Query) -> SearchResult:
        vec = self._embeddings.embed_query(query.text)
        dense = self._retriever.dense(vec=vec, k=self._k_dense, filters=query.filters)
        sparse = self._retriever.sparse(query=query.text, k=self._k_sparse, filters=query.filters)
        # RRF is a recall gatherer: fuse a large pool, then let the reranker
        # pick the final top_k by true query-passage relevance.
        fused = self._fuser.fuse(dense=dense, sparse=sparse, k=self._pool)
        ranked = self._reranker.rerank(query.text, fused, top_n=query.k)
        return SearchResult(chunks=tuple(ranked))
