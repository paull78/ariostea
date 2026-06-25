from __future__ import annotations

from ariostea.domain.models import Query, SearchResult
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.fusion import Fuser
from ariostea.ports.store import ChunkRetriever


class SearchKnowledge:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        retriever: ChunkRetriever,
        fuser: Fuser,
        k_dense: int = 50,
        k_sparse: int = 50,
    ) -> None:
        self._embeddings = embeddings
        self._retriever = retriever
        self._fuser = fuser
        self._k_dense = k_dense
        self._k_sparse = k_sparse

    def search(self, query: Query) -> SearchResult:
        vec = self._embeddings.embed_query(query.text)
        dense = self._retriever.dense(vec=vec, k=self._k_dense, filters=query.filters)
        sparse = self._retriever.sparse(query=query.text, k=self._k_sparse, filters=query.filters)
        fused = self._fuser.fuse(dense=dense, sparse=sparse, k=query.k)
        return SearchResult(chunks=tuple(fused))
