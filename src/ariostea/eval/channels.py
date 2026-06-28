"""Per-channel search functions for the eval harness.

Each factory returns a SearchFn (query, k -> note paths) that exercises a
single retrieval channel in isolation, so the harness can attribute results
to the dense or sparse side rather than only the blended pipeline.
"""

from __future__ import annotations

from ariostea.eval.harness import SearchFn, dedupe
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.store import ChunkRetriever


def make_dense_search_fn(
    embeddings: EmbeddingProvider, retriever: ChunkRetriever, pool: int
) -> SearchFn:
    def search_fn(query: str, k: int) -> list[str]:
        vec = embeddings.embed_query(query)
        hits = retriever.dense(vec=vec, k=pool, filters=None)
        return dedupe([h.chunk.note_path for h in hits])[:k]

    return search_fn


def make_sparse_search_fn(retriever: ChunkRetriever, pool: int) -> SearchFn:
    def search_fn(query: str, k: int) -> list[str]:
        hits = retriever.sparse(query=query, k=pool, filters=None)
        return dedupe([h.chunk.note_path for h in hits])[:k]

    return search_fn
