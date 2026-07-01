"""Per-channel search functions for the eval harness.

Each factory returns a SearchFn (query, k -> note paths) that exercises a
single retrieval channel in isolation, so the harness can attribute results
to the dense or sparse side rather than only the blended pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ariostea.eval.harness import SearchFn, dedupe
from ariostea.mcp.handlers import search_payload
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.store import ChunkRetriever

if TYPE_CHECKING:
    from ariostea.config.container import Container


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


def make_hybrid_search_fn(container: "Container", pool: int) -> SearchFn:
    """Full blended pipeline (dense+sparse+fuse+rerank) via the production
    search use case, deduped to notes. Pulls a generous chunk pool, then
    collapses to note paths before taking the top k."""

    def search_fn(query: str, k: int) -> list[str]:
        payload = search_payload(container, query=query, k=pool)
        return dedupe([r["note_path"] for r in payload["results"]])[:k]

    return search_fn
