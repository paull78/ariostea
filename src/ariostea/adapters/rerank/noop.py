from __future__ import annotations

from ariostea.domain.models import RetrievedChunk
from ariostea.ports.rerank import Reranker


class NoopReranker(Reranker):
    """Identity reranker: keep the fused order, just truncate to top_n.

    Used when reranking is disabled or the model is unavailable, and as a
    deterministic test double.
    """

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return list(candidates[:top_n])
