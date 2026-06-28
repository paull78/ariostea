from __future__ import annotations

from typing import Protocol, runtime_checkable

from ariostea.domain.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...
