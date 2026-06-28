from __future__ import annotations

from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ariostea.domain.models import RetrievedChunk
from ariostea.ports.rerank import Reranker


class FastEmbedReranker(Reranker):
    """Multilingual cross-encoder reranker (ONNX via fastembed).

    Scores each candidate passage against the query and returns the top_n by
    relevance. The default model is multilingual on purpose: an English-only
    cross-encoder would score cross-lingual passages low and defeat the point.
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual") -> None:
        self._model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [rc.chunk.text for rc in candidates]))
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(rc, score=float(score)) for rc, score in ranked[:top_n]]
