from __future__ import annotations

from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ariostea.domain.models import RetrievedChunk, contextual_text
from ariostea.ports.rerank import Reranker


class FastEmbedReranker(Reranker):
    """Multilingual cross-encoder reranker (ONNX via fastembed).

    Scores each candidate passage against the query and returns the top_n by
    relevance. The default model is multilingual on purpose: an English-only
    cross-encoder would score cross-lingual passages low and defeat the point.

    With `use_context`, a candidate's context blurb is prepended to its text in
    the same format contextual indexing embeds, so the reranker judges what the
    dense and sparse stages matched rather than the bare chunk.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        use_context: bool = False,
    ) -> None:
        self._model_name = model_name
        self._use_context = use_context
        self._model = TextCrossEncoder(model_name=model_name)

    def _passage(self, rc: RetrievedChunk) -> str:
        return (
            contextual_text(rc.context_blurb, rc.chunk.text) if self._use_context else rc.chunk.text
        )

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [self._passage(rc) for rc in candidates]))
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(rc, score=float(score)) for rc, score in ranked[:top_n]]
