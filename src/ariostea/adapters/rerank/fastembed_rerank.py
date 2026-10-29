from __future__ import annotations

from dataclasses import replace

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

from ariostea.adapters.fastembed_cache import fastembed_cache_dir
from ariostea.domain.models import RetrievedChunk, contextual_text
from ariostea.ports.rerank import Reranker

# Cross-encoders fastembed does not list but can load from a Hugging Face ONNX
# export once registered. mMiniLM-L12 is the small multilingual alternative
# measured in the search-speed cycle; the fp32 file runs the same on every
# platform (search-speed.13).
_CUSTOM_MODELS: dict[str, dict] = {
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": {
        "sources": ModelSource(hf="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        "model_file": "onnx/model.onnx",
        "license": "apache-2.0",
        "size_in_gb": 0.47,
    },
}


def _register_if_custom(model_name: str) -> None:
    """Register `model_name` with fastembed if it is one of ours and not yet
    known. fastembed keeps the registry per process and raises on a second
    registration, so ask it rather than trusting a local flag."""
    spec = _CUSTOM_MODELS.get(model_name)
    if spec is None:
        return
    if any(m["model"] == model_name for m in TextCrossEncoder.list_supported_models()):
        return
    TextCrossEncoder.add_custom_model(model=model_name, **spec)


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
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        use_context: bool = False,
    ) -> None:
        self._model_name = model_name
        self._use_context = use_context
        _register_if_custom(model_name)
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=fastembed_cache_dir())

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
