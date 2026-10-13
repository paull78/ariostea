from __future__ import annotations

from typing import Sequence

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from ariostea.adapters.fastembed_cache import fastembed_cache_dir
from ariostea.ports.embedding import EmbeddingProvider

# Models trained with markers on each side of the retrieval pair: (query,
# passage). fastembed does not add them, and without them retrieval quality
# drops. Models absent here take the text as is; jina-embeddings-v3 selects
# its query and passage tasks inside fastembed's query_embed/passage_embed.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


# Models fastembed does not list but can load from a Hugging Face ONNX export
# once registered. bge-m3's dense output is the normalised CLS vector.
_CUSTOM_MODELS: dict[str, dict] = {
    "BAAI/bge-m3": {
        "pooling": PoolingType.CLS,
        "normalization": True,
        "sources": ModelSource(hf="BAAI/bge-m3"),
        "dim": 1024,
        "model_file": "onnx/model.onnx",
        "additional_files": ["onnx/model.onnx_data"],
        "license": "mit",
        "size_in_gb": 2.27,
    },
}


def _register_if_custom(model_name: str) -> None:
    """Register `model_name` with fastembed if it is one of ours and not yet
    known. fastembed keeps the registry per process and raises on a second
    registration, so ask it rather than trusting a local flag."""
    spec = _CUSTOM_MODELS.get(model_name)
    if spec is None:
        return
    if any(m["model"] == model_name for m in TextEmbedding.list_supported_models()):
        return
    TextEmbedding.add_custom_model(model=model_name, **spec)


class FastEmbedEmbeddings(EmbeddingProvider):
    # Kept in step with EmbeddingCfg.local_model, the configured default.
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        normalize: bool = False,
    ) -> None:
        self._model_name = model_name
        self._normalize = normalize
        self._query_prefix, self._passage_prefix = _PREFIXES.get(model_name, ("", ""))
        _register_if_custom(model_name)
        self._model = TextEmbedding(model_name=model_name, cache_dir=fastembed_cache_dir())
        self._dim: int | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        passages = [self._passage_prefix + text for text in texts]
        return [self._finish(vec) for vec in self._model.passage_embed(passages)]

    def embed_query(self, text: str) -> list[float]:
        return self._finish(next(iter(self._model.query_embed([self._query_prefix + text]))))

    def _finish(self, vec: np.ndarray) -> list[float]:
        """The vector as a list, scaled to length 1 when normalising. A zero
        vector has no direction and is returned as is."""
        if not self._normalize:
            return vec.tolist()
        vec = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm else vec.tolist()

    def count_tokens(self, text: str) -> int:
        """Subword tokens the model reads for `text`, without the start and end
        markers it adds. Not on the EmbeddingProvider port: only a local model
        has a tokenizer to ask. Used to size chunks in the model's own units.

        Counts the raw text, never a query or passage prefix, so a chunk size
        means the same text under every model."""
        return len(self._model.model.tokenizer.encode(text, add_special_tokens=False).ids)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    @property
    def fingerprint(self) -> str:
        # Unprefixed models keep the bare form, so vaults indexed before
        # prefixes existed keep their fingerprint and are not re-embedded.
        fingerprint = f"fastembed:{self._model_name}"
        if self._query_prefix or self._passage_prefix:
            fingerprint += f"|query={self._query_prefix!r}|passage={self._passage_prefix!r}"
        if self._normalize:
            fingerprint += "|unit"
        return fingerprint
