from __future__ import annotations

from typing import Sequence

from fastembed import TextEmbedding

from ariostea.ports.embedding import EmbeddingProvider


class FastEmbedEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._dim: int | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()

    def count_tokens(self, text: str) -> int:
        """Subword tokens the model reads for `text`, without the start and end
        markers it adds. Not on the EmbeddingProvider port: only a local model
        has a tokenizer to ask. Used to size chunks in the model's own units."""
        return len(self._model.model.tokenizer.encode(text, add_special_tokens=False).ids)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"fastembed:{self._model_name}"
