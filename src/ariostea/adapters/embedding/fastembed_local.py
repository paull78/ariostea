from __future__ import annotations

from typing import Sequence

from fastembed import TextEmbedding

from ariostea.ports.embedding import EmbeddingProvider

# Models trained with markers on each side of the retrieval pair: (query,
# passage). fastembed does not add them, and without them retrieval quality
# drops. Models absent here take the text as is; jina-embeddings-v3 selects
# its query and passage tasks inside fastembed's query_embed/passage_embed.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


class FastEmbedEmbeddings(EmbeddingProvider):
    # Kept in step with EmbeddingCfg.local_model, the configured default.
    def __init__(
        self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ) -> None:
        self._model_name = model_name
        self._query_prefix, self._passage_prefix = _PREFIXES.get(model_name, ("", ""))
        self._model = TextEmbedding(model_name=model_name)
        self._dim: int | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        passages = [self._passage_prefix + text for text in texts]
        return [vec.tolist() for vec in self._model.passage_embed(passages)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([self._query_prefix + text]))).tolist()

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
        base = f"fastembed:{self._model_name}"
        if not (self._query_prefix or self._passage_prefix):
            return base
        return f"{base}|query={self._query_prefix!r}|passage={self._passage_prefix!r}"
