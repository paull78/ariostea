import numpy as np
import pytest

from ariostea.adapters.embedding import fastembed_local
from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings

CONTROL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
E5 = "intfloat/multilingual-e5-large"


class _FakeTokenizer:
    """Counts characters, so a prefix would show up in the count."""

    def encode(self, text, add_special_tokens=True):
        class _Encoding:
            ids = list(text)

        return _Encoding()


class _FakeTextEmbedding:
    """Stands in for fastembed's TextEmbedding and records what it is sent."""

    instances: list["_FakeTextEmbedding"] = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.calls: list[tuple[str, list[str]]] = []
        self.model = type("_Inner", (), {"tokenizer": _FakeTokenizer()})()
        _FakeTextEmbedding.instances.append(self)

    def _vectors(self, method, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append((method, texts))
        return iter(np.ones(3, dtype=np.float32) for _ in texts)

    def embed(self, texts, **kwargs):
        return self._vectors("embed", texts)

    def query_embed(self, query, **kwargs):
        return self._vectors("query_embed", query)

    def passage_embed(self, texts, **kwargs):
        return self._vectors("passage_embed", texts)


@pytest.fixture
def fake(monkeypatch):
    _FakeTextEmbedding.instances = []
    monkeypatch.setattr(fastembed_local, "TextEmbedding", _FakeTextEmbedding)

    def calls():
        return _FakeTextEmbedding.instances[-1].calls

    return calls


def test_documents_go_through_passage_embed(fake):
    FastEmbedEmbeddings(CONTROL).embed_documents(["a", "b"])
    assert fake() == [("passage_embed", ["a", "b"])]


def test_queries_go_through_query_embed(fake):
    FastEmbedEmbeddings(CONTROL).embed_query("q")
    assert fake() == [("query_embed", ["q"])]


def test_e5_prefixes_passages_and_queries(fake):
    # e5 was trained with these markers; without them its retrieval degrades,
    # so a run without them would measure a misconfiguration.
    emb = FastEmbedEmbeddings(E5)
    emb.embed_documents(["a"])
    emb.embed_query("q")
    assert fake() == [("passage_embed", ["passage: a"]), ("query_embed", ["query: q"])]


def test_models_without_prefixes_get_the_text_unchanged(fake):
    emb = FastEmbedEmbeddings(CONTROL)
    emb.embed_documents(["a"])
    emb.embed_query("q")
    assert fake() == [("passage_embed", ["a"]), ("query_embed", ["q"])]


def test_control_fingerprint_is_unchanged(fake):
    # Existing vaults were indexed under this fingerprint; changing it would
    # force every one of them to re-embed for no change in the vectors.
    assert FastEmbedEmbeddings(CONTROL).fingerprint == f"fastembed:{CONTROL}"


def test_prefixed_model_fingerprint_names_the_prefixes(fake):
    fingerprint = FastEmbedEmbeddings(E5).fingerprint
    assert fingerprint != f"fastembed:{E5}"
    assert fingerprint.startswith(f"fastembed:{E5}")
    assert "query: " in fingerprint and "passage: " in fingerprint


def test_count_tokens_ignores_the_prefix(fake):
    # Chunk sizes must mean the same text under every model.
    assert FastEmbedEmbeddings(E5).count_tokens("abc") == 3
    assert FastEmbedEmbeddings(CONTROL).count_tokens("abc") == 3


@pytest.mark.integration
def test_control_vectors_match_plain_embed():
    from fastembed import TextEmbedding

    text = "Il violino si accorda per quinte."
    plain = next(iter(TextEmbedding(model_name=CONTROL).embed([text])))
    emb = FastEmbedEmbeddings(CONTROL)
    assert np.allclose(emb.embed_documents([text])[0], plain, atol=1e-6)
    assert np.allclose(emb.embed_query(text), plain, atol=1e-6)


@pytest.mark.integration
def test_embeds_documents_and_query_consistently():
    emb = FastEmbedEmbeddings()  # the default multilingual model
    docs = emb.embed_documents(["cats and dogs", "vector databases"])
    q = emb.embed_query("vector databases")

    assert len(docs) == 2
    assert len(docs[0]) == emb.dimension == len(q)
    assert emb.fingerprint.startswith("fastembed:")

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    # query is closer to its matching doc than the unrelated one
    assert dot(q, docs[1]) > dot(q, docs[0])


@pytest.mark.integration
def test_count_tokens_excludes_the_models_special_tokens():
    emb = FastEmbedEmbeddings()
    assert emb.count_tokens("") == 0
    # Subword counting: a rare word costs more tokens than a common one.
    assert emb.count_tokens("the") == 1
    assert emb.count_tokens("Hallstatt") > 1
