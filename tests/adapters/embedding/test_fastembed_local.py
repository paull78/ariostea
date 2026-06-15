import pytest

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings


@pytest.mark.integration
def test_embeds_documents_and_query_consistently():
    emb = FastEmbedEmbeddings()  # default BAAI/bge-small-en-v1.5
    docs = emb.embed_documents(["cats and dogs", "vector databases"])
    q = emb.embed_query("vector databases")

    assert len(docs) == 2
    assert len(docs[0]) == emb.dimension == len(q)
    assert emb.fingerprint.startswith("fastembed:")

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    # query is closer to its matching doc than the unrelated one
    assert dot(q, docs[1]) > dot(q, docs[0])
