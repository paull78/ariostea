from ariostea.adapters.fuse.rrf import RRFFuser
from ariostea.domain.models import Chunk, Query, RetrievedChunk
from ariostea.search.search_knowledge import SearchKnowledge


class FakeEmbed:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [float(len(text))]

    @property
    def dimension(self):
        return 1

    @property
    def fingerprint(self):
        return "fake"


def _chunk(ordinal, text, path="a.md"):
    return Chunk(note_path=path, ordinal=ordinal, heading_path=("A",), text=text, token_count=1)


class FakeRetriever:
    def __init__(self):
        self.dense_call = None
        self.sparse_call = None

    def dense(self, vec, k, filters=None):
        self.dense_call = (vec, k, filters)
        return [RetrievedChunk(chunk=_chunk(0, "semantic"), score=0.5, dense_rank=0)]

    def sparse(self, query, k, filters=None):
        self.sparse_call = (query, k, filters)
        return [RetrievedChunk(chunk=_chunk(1, "lexical"), score=2.0, sparse_rank=0)]


def test_search_runs_dense_and_sparse_and_fuses():
    retriever = FakeRetriever()
    uc = SearchKnowledge(
        embeddings=FakeEmbed(),
        retriever=retriever,
        fuser=RRFFuser(),
        k_dense=40,
        k_sparse=30,
    )
    result = uc.search(Query(text="hello", k=5))

    # both channels were exercised
    assert retriever.dense_call[0] == [5.0]  # embedded query ("hello" -> len 5)
    assert retriever.dense_call[1] == 40  # k_dense from construction
    assert retriever.sparse_call[0] == "hello"  # raw text to BM25
    assert retriever.sparse_call[1] == 30  # k_sparse from construction

    # fusion merged both unique chunks into the result
    texts = {c.chunk.text for c in result.chunks}
    assert texts == {"semantic", "lexical"}


def test_search_truncates_to_query_k():
    class ManyRetriever(FakeRetriever):
        def dense(self, vec, k, filters=None):
            return [
                RetrievedChunk(chunk=_chunk(i, f"d{i}"), score=1.0, dense_rank=i) for i in range(5)
            ]

        def sparse(self, query, k, filters=None):
            return []

    uc = SearchKnowledge(embeddings=FakeEmbed(), retriever=ManyRetriever(), fuser=RRFFuser())
    result = uc.search(Query(text="x", k=2))
    assert len(result.chunks) == 2
