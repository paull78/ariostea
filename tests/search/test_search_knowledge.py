from ariostea.search.search_knowledge import SearchKnowledge
from ariostea.domain.models import Query, Chunk, RetrievedChunk


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


class FakeRetriever:
    def __init__(self):
        self.last = None

    def dense(self, vec, k, filters=None):
        self.last = (vec, k, filters)
        chunk = Chunk(note_path="a.md", ordinal=0, heading_path=("A",), text="match", token_count=1)
        return [RetrievedChunk(chunk=chunk, score=0.5, dense_rank=0)]


def test_search_embeds_query_and_returns_results():
    retriever = FakeRetriever()
    uc = SearchKnowledge(embeddings=FakeEmbed(), retriever=retriever)
    result = uc.search(Query(text="hello", k=5))

    assert result.chunks[0].chunk.text == "match"
    # query was embedded and passed through with k
    assert retriever.last[0] == [5.0]  # len("hello")
    assert retriever.last[1] == 5
