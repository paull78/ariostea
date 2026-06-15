from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import Chunker, MarkdownParser
from ariostea.ports.store import ChunkRetriever, DocumentWriter, IndexAdmin


class FakeEmbed:
    def embed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 1.0]

    @property
    def dimension(self):
        return 2

    @property
    def fingerprint(self):
        return "fake:v1"


def test_fake_embedder_conforms():
    assert isinstance(FakeEmbed(), EmbeddingProvider)


def test_store_role_ports_are_distinct():
    # A class can satisfy multiple role ports; the ports themselves are separate types.
    assert DocumentWriter is not ChunkRetriever is not IndexAdmin
    assert {DocumentWriter, ChunkRetriever, IndexAdmin}.__len__() == 3


def test_pipeline_ports_exist():
    assert MarkdownParser is not None and Chunker is not None
