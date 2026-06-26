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


def test_chunk_retriever_requires_dense_and_sparse():
    class DenseOnly:
        def dense(self, vec, k, filters=None):
            return []

    class DenseAndSparse:
        def dense(self, vec, k, filters=None):
            return []

        def sparse(self, query, k, filters=None):
            return []

    # A retriever missing sparse() does NOT satisfy the widened port...
    assert not isinstance(DenseOnly(), ChunkRetriever)
    # ...one providing both does.
    assert isinstance(DenseAndSparse(), ChunkRetriever)


def test_rrf_fuser_conforms_to_port():
    from ariostea.adapters.fuse.rrf import RRFFuser
    from ariostea.ports.fusion import Fuser

    assert isinstance(RRFFuser(), Fuser)


def test_sqlite_store_conforms_to_document_reader():
    from ariostea.adapters.store.sqlite_store import SqliteStore
    from ariostea.ports.store import DocumentReader

    assert issubclass(SqliteStore, DocumentReader)
