from ariostea.indexing.index_vault import IndexVault
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker


class FakeEmbed:
    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return [[float(len(t)), 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0]

    @property
    def dimension(self):
        return 2

    @property
    def fingerprint(self):
        return "fake:v1"


class FakeStore:
    def __init__(self):
        self.notes = {}
        self._fp = ""

    def upsert_note(self, note, chunks, embeddings):
        self.notes[note.path] = (note, list(chunks), list(embeddings))

    def delete_note(self, path):
        self.notes.pop(path, None)

    def known_hashes(self):
        return {p: n.content_hash for p, (n, _, _) in self.notes.items()}

    def stats(self):
        from ariostea.domain.models import IndexStats
        return IndexStats(len(self.notes), 0, 0.0, self._fp)

    def fingerprint(self):
        return self._fp

    def set_fingerprint(self, value):
        self._fp = value


def test_index_vault_indexes_each_note(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    (tmp_path / "b.md").write_text("# B\nbeta content here")

    embed, store = FakeEmbed(), FakeStore()
    indexer = IndexVault(
        parser=ObsidianMarkdownParser(),
        chunker=HeadingAwareChunker(max_tokens=200),
        embeddings=embed,
        store=store,
    )
    stats = indexer.index(tmp_path, ignore=[])

    assert set(store.notes) == {"a.md", "b.md"}
    assert stats.notes == 2
    # embeddings were requested for the chunk text
    assert any("alpha content here" in t for t in embed.seen)
    # fingerprint recorded so later runs can detect model changes
    assert store.fingerprint() == "fake:v1"


def test_embedding_text_defaults_to_chunk_text(tmp_path):
    (tmp_path / "a.md").write_text("# A\nplain text")
    store = FakeStore()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store).index(tmp_path, ignore=[])
    _, chunks, _ = store.notes["a.md"]
    assert chunks[0].context_blurb is None
    assert chunks[0].embedding_text == chunks[0].chunk.text
