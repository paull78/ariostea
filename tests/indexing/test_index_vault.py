from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.indexing.index_vault import IndexVault


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
        contextualizer=NoopContextualizer(),
    )
    stats = indexer.index(tmp_path, ignore=[])

    assert set(store.notes) == {"a.md", "b.md"}
    assert stats.notes == 2
    # embeddings were requested for the chunk text
    assert any("alpha content here" in t for t in embed.seen)
    # fingerprint recorded so later runs can detect model changes
    assert store.fingerprint() == "fake:v1|noop"  # combined embeddings|contextualizer


def test_embedding_text_defaults_to_chunk_text(tmp_path):
    (tmp_path / "a.md").write_text("# A\nplain text")
    store = FakeStore()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store, NoopContextualizer()).index(
        tmp_path, ignore=[]
    )
    _, chunks, _ = store.notes["a.md"]
    assert chunks[0].context_blurb is None
    assert chunks[0].embedding_text == chunks[0].chunk.text


def test_index_removes_notes_deleted_from_disk(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    (tmp_path / "b.md").write_text("# B\nbeta content here")

    embed, store = FakeEmbed(), FakeStore()
    indexer = IndexVault(
        parser=ObsidianMarkdownParser(),
        chunker=HeadingAwareChunker(max_tokens=200),
        embeddings=embed,
        store=store,
        contextualizer=NoopContextualizer(),
    )
    indexer.index(tmp_path, ignore=[])
    assert set(store.notes) == {"a.md", "b.md"}

    # delete b.md from disk and reindex
    (tmp_path / "b.md").unlink()
    stats = indexer.index(tmp_path, ignore=[])
    assert set(store.notes) == {"a.md"}
    assert stats.notes == 1


def test_index_skips_unchanged_notes_on_reindex(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    (tmp_path / "b.md").write_text("# B\nbeta content here")
    embed, store = FakeEmbed(), FakeStore()
    indexer = IndexVault(
        ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed, store, NoopContextualizer()
    )
    indexer.index(tmp_path, ignore=[])

    # Second run, nothing changed on disk: no text should be re-embedded.
    embed.seen.clear()
    stats = indexer.index(tmp_path, ignore=[])
    assert embed.seen == []
    assert set(store.notes) == {"a.md", "b.md"}  # unchanged notes are kept, not swept
    assert stats.notes == 2


def test_index_reembeds_only_the_changed_note(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    (tmp_path / "b.md").write_text("# B\nbeta content here")
    embed, store = FakeEmbed(), FakeStore()
    indexer = IndexVault(
        ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed, store, NoopContextualizer()
    )
    indexer.index(tmp_path, ignore=[])

    (tmp_path / "a.md").write_text("# A\nalpha content CHANGED now")
    embed.seen.clear()
    indexer.index(tmp_path, ignore=[])
    assert any("CHANGED" in t for t in embed.seen)  # changed note re-embedded
    assert not any("beta" in t for t in embed.seen)  # unchanged note skipped


def test_index_reembeds_all_when_fingerprint_changes(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    embed, store = FakeEmbed(), FakeStore()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed, store, NoopContextualizer()).index(
        tmp_path, ignore=[]
    )

    # Simulate a model swap: same content, different fingerprint -> must re-embed.
    class FakeEmbed2(FakeEmbed):
        @property
        def fingerprint(self):
            return "fake:v2"

    embed2 = FakeEmbed2()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed2, store, NoopContextualizer()).index(
        tmp_path, ignore=[]
    )
    assert any("alpha" in t for t in embed2.seen)  # re-embedded despite unchanged content


def test_contextualizer_output_flows_to_store(tmp_path):
    from collections.abc import Sequence

    from ariostea.domain.models import ContextualizedChunk
    from ariostea.ports.pipeline import Contextualizer

    class TitleCtx(Contextualizer):
        def contextualize(self, note, full_doc, chunks):
            return [
                ContextualizedChunk(chunk=c, context_blurb=note.title, embedding_text=f"{note.title}\n\n{c.text}")
                for c in chunks
            ]

        @property
        def fingerprint(self):
            return "titlectx"

    (tmp_path / "a.md").write_text("# Topic\nbody text")
    store = FakeStore()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store, TitleCtx()).index(
        tmp_path, ignore=[]
    )

    _, chunks, _ = store.notes["a.md"]
    assert chunks[0].context_blurb == "Topic"
    assert chunks[0].embedding_text.startswith("Topic\n\n")
    assert store.fingerprint() == "fake:v1|titlectx"  # contextualizer in the fingerprint
