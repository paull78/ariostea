import pytest

from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.vault.filesystem import FilesystemVault
from ariostea.adapters.vault.memory import InMemoryVault
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.vault import VaultUnavailable


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
        self.fingerprints = {}

    def upsert_note(self, note, chunks, embeddings, fingerprint):
        self.notes[note.path] = (note, list(chunks), list(embeddings))
        self.fingerprints[note.path] = fingerprint

    def delete_note(self, path):
        self.notes.pop(path, None)
        self.fingerprints.pop(path, None)

    def known_notes(self):
        from ariostea.domain.models import StoredNote

        return {
            p: StoredNote(n.content_hash, self.fingerprints[p])
            for p, (n, _, _) in self.notes.items()
        }

    def stats(self):
        from ariostea.domain.models import IndexStats

        chunks = sum(len(c) for _, c, _ in self.notes.values())
        return IndexStats(len(self.notes), chunks)


def _indexer(vault, embed=None, store=None, chunker=None, contextualizer=None):
    return IndexVault(
        parser=ObsidianMarkdownParser(),
        chunker=chunker or HeadingAwareChunker(),
        embeddings=embed or FakeEmbed(),
        store=store if store is not None else FakeStore(),
        contextualizer=contextualizer or NoopContextualizer(),
        vault=vault,
    )


def _two_notes():
    return InMemoryVault({"a.md": "# A\nalpha content here", "b.md": "# B\nbeta content here"})


def test_index_vault_indexes_each_note():
    embed, store = FakeEmbed(), FakeStore()
    report = _indexer(_two_notes(), embed, store).index()

    assert set(store.notes) == {"a.md", "b.md"}
    assert report.notes == 2
    assert report.chunks == 2
    # embeddings were requested for the chunk text
    assert any("alpha content here" in t for t in embed.seen)
    # each note records the fingerprint it was embedded with
    assert store.fingerprints == {"a.md": "fake:v1|noop", "b.md": "fake:v1|noop"}


def test_embedding_text_defaults_to_chunk_text():
    store = FakeStore()
    _indexer(InMemoryVault({"a.md": "# A\nplain text"}), store=store).index()
    _, chunks, _ = store.notes["a.md"]
    assert chunks[0].context_blurb is None
    assert chunks[0].embedding_text == chunks[0].chunk.text


def test_index_removes_notes_deleted_from_disk():
    vault, store = _two_notes(), FakeStore()
    indexer = _indexer(vault, store=store)
    indexer.index()
    assert set(store.notes) == {"a.md", "b.md"}

    del vault.files["b.md"]
    report = indexer.index()
    assert set(store.notes) == {"a.md"}
    assert report.notes == 1


def test_index_skips_unchanged_notes_on_reindex():
    embed, store = FakeEmbed(), FakeStore()
    indexer = _indexer(_two_notes(), embed, store)
    indexer.index()

    # Second run, nothing changed: no text should be re-embedded.
    embed.seen.clear()
    report = indexer.index()
    assert embed.seen == []
    assert set(store.notes) == {"a.md", "b.md"}  # unchanged notes are kept, not swept
    assert report.notes == 2
    assert report.embedded == ()


def test_index_reembeds_only_the_changed_note():
    vault, embed = _two_notes(), FakeEmbed()
    indexer = _indexer(vault, embed)
    indexer.index()

    vault.files["a.md"] = "# A\nalpha content CHANGED now"
    embed.seen.clear()
    indexer.index()
    assert any("CHANGED" in t for t in embed.seen)  # changed note re-embedded
    assert not any("beta" in t for t in embed.seen)  # unchanged note skipped


def test_report_lists_embedded_and_deleted():
    vault = _two_notes()
    indexer = _indexer(vault)
    first = indexer.index()
    assert first.embedded == ("a.md", "b.md")
    assert first.deleted == ()

    vault.files["a.md"] = "# A\nedited"
    del vault.files["b.md"]
    second = indexer.index()
    assert second.embedded == ("a.md",)
    assert second.deleted == ("b.md",)
    assert second.skipped == ()


def test_index_reembeds_all_when_fingerprint_changes():
    vault, store = InMemoryVault({"a.md": "# A\nalpha content here"}), FakeStore()
    _indexer(vault, store=store).index()

    # Simulate a model swap: same content, different fingerprint -> must re-embed.
    class FakeEmbed2(FakeEmbed):
        @property
        def fingerprint(self):
            return "fake:v2"

    embed2 = FakeEmbed2()
    _indexer(vault, embed2, store).index()
    assert any("alpha" in t for t in embed2.seen)  # re-embedded despite unchanged content


def test_contextualizer_output_flows_to_store():
    from ariostea.domain.models import ContextualizedChunk
    from ariostea.ports.pipeline import Contextualizer

    class TitleCtx(Contextualizer):
        def contextualize(self, note, full_doc, chunks):
            return [
                ContextualizedChunk(
                    chunk=c, context_blurb=note.title, embedding_text=f"{note.title}\n\n{c.text}"
                )
                for c in chunks
            ]

        @property
        def fingerprint(self):
            return "titlectx"

    store = FakeStore()
    vault = InMemoryVault({"a.md": "# Topic\nbody text"})
    _indexer(vault, store=store, contextualizer=TitleCtx()).index()

    _, chunks, _ = store.notes["a.md"]
    assert chunks[0].context_blurb == "Topic"
    assert chunks[0].embedding_text.startswith("Topic\n\n")
    assert store.fingerprints["a.md"] == "fake:v1|titlectx"  # contextualizer in it


def test_index_reembeds_when_contextualizer_changes():
    from ariostea.domain.models import ContextualizedChunk
    from ariostea.ports.pipeline import Contextualizer

    class StubCtx(Contextualizer):
        def contextualize(self, note, full_doc, chunks):
            return [
                ContextualizedChunk(chunk=c, context_blurb="b", embedding_text=f"b\n\n{c.text}")
                for c in chunks
            ]

        @property
        def fingerprint(self):
            return "stub"

    vault = InMemoryVault({"a.md": "# A\nalpha content here"})
    embed, store = FakeEmbed(), FakeStore()
    # First index with Noop (combined fingerprint "fake:v1|noop").
    _indexer(vault, embed, store).index()

    # Same content, different contextualizer -> combined fingerprint changes -> re-embed.
    embed.seen.clear()
    _indexer(vault, embed, store, contextualizer=StubCtx()).index()
    assert any("alpha" in t for t in embed.seen)  # re-embedded despite unchanged content
    assert store.fingerprints["a.md"] == "fake:v1|stub"


def test_the_default_chunker_leaves_the_stored_fingerprint_as_it_was():
    # An index built before chunking was configurable stored "emb|ctx". The
    # default policy must reproduce that string exactly, or every existing
    # vault is forced into a full reindex on upgrade for no reason.
    store = FakeStore()
    _indexer(InMemoryVault({"a.md": "# A\nalpha"}), store=store).index()
    assert store.fingerprints["a.md"] == "fake:v1|noop"


def test_a_chunking_change_rechunks_unchanged_notes():
    # Same content, new chunking policy: the stored chunks are stale and must
    # be rebuilt, even though no note's hash changed.
    vault = InMemoryVault({"a.md": "# A\n" + " ".join(f"w{i}" for i in range(40))})
    embed, store = FakeEmbed(), FakeStore()
    _indexer(vault, embed, store).index()
    assert len(store.notes["a.md"][1]) == 1

    _indexer(vault, embed, store, chunker=HeadingAwareChunker(max_tokens=10)).index()
    assert len(store.notes["a.md"][1]) > 1
    assert "chunk:heading_aware:10" in store.fingerprints["a.md"]


# --- guards: a vault that cannot be trusted never empties the index (AC-01, AC-02)


def test_missing_vault_raises_and_keeps_the_index():
    vault, store = _two_notes(), FakeStore()
    indexer = _indexer(vault, store=store)
    indexer.index()
    before = dict(store.notes)

    vault.available = False
    with pytest.raises(VaultUnavailable):
        indexer.index()
    assert store.notes == before


def test_empty_vault_with_notes_in_index_raises_and_keeps_the_index():
    vault, store = _two_notes(), FakeStore()
    indexer = _indexer(vault, store=store)
    indexer.index()
    before = dict(store.notes)

    vault.files.clear()
    with pytest.raises(VaultUnavailable) as err:
        indexer.index()
    assert "2" in str(err.value)  # says how many notes the index holds
    assert store.notes == before


def test_empty_vault_on_empty_index_indexes_zero():
    store = FakeStore()
    report = _indexer(InMemoryVault({}), store=store).index()
    assert report.notes == 0


# --- unreadable notes are skipped, named, and keep what they had (AC-04)


def test_unreadable_note_is_skipped_and_reported():
    vault = InMemoryVault(
        {"a.md": "# A\nalpha", "bad.md": "caff\xe8".encode("latin-1"), "c.md": "# C\ngamma"}
    )
    store = FakeStore()
    report = _indexer(vault, store=store).index()

    assert set(store.notes) == {"a.md", "c.md"}
    assert [path for path, _ in report.skipped] == ["bad.md"]
    assert "UTF-8" in report.skipped[0][1]


def test_unreadable_note_keeps_its_earlier_chunks():
    vault, store = _two_notes(), FakeStore()
    indexer = _indexer(vault, store=store)
    indexer.index()
    before = store.notes["b.md"]

    vault.files["b.md"] = b"\xff\xfe broken"
    report = indexer.index()
    assert store.notes["b.md"] == before
    assert [path for path, _ in report.skipped] == ["b.md"]
    assert "b.md" not in report.deleted


# --- the vault's rules reach the index (AC-05, AC-09)


def test_bom_frontmatter_is_not_indexed():
    store = FakeStore()
    vault = InMemoryVault({"n.md": "﻿---\ntitle: Hidden\ntags: [x]\n---\nbody text\n"})
    _indexer(vault, store=store).index()

    note, chunks, _ = store.notes["n.md"]
    assert note.title == "n"
    assert note.frontmatter["title"] == "Hidden"  # read as frontmatter, not body
    assert all("title:" not in c.chunk.text for c in chunks)


def test_excluded_notes_leave_the_index():
    # An index built before the dot rule holds a note from .trash.
    store = FakeStore()
    _indexer(InMemoryVault({"x.md": "# X\nold"}), store=store).index()
    store.notes[".trash/x.md"] = store.notes.pop("x.md")
    store.fingerprints[".trash/x.md"] = store.fingerprints.pop("x.md")

    report = _indexer(_two_notes(), store=store).index()
    assert ".trash/x.md" not in store.notes
    assert report.deleted == (".trash/x.md",)


# --- real files and the real store


class _SizedEmbed:
    """A fake embedder of a given vector size and fingerprint, standing in
    for a real model."""

    def __init__(self, dim, fingerprint):
        self._dim, self._fingerprint = dim, fingerprint

    def embed_documents(self, texts):
        return [[float(len(t))] + [0.0] * (self._dim - 1) for t in texts]

    def embed_query(self, text):
        return [float(len(text))] + [0.0] * (self._dim - 1)

    @property
    def dimension(self):
        return self._dim

    @property
    def fingerprint(self):
        return self._fingerprint


def test_reindexing_with_a_model_of_another_size_replaces_every_vector(tmp_path):
    # The upgrade path for every existing vault when the default model changes
    # size: no error, and no vector of the old size left behind
    # (bge-m3-default.01).
    from ariostea.adapters.store.sqlite_store import SqliteStore

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nThe double bass has four strings.\n")
    (vault / "b.md").write_text("# B\n\nThe viola is tuned in fifths.\n")
    db = str(tmp_path / "idx.db")

    def index_with(embed):
        store = SqliteStore(path=db, dim=embed.dimension)
        _indexer(FilesystemVault(vault), embed, store).index()
        return store

    first = index_with(_SizedEmbed(2, "fake:v1"))
    chunks_before = first.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    first.db.close()

    second = index_with(_SizedEmbed(3, "fake:v2"))
    sizes = {row[0] for row in second.db.execute("SELECT vec_length(embedding) FROM chunks_vec")}
    assert sizes == {3}
    assert second.db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == chunks_before
    assert second.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == chunks_before
    assert all("fake:v2" in s.fingerprint for s in second.known_notes().values())


def test_reindexing_an_old_index_with_a_new_model_succeeds(tmp_path, old_index):
    # The owner's real index failed here at gate M2: built before the blurb
    # column, the FTS table and the meta table existed (bge-m3-default.09).
    from ariostea.adapters.store.sqlite_store import SqliteStore

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nThe double bass has four strings.\n")
    db = str(tmp_path / "old.db")
    old_index(db, dim=2)

    store = SqliteStore(path=db, dim=3)
    _indexer(FilesystemVault(vault), _SizedEmbed(3, "fake:v2"), store).index()

    sizes = {row[0] for row in store.db.execute("SELECT vec_length(embedding) FROM chunks_vec")}
    assert sizes == {3}
    assert [h.chunk.note_path for h in store.sparse("strings", k=5)] == ["a.md"]


def test_a_note_emptied_of_chunks_leaves_the_index():
    vault, store = _two_notes(), FakeStore()
    indexer = _indexer(vault, store=store)
    indexer.index()

    vault.files["b.md"] = ""
    report = indexer.index()
    assert "b.md" not in store.notes
    assert report.deleted == ("b.md",)


# --- progress: one plain value per listed note (AC-16)


def test_progress_reports_every_listed_note():
    from ariostea.domain.models import Progress

    vault = InMemoryVault({"a.md": "# A\nalpha", "b.md": "# B\nbeta"})
    indexer = _indexer(vault)
    indexer.index()
    vault.files.update({"c.md": "# C\ngamma", "d.md": "# D\ndelta", "e.md": b"\xff"})

    seen = []
    indexer.index(progress=seen.append)
    assert [p.done for p in seen] == [1, 2, 3, 4, 5]
    assert {p.total for p in seen} == {5}
    assert seen[-1] == Progress(done=5, total=5, embedded=2, skipped=1)


def test_no_progress_callback_is_fine():
    assert _indexer(_two_notes()).index().notes == 2


# --- an interrupted run resumes (AC-07, AC-08, AC-09)


class FailingEmbed(FakeEmbed):
    """Embeds `allowed` notes, then raises: a run stopped partway."""

    def __init__(self, allowed, fingerprint="fake:v1"):
        super().__init__()
        self.allowed, self._fingerprint = allowed, fingerprint

    def embed_documents(self, texts):
        if self.allowed == 0:
            raise RuntimeError("stopped")
        self.allowed -= 1
        return super().embed_documents(texts)

    @property
    def fingerprint(self):
        return self._fingerprint


class CountingEmbed(FakeEmbed):
    def __init__(self, fingerprint="fake:v1"):
        super().__init__()
        self.calls, self._fingerprint = 0, fingerprint

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)

    @property
    def fingerprint(self):
        return self._fingerprint


def _thirty_notes():
    return InMemoryVault({f"n{i:02}.md": f"# N{i}\nnote number {i}" for i in range(30)})


def test_interrupted_first_index_resumes():  # AC-07
    vault, store = _thirty_notes(), FakeStore()
    with pytest.raises(RuntimeError):
        _indexer(vault, FailingEmbed(allowed=10), store).index()
    assert len(store.notes) == 10

    embed = CountingEmbed()
    report = _indexer(vault, embed, store).index()
    assert embed.calls == 20 and len(report.embedded) == 20
    assert len(store.notes) == 30


def test_interrupted_change_of_settings_resumes():  # AC-08
    vault, store = _thirty_notes(), FakeStore()
    _indexer(vault, CountingEmbed("fake:v1"), store).index()
    with pytest.raises(RuntimeError):
        _indexer(vault, FailingEmbed(allowed=10, fingerprint="fake:v2"), store).index()
    assert sorted(store.fingerprints.values()).count("fake:v2|noop") == 10

    embed = CountingEmbed("fake:v2")
    _indexer(vault, embed, store).index()
    assert embed.calls == 20
    assert set(store.fingerprints.values()) == {"fake:v2|noop"}


def test_unchanged_index_embeds_nothing_after_migration(tmp_path, previous_release_index):
    # AC-09: an index from the previous release, with the settings it was built with.
    import hashlib

    from ariostea.adapters.store.sqlite_store import SqliteStore

    vault = _two_notes()
    hashes = {p: hashlib.sha256(text.encode()).hexdigest() for p, text in vault.files.items()}
    path = str(tmp_path / "idx.db")
    previous_release_index(path, 2, hashes, "fake:v1|noop")

    embed = CountingEmbed()
    report = _indexer(vault, embed, SqliteStore(path=path, dim=2)).index()
    assert embed.calls == 0 and report.embedded == ()
