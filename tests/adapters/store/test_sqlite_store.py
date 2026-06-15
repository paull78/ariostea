from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.domain.models import Chunk, ContextualizedChunk, Note


def _note(path="a.md"):
    return Note(
        path=path, title="A", frontmatter={}, tags=(), wikilinks=(), content_hash="h1", mtime=1.0
    )


def _cchunk(note, ordinal, text):
    chunk = Chunk(
        note_path=note.path,
        ordinal=ordinal,
        heading_path=("A",),
        text=text,
        token_count=len(text.split()),
    )
    return ContextualizedChunk(chunk=chunk, context_blurb=None, embedding_text=text)


def test_upsert_then_dense_retrieves_nearest(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    chunks = [_cchunk(note, 0, "alpha"), _cchunk(note, 1, "beta")]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert_note(note, chunks, embeddings)

    hits = store.dense([0.9, 0.1, 0.0], k=2)
    assert hits[0].chunk.text == "alpha"  # nearest to [1,0,0]
    assert hits[0].chunk.note_path == "a.md"
    assert len(hits) == 2


def test_upsert_replaces_previous_chunks(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "old")], [[1.0, 0.0, 0.0]])
    store.upsert_note(note, [_cchunk(note, 0, "new")], [[1.0, 0.0, 0.0]])
    hits = store.dense([1.0, 0.0, 0.0], k=5)
    assert [h.chunk.text for h in hits] == ["new"]  # old chunk gone


def test_admin_reports_hashes_and_stats(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "x")], [[1.0, 0.0, 0.0]])
    store.set_fingerprint("fp-1")

    assert store.known_hashes() == {"a.md": "h1"}
    stats = store.stats()
    assert stats.notes == 1 and stats.chunks == 1 and stats.config_fingerprint == "fp-1"


def test_delete_note_removes_everything(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "x")], [[1.0, 0.0, 0.0]])
    store.delete_note("a.md")
    assert store.known_hashes() == {}
    assert store.dense([1.0, 0.0, 0.0], k=5) == []
