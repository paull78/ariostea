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


def test_sparse_bm25_ranks_keyword_matches(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    chunks = [
        _cchunk(note, 0, "the quick brown fox jumps"),
        _cchunk(note, 1, "lorem ipsum dolor sit amet"),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert_note(note, chunks, embeddings)

    hits = store.sparse("fox", k=5)
    assert hits[0].chunk.text == "the quick brown fox jumps"
    assert hits[0].sparse_rank == 0
    assert hits[0].dense_rank is None
    assert hits[0].score > 0.0


def test_sparse_matches_accented_terms_in_main_languages(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    chunks = [
        _cchunk(note, 0, "la grande citta di Roma"),  # Italian, indexed without accent
        _cchunk(note, 1, "hasta manana amigo"),  # Spanish
        _cchunk(note, 2, "Herr Muller wohnt hier"),  # German
        _cchunk(note, 3, "pedido de informacao urgente"),  # Portuguese
    ]
    store.upsert_note(note, chunks, [[1.0, 0.0, 0.0]] * 4)

    # Accented queries must fold to the same token as the unaccented stored text.
    assert store.sparse("città", k=5)[0].chunk.ordinal == 0
    assert store.sparse("mañana", k=5)[0].chunk.ordinal == 1
    assert store.sparse("Müller", k=5)[0].chunk.ordinal == 2
    assert store.sparse("informação", k=5)[0].chunk.ordinal == 3


def test_sparse_returns_empty_when_no_term_matches(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "alpha beta gamma")], [[1.0, 0.0, 0.0]])
    assert store.sparse("zebra", k=5) == []


def test_sparse_sanitizes_punctuation_and_empty_queries(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "alpha beta gamma")], [[1.0, 0.0, 0.0]])
    # punctuation around a real term must not raise and must still match
    assert store.sparse("  beta?! ", k=5)[0].chunk.text == "alpha beta gamma"
    # a query with no word characters yields no results (and no SQL error)
    assert store.sparse("?? -- ::", k=5) == []


def test_delete_note_removes_from_fts(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "findable keyword")], [[1.0, 0.0, 0.0]])
    assert store.sparse("findable", k=5)  # present before delete
    store.delete_note("a.md")
    assert store.sparse("findable", k=5) == []


def test_reupsert_does_not_duplicate_fts_rows(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "unique token")], [[1.0, 0.0, 0.0]])
    store.upsert_note(note, [_cchunk(note, 0, "unique token")], [[1.0, 0.0, 0.0]])
    hits = store.sparse("unique", k=10)
    assert len(hits) == 1  # old FTS row was cleaned, not duplicated


def test_note_titles_batch_lookup(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    store.upsert_note(_note("a.md"), [_cchunk(_note("a.md"), 0, "x")], [[1.0, 0.0, 0.0]])
    store.upsert_note(_note("b.md"), [_cchunk(_note("b.md"), 0, "y")], [[0.0, 1.0, 0.0]])
    titles = store.note_titles(["a.md", "b.md", "missing.md"])
    assert titles == {"a.md": "A", "b.md": "A"}  # _note() always titles "A"
    assert store.note_titles([]) == {}


def test_read_note_reconstructs_in_ordinal_order(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note("a.md")
    chunks = [_cchunk(note, 0, "first part"), _cchunk(note, 1, "second part")]
    store.upsert_note(note, chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    doc = store.read_note("a.md")
    assert doc is not None
    assert doc.note_path == "a.md"
    assert doc.title == "A"
    assert doc.text == "first part\n\nsecond part"


def test_read_note_returns_none_when_absent(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    assert store.read_note("nope.md") is None


def test_upsert_persists_context_blurb(tmp_path):
    from ariostea.domain.models import Chunk, ContextualizedChunk

    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    chunk = Chunk(note_path=note.path, ordinal=0, heading_path=("A",), text="bare", token_count=1)
    cc = ContextualizedChunk(chunk=chunk, context_blurb="the blurb", embedding_text="the blurb\n\nbare")
    store.upsert_note(note, [cc], [[1.0, 0.0, 0.0]])

    rows = store.db.execute("SELECT context_blurb FROM chunks").fetchall()
    assert rows[0]["context_blurb"] == "the blurb"
