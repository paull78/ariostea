from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.domain.models import ContextualizedChunk
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.pipeline import Contextualizer


class FakeEmbed:
    """Sparse-channel test: vectors are irrelevant, so return constant dummies."""

    def embed_documents(self, texts):
        return [[0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0]

    @property
    def dimension(self):
        return 2

    @property
    def fingerprint(self):
        return "fake:v1"


class TitleStubContextualizer(Contextualizer):
    """Deterministic stand-in for the LLM: prepends the note title as the blurb."""

    def contextualize(self, note, full_doc, chunks):
        return [
            ContextualizedChunk(
                chunk=c, context_blurb=note.title, embedding_text=f"{note.title}\n\n{c.text}"
            )
            for c in chunks
        ]

    @property
    def fingerprint(self):
        return "stub"


def _index(vault, db_path, contextualizer):
    store = SqliteStore(path=str(db_path), dim=2)
    IndexVault(
        ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store, contextualizer
    ).index(vault, ignore=[])
    return store


def test_contextualization_makes_an_ambiguous_chunk_findable(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # No H1, so the title comes from the filename ("piano"); the body never says "piano".
    (vault / "piano.md").write_text("It has 88 of them, black and white, struck by felt hammers.")

    # Without context: the bare chunk has no "piano" token -> sparse keyword search misses.
    plain = _index(vault, tmp_path / "plain.db", NoopContextualizer())
    assert plain.sparse("piano", k=5) == []

    # With note-level context (title prepended): "piano" is now indexed -> found.
    ctx = _index(vault, tmp_path / "ctx.db", TitleStubContextualizer())
    hits = ctx.sparse("piano", k=5)
    assert [h.chunk.note_path for h in hits] == ["piano.md"]
