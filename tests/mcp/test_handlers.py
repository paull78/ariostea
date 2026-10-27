from ariostea.domain.models import IndexStats
from ariostea.mcp.handlers import status_payload


class FakeAdmin:
    def known_hashes(self):
        return {}

    def stats(self):
        return IndexStats(notes=3, chunks=12, last_indexed=42.0, config_fingerprint="fp")

    def fingerprint(self):
        return "fp"

    def set_fingerprint(self, value):
        pass


def test_status_payload_reports_counts():
    from types import SimpleNamespace

    container = SimpleNamespace(admin=FakeAdmin(), runs=_runs(None))
    payload = status_payload(container)
    assert payload == {
        "notes": 3,
        "chunks": 12,
        "running": False,
        "last_indexed": 42.0,
        "config_fingerprint": "fp",
    }


def test_search_sources_payload_shapes_hits():
    from types import SimpleNamespace

    from ariostea.domain.models import SourceHit
    from ariostea.mcp.handlers import search_sources_payload

    class FakeSources:
        def search(self, query):
            return [
                SourceHit(
                    note_path="a.md",
                    title="A",
                    hit_count=2,
                    best_score=0.9,
                    snippets=("s1", "s2"),
                )
            ]

    container = SimpleNamespace(sources=FakeSources())
    payload = search_sources_payload(container, query="x", k=5)
    assert payload["sources"][0] == {
        "note_path": "a.md",
        "title": "A",
        "hit_count": 2,
        "best_score": 0.9,
        "snippets": ["s1", "s2"],
    }


def test_get_note_payload_shape():
    from types import SimpleNamespace

    from ariostea.domain.models import NoteMissing, NoteText
    from ariostea.mcp.handlers import get_note_payload

    class FakeNotes:
        def __init__(self, result):
            self.result = result

        def get(self, path):
            return self.result

    found = get_note_payload(
        SimpleNamespace(notes=FakeNotes(NoteText("a.md", "A", "# A\n\nbody\n"))), path="a.md"
    )
    assert found == {"found": True, "note_path": "a.md", "title": "A", "text": "# A\n\nbody\n"}

    missing = get_note_payload(SimpleNamespace(notes=FakeNotes(NoteMissing("x.md", None))), "x.md")
    assert missing == {"found": False, "note_path": "x.md"}

    gone = get_note_payload(
        SimpleNamespace(notes=FakeNotes(NoteMissing("g.md", "no such file"))), "g.md"
    )
    assert gone == {"found": False, "note_path": "g.md", "reason": "no such file"}


class _FakeIndexer:
    def __init__(self, result):
        self._result = result

    def index(self, progress=None):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_reindex_payload_lists_skipped():
    from types import SimpleNamespace

    from ariostea.domain.models import IndexReport
    from ariostea.mcp.handlers import reindex_payload

    report = IndexReport(
        notes=2, chunks=5, embedded=("a.md",), deleted=(), skipped=(("bad.md", "not UTF-8 text"),)
    )
    payload = reindex_payload(SimpleNamespace(runs=_runs(_FakeIndexer(report))))
    assert payload == {
        "notes": 2,
        "chunks": 5,
        "skipped": [{"path": "bad.md", "reason": "not UTF-8 text"}],
    }


def test_reindex_payload_returns_error_on_unavailable_vault():
    from types import SimpleNamespace

    from ariostea.mcp.handlers import reindex_payload
    from ariostea.ports.vault import VaultUnavailable

    error = VaultUnavailable("/gone", "no such folder")
    payload = reindex_payload(SimpleNamespace(runs=_runs(_FakeIndexer(error))))
    assert payload == {"error": "vault /gone is unavailable: no such folder"}


def test_reindex_payload_without_progress_passes_none():
    from types import SimpleNamespace

    from ariostea.domain.models import IndexReport
    from ariostea.mcp.handlers import reindex_payload

    calls = []

    class Indexer:
        def index(self, progress=None):
            calls.append(progress)
            return IndexReport(0, 0, (), (), ())

    reindex_payload(SimpleNamespace(runs=_runs(Indexer())))
    assert calls == [None]


def _runs(indexer):
    """A real IndexRuns on the in-memory lock, around a fake indexer."""
    from ariostea.adapters.lock.memory import InMemoryRunLock
    from ariostea.indexing.index_runs import IndexRuns

    return IndexRuns(indexer, InMemoryRunLock())
