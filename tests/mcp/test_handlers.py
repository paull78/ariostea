from datetime import datetime, timezone
from types import SimpleNamespace

from fixtures.runs import NOW, MemoryRunLog

from ariostea.domain.models import IndexReport, IndexStats, RunRecord
from ariostea.mcp.handlers import status_payload


class FakeAdmin:
    def known_notes(self):
        return {}

    def stats(self):
        return IndexStats(notes=3, chunks=12)


def _status_after(*records):
    log = MemoryRunLog()
    for record in records:
        log.record_run(record)
    from fixtures.runs import index_runs

    container = SimpleNamespace(admin=FakeAdmin(), runs=index_runs(None, log=log))
    return status_payload(container)


EARLIER = RunRecord(datetime(2026, 10, 26, 18, 0, 0, tzinfo=timezone.utc), "fp-old", 2, 6, 0, None)


def test_status_on_a_new_index():
    assert _status_after() == {
        "notes": 3,
        "chunks": 12,
        "running": False,
        "last_indexed": None,
        "config_fingerprint": None,
        "last_run": None,
    }


def test_status_after_a_completed_run():
    payload = _status_after(RunRecord(NOW, "fp", 3, 12, 1, None))
    assert payload["last_indexed"] == "2026-10-27T09:41:05Z"
    assert payload["config_fingerprint"] == "fp"
    assert payload["last_run"] == {
        "finished_at": "2026-10-27T09:41:05Z",
        "notes": 3,
        "chunks": 12,
        "skipped": 1,
        "error": None,
    }


def test_status_after_a_failed_run():  # AC-02
    payload = _status_after(EARLIER, RunRecord(NOW, "fp", 2, 6, 0, "vault /v is unavailable"))
    assert payload["last_run"]["error"] == "vault /v is unavailable"
    assert payload["last_run"]["finished_at"] == "2026-10-27T09:41:05Z"
    assert payload["last_indexed"] == "2026-10-26T18:00:00Z"
    assert payload["config_fingerprint"] == "fp-old"


def test_status_is_stable_between_runs():  # AC-10, third part
    log = MemoryRunLog()
    log.record_run(RunRecord(NOW, "fp", 3, 12, 0, None))
    from fixtures.runs import index_runs

    container = SimpleNamespace(admin=FakeAdmin(), runs=index_runs(None, log=log))
    assert status_payload(container)["last_indexed"] == status_payload(container)["last_indexed"]


def test_status_reports_a_run_of_this_process_through_index_runs():
    class Indexer:
        fingerprint = "fp-now"

        def index(self, progress=None):
            return IndexReport(3, 12, ("a.md",), (), (("bad.md", "not UTF-8 text"),))

    from fixtures.runs import index_runs

    runs = index_runs(Indexer())
    runs.begin(wait=False).execute()
    payload = status_payload(SimpleNamespace(admin=FakeAdmin(), runs=runs))
    assert payload["config_fingerprint"] == "fp-now"
    assert payload["last_run"]["skipped"] == 1


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
    fingerprint = "fp"

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
        fingerprint = "fp"

        def index(self, progress=None):
            calls.append(progress)
            return IndexReport(0, 0, (), (), ())

    reindex_payload(SimpleNamespace(runs=_runs(Indexer())))
    assert calls == [None]


def _runs(indexer):
    """A real IndexRuns on the in-memory lock, around a fake indexer."""
    from fixtures.runs import index_runs

    return index_runs(indexer)
