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
    payload = status_payload(FakeAdmin())
    assert payload == {
        "notes": 3,
        "chunks": 12,
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


def test_get_note_payload_found_and_missing():
    from types import SimpleNamespace

    from ariostea.domain.models import NoteDocument
    from ariostea.mcp.handlers import get_note_payload

    class FakeReader:
        def __init__(self, doc):
            self._doc = doc

        def read_note(self, path):
            return self._doc

    doc = NoteDocument(note_path="a.md", title="A", text="hello body")
    found = get_note_payload(SimpleNamespace(reader=FakeReader(doc)), path="a.md")
    assert found == {"found": True, "note_path": "a.md", "title": "A", "text": "hello body"}

    missing = get_note_payload(SimpleNamespace(reader=FakeReader(None)), path="x.md")
    assert missing == {"found": False, "note_path": "x.md"}
