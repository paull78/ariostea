from ariostea.mcp.handlers import status_payload
from ariostea.domain.models import IndexStats


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
