from ariostea.indexing.watch_vault import WatchVault


class FakeIndexer:
    def __init__(self):
        self.calls = []

    def index(self):
        self.calls.append("index")


def test_watch_indexes_once_initially_then_per_change_batch():
    idx = FakeIndexer()

    def fake_watch(root, stop_event=None):
        yield {("modified", "a.md")}
        yield {("modified", "b.md")}

    WatchVault(idx, "/vault", watch_fn=fake_watch).run()

    # 1 initial full index + 1 per change batch = 3
    assert len(idx.calls) == 3


def test_watch_passes_stop_event_to_watch_fn():
    idx = FakeIndexer()
    received = {}

    def fake_watch(root, stop_event=None):
        received["stop_event"] = stop_event
        return iter(())  # no change batches

    sentinel = object()
    WatchVault(idx, "/vault", watch_fn=fake_watch).run(stop_event=sentinel)

    assert received["stop_event"] is sentinel
    assert len(idx.calls) == 1  # only the initial index
