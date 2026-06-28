from ariostea.indexing.watch_vault import WatchVault


class FakeIndexer:
    def __init__(self):
        self.calls = []

    def index(self, root, ignore=()):
        self.calls.append((str(root), tuple(ignore)))


def test_watch_indexes_once_initially_then_per_change_batch():
    idx = FakeIndexer()

    def fake_watch(root, stop_event=None):
        yield {("modified", "a.md")}
        yield {("modified", "b.md")}

    WatchVault(idx, "/vault", ignore=[".obsidian/"], watch_fn=fake_watch).run()

    # 1 initial full index + 1 per change batch = 3
    assert len(idx.calls) == 3
    # every call targets the configured root + ignore
    assert all(call == ("/vault", (".obsidian/",)) for call in idx.calls)


def test_watch_passes_stop_event_to_watch_fn():
    idx = FakeIndexer()
    received = {}

    def fake_watch(root, stop_event=None):
        received["stop_event"] = stop_event
        return iter(())  # no change batches

    sentinel = object()
    WatchVault(idx, "/vault", ignore=[], watch_fn=fake_watch).run(stop_event=sentinel)

    assert received["stop_event"] is sentinel
    assert len(idx.calls) == 1  # only the initial index
