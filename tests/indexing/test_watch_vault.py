"""WatchVault against a scripted in-memory vault and the real IndexVault:
each scripted change edits the vault, then the watcher runs a pass."""

import pytest

from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.adapters.lock.memory import InMemoryRunLock
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.vault.memory import InMemoryVault
from ariostea.domain.models import IndexStats, StoredNote
from ariostea.indexing.index_runs import IndexRuns
from ariostea.indexing.index_vault import IndexVault
from ariostea.indexing.watch_vault import WatchVault
from ariostea.ports.vault import VaultUnavailable


class FakeEmbed:
    def embed_documents(self, texts):
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

    def upsert_note(self, note, chunks, embeddings, fingerprint):
        self.notes[note.path] = (note, list(chunks), fingerprint)

    def delete_note(self, path):
        self.notes.pop(path, None)

    def known_notes(self):
        return {p: StoredNote(n.content_hash, fp) for p, (n, _, fp) in self.notes.items()}

    def stats(self):
        return IndexStats(len(self.notes), sum(len(c) for _, c, _ in self.notes.values()), 0.0, "")


def _watcher(vault, store, lock=None):
    indexer = IndexVault(
        ObsidianMarkdownParser(),
        HeadingAwareChunker(),
        FakeEmbed(),
        store,
        NoopContextualizer(),
        vault=vault,
    )
    return WatchVault(IndexRuns(indexer, lock or InMemoryRunLock()), vault)


def edit(files):
    """A scripted change that writes `files` into the vault."""

    def change(vault):
        vault.files.update(files)
        return list(files)

    return change


def delete(path):
    def change(vault):
        del vault.files[path]
        return [path]

    return change


def available(flag):
    def change(vault):
        vault.available = flag
        return ["."]

    return change


def run(vault, store=None):
    store = store if store is not None else FakeStore()
    watcher = _watcher(vault, store)
    reports, warnings = [], []
    watcher.run(on_report=reports.append, on_warning=warnings.append)
    return store, reports, warnings


def test_one_pass_at_start_then_one_per_batch():
    vault = InMemoryVault(
        {"a.md": "# A\nalpha"},
        script=[edit({"b.md": "# B\nbeta"}), edit({".obsidian/workspace.json": "{}"})],
    )
    store, reports, warnings = run(vault)
    # The .obsidian change yields no batch, so no pass (AC-10).
    assert [r.embedded for r in reports] == [("a.md",), ("b.md",)]
    assert warnings == []
    assert set(store.notes) == {"a.md", "b.md"}


def test_report_per_pass():  # AC-17
    vault = InMemoryVault(
        {"a.md": "# A\nalpha", "b.md": "# B\nbeta"},
        script=[
            lambda v: edit({"a.md": "# A\nchanged"})(v) + delete("b.md")(v),
            edit({"img.png": "x"}),
        ],
    )
    _, reports, _ = run(vault)
    edit_pass, attachment_pass = reports[1], reports[2]
    assert (edit_pass.embedded, edit_pass.deleted) == (("a.md",), ("b.md",))
    assert (attachment_pass.embedded, attachment_pass.deleted) == ((), ())


def test_guard_failure_at_start_propagates():  # AC-03
    vault = InMemoryVault({"a.md": "# A\nalpha"}, available=False, script=[edit({"b.md": "x"})])
    with pytest.raises(VaultUnavailable):
        run(vault)


def test_guard_failure_after_start_warns_and_continues():  # AC-03
    store, held = FakeStore(), []

    def note_what_the_index_holds(vault):
        held.append(set(store.notes))
        return ["."]

    vault = InMemoryVault(
        {"a.md": "# A\nalpha", "b.md": "# B\nbeta"},
        script=[
            available(False),
            note_what_the_index_holds,
            available(True),
            edit({"c.md": "# C\ngamma"}),
        ],
    )
    _, reports, warnings = run(vault, store)

    # The vault vanished for two passes: one warning each, the index untouched.
    assert len(warnings) == 2
    assert all("unavailable" in w for w in warnings)
    assert held == [{"a.md", "b.md"}]
    # Restored: the next passes index normally.
    assert set(store.notes) == {"a.md", "b.md", "c.md"}
    assert reports[-1].embedded == ("c.md",)


def test_new_unreadable_note_warns_and_watch_continues():  # AC-06
    vault = InMemoryVault(
        {"a.md": "# A\nalpha"},
        script=[edit({"bad.md": b"caff\xe8"}), edit({"c.md": "# C\ngamma"})],
    )
    store, reports, warnings = run(vault)
    assert warnings == ["skipped bad.md: not UTF-8 text"]
    assert "c.md" in store.notes


def test_unreadable_note_is_warned_once():
    vault = InMemoryVault(
        {"a.md": "# A\nalpha", "bad.md": b"\xff\xfe"},
        script=[
            edit({"c.md": "# C\ngamma"}),
            edit({"d.md": "# D\ndelta"}),
            edit({"bad.md": "# Bad\nfixed"}),
            edit({"bad.md": b"\xff\xfe again"}),
        ],
    )
    _, _, warnings = run(vault)
    # Warned at start, silent while still bad, warned again once it breaks anew.
    assert warnings == ["skipped bad.md: not UTF-8 text"] * 2


def test_stop_event_ends_the_watch():
    import threading

    stop = threading.Event()
    stop.set()
    vault = InMemoryVault({"a.md": "# A\nalpha"}, script=[edit({"b.md": "x"})])
    store = FakeStore()
    watcher = _watcher(vault, store)
    watcher.run(stop_event=stop)
    assert set(store.notes) == {"a.md"}  # the initial pass only


def test_progress_goes_to_the_first_pass_only():
    vault = InMemoryVault({"a.md": "# A\nalpha"}, script=[edit({"b.md": "# B\nbeta"})])
    watcher = _watcher(vault, FakeStore())
    seen = []
    watcher.run(progress=seen.append)
    assert [(p.done, p.total) for p in seen] == [(1, 1)]


def test_idle_watch_holds_no_lock():  # AC-06, first half
    lock = InMemoryRunLock()
    free_while_idle = []

    def look_at_the_lock(vault):
        other = lock.handle()
        free_while_idle.append(other.try_acquire())
        other.release()
        return ["a.md"]

    vault = InMemoryVault({"a.md": "# A\nalpha"}, script=[look_at_the_lock])
    _watcher(vault, FakeStore(), lock).run()
    assert free_while_idle == [True]


def _held_for_a_moment(lock, events):
    """Another process's run: take the lock now, release it 0.2 s later."""
    import threading

    other = lock.handle()
    assert other.try_acquire()

    def release():
        events.append("released")
        other.release()

    threading.Timer(0.2, release).start()


def test_pass_waits_for_a_run_in_progress():  # AC-06, second half
    lock, events = InMemoryRunLock(), []

    def edit_during_another_run(vault):
        _held_for_a_moment(lock, events)
        return edit({"b.md": "# B\nbeta"})(vault)

    vault = InMemoryVault({"a.md": "# A\nalpha"}, script=[edit_during_another_run])
    _watcher(vault, FakeStore(), lock).run(
        on_wait=lambda: events.append("waiting"),
        on_report=lambda r: events.append(r.embedded),
    )
    assert events == [("a.md",), "waiting", "released", ("b.md",)]


def test_first_pass_waits_too():
    lock, events = InMemoryRunLock(), []
    _held_for_a_moment(lock, events)
    vault = InMemoryVault({"a.md": "# A\nalpha"})
    _watcher(vault, FakeStore(), lock).run(
        on_wait=lambda: events.append("waiting"),
        on_report=lambda r: events.append(r.embedded),
    )
    assert events == ["waiting", "released", ("a.md",)]
