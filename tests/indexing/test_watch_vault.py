"""WatchVault against a scripted in-memory vault and the real IndexVault:
each scripted change edits the vault, then the watcher runs a pass."""

import pytest

from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.vault.memory import InMemoryVault
from ariostea.domain.models import IndexStats
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
        self.notes, self._fp = {}, ""

    def upsert_note(self, note, chunks, embeddings):
        self.notes[note.path] = (note, list(chunks))

    def delete_note(self, path):
        self.notes.pop(path, None)

    def known_hashes(self):
        return {p: n.content_hash for p, (n, _) in self.notes.items()}

    def stats(self):
        return IndexStats(len(self.notes), sum(len(c) for _, c in self.notes.values()), 0.0, "")

    def fingerprint(self):
        return self._fp

    def set_fingerprint(self, value):
        self._fp = value


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
    indexer = IndexVault(
        ObsidianMarkdownParser(),
        HeadingAwareChunker(),
        FakeEmbed(),
        store,
        NoopContextualizer(),
        vault=vault,
    )
    reports, warnings = [], []
    WatchVault(indexer, vault).run(on_report=reports.append, on_warning=warnings.append)
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
    indexer = IndexVault(
        ObsidianMarkdownParser(),
        HeadingAwareChunker(),
        FakeEmbed(),
        store,
        NoopContextualizer(),
        vault=vault,
    )
    WatchVault(indexer, vault).run(stop_event=stop)
    assert set(store.notes) == {"a.md"}  # the initial pass only


def test_progress_goes_to_the_first_pass_only():
    vault = InMemoryVault({"a.md": "# A\nalpha"}, script=[edit({"b.md": "# B\nbeta"})])
    indexer = IndexVault(
        ObsidianMarkdownParser(),
        HeadingAwareChunker(),
        FakeEmbed(),
        FakeStore(),
        NoopContextualizer(),
        vault=vault,
    )
    seen = []
    WatchVault(indexer, vault).run(progress=seen.append)
    assert [(p.done, p.total) for p in seen] == [(1, 1)]
