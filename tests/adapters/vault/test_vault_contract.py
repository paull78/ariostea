"""One contract for every Vault adapter: the real filesystem one and the
in-memory test double must agree on what a note is and how it reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from ariostea.adapters.vault.filesystem import FilesystemVault
from ariostea.adapters.vault.memory import InMemoryVault
from ariostea.ports.vault import NoteUnreadable, Vault


def _filesystem(tmp_path: Path, files: dict[str, str | bytes], ignore=()) -> Vault:
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_bytes(content.encode("utf-8"))
    return FilesystemVault(tmp_path, ignore=ignore)


def _memory(tmp_path: Path, files: dict[str, str | bytes], ignore=()) -> Vault:
    return InMemoryVault(files, ignore=ignore)


@pytest.fixture(params=[_filesystem, _memory], ids=["filesystem", "memory"])
def make_vault(request, tmp_path):
    return lambda files, ignore=(): request.param(tmp_path, files, ignore)


def test_adapters_subclass_the_port():
    # issubclass() refuses protocols with properties; the MRO shows the intent.
    assert Vault in FilesystemVault.__mro__
    assert Vault in InMemoryVault.__mro__


def test_notes_are_sorted_relative_posix_paths(make_vault):
    vault = make_vault({"b.md": "b", "a/z.md": "z", "a/b/c.md": "c", "A.md": "A"})
    assert vault.notes() == sorted(["A.md", "a/b/c.md", "a/z.md", "b.md"])


def test_notes_sort_by_path_segments(make_vault):
    # The order sorted(rglob) gave earlier indexes: by segment, so "a/x.md"
    # comes before "a-b/x.md" although "-" sorts before "/" as a character.
    vault = make_vault({"a-b/x.md": "x", "a/x.md": "x"})
    assert vault.notes() == ["a/x.md", "a-b/x.md"]


def test_dot_segments_are_never_notes(make_vault):
    vault = make_vault(
        {
            ".git/a.md": "x",
            ".trash/b.md": "x",
            "sub/.obsidian/c.md": "x",
            ".draft.md": "x",
            "ok.md": "x",
        }
    )
    assert vault.notes() == ["ok.md"]


def test_excalidraw_drawings_are_never_notes(make_vault):
    vault = make_vault({"Board.excalidraw.md": "x", "d/Plan.excalidraw.md": "x", "ok.md": "x"})
    assert vault.notes() == ["ok.md"]


def test_ignore_adds_prefixes(make_vault):
    vault = make_vault(
        {"Templates/t.md": "x", "Templates.md": "x", ".git/a.md": "x", "ok.md": "x"},
        ignore=["Templates/"],
    )
    # A prefix with a slash leaves the sibling file "Templates.md" alone.
    assert vault.notes() == ["Templates.md", "ok.md"]


def test_only_md_files_are_notes(make_vault):
    vault = make_vault({"a.png": b"\x89PNG", "b.canvas": "{}", "c.base": "x", "ok.md": "x"})
    assert vault.notes() == ["ok.md"]


def test_read_returns_text(make_vault):
    vault = make_vault({"a/n.md": "# Title\n\nbody\n"})
    note = vault.read("a/n.md")
    assert note.path == "a/n.md"
    assert note.text == "# Title\n\nbody\n"
    assert isinstance(note.mtime, float)


def test_read_returns_text_with_bom_dropped(make_vault):
    vault = make_vault({"n.md": "﻿---\ntags: [a]\n---\nbody\n"})
    assert vault.read("n.md").text.startswith("---\n")


def test_read_unreadable_raises_note_unreadable(make_vault):
    vault = make_vault({"latin.md": "caff\xe8".encode("latin-1")})
    assert vault.notes() == ["latin.md"]
    with pytest.raises(NoteUnreadable) as err:
        vault.read("latin.md")
    assert err.value.path == "latin.md"
    assert "latin.md" in str(err.value)


@pytest.mark.parametrize("path", ["../x.md", "/etc/passwd", ".git/a.md", "a/../../x.md", "n.txt"])
def test_read_refuses_paths_outside_the_rule(make_vault, path):
    vault = make_vault({".git/a.md": "secret", "n.txt": "x", "ok.md": "x"})
    with pytest.raises(NoteUnreadable):
        vault.read(path)


def test_read_missing_note_raises_note_unreadable(make_vault):
    vault = make_vault({"ok.md": "x"})
    with pytest.raises(NoteUnreadable):
        vault.read("gone.md")


# --- changes(): what may have changed, without the churn that holds no notes

# Each step is the set of files written at once; the real watcher may merge
# steps into one batch, so the contract compares the union of what it yields.
_STEPS = [
    {".obsidian/workspace.json": "{}"},
    {".git/index.md": "x", "Board.excalidraw.md": "x", "Templates/t.md": "x"},
    {"a.md": "# A"},
    {"sub/b.md": "# B", "img.png": "x"},
]


def _changes_on_disk(tmp_path, steps, ignore):
    import threading
    import time

    vault = FilesystemVault(tmp_path, ignore=ignore)
    stop = threading.Event()

    def write():
        time.sleep(0.5)  # let the watcher start
        for step in steps:
            for rel, text in step.items():
                path = tmp_path / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            time.sleep(0.3)
        time.sleep(0.8)
        stop.set()

    threading.Thread(target=write, daemon=True).start()
    return list(vault.changes(stop_event=stop))


def _changes_in_memory(tmp_path, steps, ignore):
    def apply(step):
        return lambda vault: (vault.files.update(step), list(step))[1]

    vault = InMemoryVault({}, ignore=ignore, script=[apply(step) for step in steps])
    return list(vault.changes())


@pytest.fixture(params=[_changes_on_disk, _changes_in_memory], ids=["filesystem", "memory"])
def changes_of(request, tmp_path):
    return lambda steps, ignore=(): request.param(tmp_path, steps, ignore)


def test_changes_skip_paths_outside_the_rule(changes_of):
    batches = changes_of(_STEPS, ignore=["Templates/"])
    assert all(batches), "an empty batch was yielded"
    changed = set().union(*batches)
    # Folders and attachments may hold or move notes, so they are kept.
    assert {"a.md", "sub/b.md", "img.png"} <= changed
    assert not changed & {".obsidian/workspace.json", ".git/index.md", "Board.excalidraw.md"}
    assert not any(path.startswith("Templates") for path in changed)


def test_dot_folder_churn_alone_yields_nothing(changes_of):
    assert changes_of([{".obsidian/workspace.json": "{}"}, {".obsidian/app.json": "{}"}]) == []
