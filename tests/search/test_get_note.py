"""GetNote returns the note's file as written, and only for a path the
index holds (vault-hardening.09)."""

import pytest

from ariostea.adapters.vault.memory import InMemoryVault
from ariostea.domain.models import NoteMissing, NoteText
from ariostea.search.get_note import GetNote

LONG = (
    "---\ntags: [music]\nstatus: draft\n---\n# Violin\n\n"
    + "\n\n".join(f"Paragraph {i}: " + "word " * 120 for i in range(12))
    + "\n\n```python\n# a comment\nprint('x')\n```\n\n\n- last item\n"
)


class FakeReader:
    def __init__(self, titles):
        self.titles = titles

    def note_titles(self, paths):
        return {p: self.titles[p] for p in paths if p in self.titles}


class RecordingVault(InMemoryVault):
    def __init__(self, files):
        super().__init__(files)
        self.reads = []

    def read(self, path):
        self.reads.append(path)
        return super().read(path)


def test_returns_file_text_for_indexed_note():  # AC-12
    vault = RecordingVault({"music/Violin.md": LONG})
    result = GetNote(FakeReader({"music/Violin.md": "Violin"}), vault).get("music/Violin.md")
    assert result == NoteText(path="music/Violin.md", title="Violin", text=LONG)


def test_unindexed_path_is_missing_and_not_read():  # AC-13
    vault = RecordingVault({"private.md": "secret"})
    result = GetNote(FakeReader({}), vault).get("private.md")
    assert result == NoteMissing(path="private.md", reason=None)
    assert vault.reads == []


@pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd", "a/../private.md"])
def test_traversal_paths_are_missing_and_not_read(path):  # AC-13
    vault = RecordingVault({"private.md": "secret"})
    assert GetNote(FakeReader({"a.md": "A"}), vault).get(path) == NoteMissing(path, None)
    assert vault.reads == []


def test_indexed_but_deleted_file_is_missing_with_reason():  # AC-13
    vault = RecordingVault({})
    result = GetNote(FakeReader({"gone.md": "Gone"}), vault).get("gone.md")
    assert result == NoteMissing(path="gone.md", reason="no such file")


def test_indexed_but_unreadable_file_is_missing_with_reason():
    vault = RecordingVault({"bad.md": b"\xff\xfe"})
    result = GetNote(FakeReader({"bad.md": "Bad"}), vault).get("bad.md")
    assert result == NoteMissing(path="bad.md", reason="not UTF-8 text")
