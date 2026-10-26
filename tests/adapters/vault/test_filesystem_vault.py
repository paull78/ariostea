"""Cases only real files have: a missing root, permissions, files vanishing
between listing and reading, line endings, `~` and modification times."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ariostea.adapters.vault.filesystem import FilesystemVault
from ariostea.ports.vault import NoteUnreadable, VaultUnavailable


def test_missing_root_raises_vault_unavailable(tmp_path):
    root = tmp_path / "nope"
    with pytest.raises(VaultUnavailable) as err:
        FilesystemVault(root).notes()
    assert str(root) in str(err.value)


def test_root_that_is_a_file_raises_vault_unavailable(tmp_path):
    root = tmp_path / "file.md"
    root.write_text("x")
    with pytest.raises(VaultUnavailable) as err:
        FilesystemVault(root).notes()
    assert str(root) in str(err.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_permission_denied_raises_note_unreadable(tmp_path):
    (tmp_path / "locked.md").write_text("x")
    (tmp_path / "locked.md").chmod(0)
    try:
        with pytest.raises(NoteUnreadable) as err:
            FilesystemVault(tmp_path).read("locked.md")
        assert err.value.path == "locked.md"
    finally:
        (tmp_path / "locked.md").chmod(0o644)


def test_vanished_file_raises_note_unreadable(tmp_path):
    (tmp_path / "n.md").write_text("x")
    vault = FilesystemVault(tmp_path)
    assert vault.notes() == ["n.md"]
    (tmp_path / "n.md").unlink()
    with pytest.raises(NoteUnreadable):
        vault.read("n.md")


def test_directory_named_like_a_note_is_not_a_note(tmp_path):
    (tmp_path / "folder.md").mkdir()
    (tmp_path / "folder.md" / "inner.md").write_text("x")
    vault = FilesystemVault(tmp_path)
    assert vault.notes() == ["folder.md/inner.md"]
    with pytest.raises(NoteUnreadable):
        vault.read("folder.md")


def test_symlink_escaping_the_root_is_refused(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    root = tmp_path / "vault"
    root.mkdir()
    (root / "link.md").symlink_to(outside)
    with pytest.raises(NoteUnreadable):
        FilesystemVault(root).read("link.md")


def test_crlf_reads_as_lf(tmp_path):
    (tmp_path / "n.md").write_bytes(b"# T\r\n\r\nbody\r\n")
    assert FilesystemVault(tmp_path).read("n.md").text == "# T\n\nbody\n"


def test_tilde_root_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "v").mkdir()
    (tmp_path / "v" / "n.md").write_text("x")
    vault = FilesystemVault("~/v")
    assert vault.notes() == ["n.md"]
    assert vault.location == str(tmp_path / "v")


def test_mtime_is_the_files(tmp_path):
    path = tmp_path / "n.md"
    path.write_text("x")
    os.utime(path, (1_000_000_000, 1_234_567_890))
    assert FilesystemVault(tmp_path).read("n.md").mtime == 1_234_567_890.0


def test_location_is_the_root(tmp_path):
    assert FilesystemVault(Path(tmp_path)).location == str(tmp_path)
