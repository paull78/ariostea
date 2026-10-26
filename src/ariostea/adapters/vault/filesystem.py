from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from ariostea.adapters.vault.rule import is_note
from ariostea.ports.vault import NoteFile, NoteUnreadable, Vault, VaultUnavailable


class FilesystemVault(Vault):
    """An Obsidian vault folder on disk."""

    def __init__(self, root: str | Path, ignore: Sequence[str] = ()) -> None:
        self._root = Path(root).expanduser()
        self._ignore = tuple(ignore)

    @property
    def location(self) -> str:
        return str(self._root)

    def notes(self) -> list[str]:
        if not self._root.exists():
            raise VaultUnavailable(self.location, "no such folder")
        if not self._root.is_dir():
            raise VaultUnavailable(self.location, "not a folder")
        try:
            # sorted(rglob) keeps the order earlier indexes and the eval used.
            found = sorted(self._root.rglob("*.md"))
        except OSError as exc:
            raise VaultUnavailable(self.location, exc.strerror or str(exc)) from exc
        rels = (p.relative_to(self._root).as_posix() for p in found if p.is_file())
        return [rel for rel in rels if is_note(rel, self._ignore)]

    def read(self, path: str) -> NoteFile:
        if os.path.isabs(path) or ".." in path.split("/") or not is_note(path, self._ignore):
            raise NoteUnreadable(path, "not a note of this vault")
        full = self._root / path
        try:
            if not full.resolve().is_relative_to(self._root.resolve()):
                raise NoteUnreadable(path, "outside the vault")
            # Text mode turns CRLF into LF; utf-8-sig drops a leading BOM.
            with open(full, encoding="utf-8-sig") as f:
                text = f.read()
            mtime = full.stat().st_mtime
        except UnicodeDecodeError:
            raise NoteUnreadable(path, "not UTF-8 text") from None
        except FileNotFoundError:
            raise NoteUnreadable(path, "no such file") from None
        except PermissionError:
            raise NoteUnreadable(path, "permission denied") from None
        except IsADirectoryError:
            raise NoteUnreadable(path, "a folder, not a file") from None
        except OSError as exc:
            raise NoteUnreadable(path, exc.strerror or str(exc)) from exc
        return NoteFile(path=path, text=text, mtime=mtime)
