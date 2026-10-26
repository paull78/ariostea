from __future__ import annotations

from collections.abc import Mapping, Sequence

from ariostea.adapters.vault.rule import is_note
from ariostea.ports.vault import NoteFile, NoteUnreadable, Vault, VaultUnavailable


class InMemoryVault(Vault):
    """A vault held in a dict, for tests. Bytes that are not UTF-8 stand for
    an unreadable file; `available=False` stands for a missing folder."""

    def __init__(
        self,
        files: Mapping[str, str | bytes],
        ignore: Sequence[str] = (),
        available: bool = True,
        mtime: float = 0.0,
    ) -> None:
        self.files = dict(files)
        self.available = available
        self._ignore = tuple(ignore)
        self._mtime = mtime

    @property
    def location(self) -> str:
        return "memory://vault"

    def notes(self) -> list[str]:
        if not self.available:
            raise VaultUnavailable(self.location, "no such folder")
        # By segment, as the filesystem adapter's sorted(rglob) orders them.
        notes = (p for p in self.files if is_note(p, self._ignore))
        return sorted(notes, key=lambda p: p.split("/"))

    def read(self, path: str) -> NoteFile:
        if not is_note(path, self._ignore) or ".." in path.split("/") or path.startswith("/"):
            raise NoteUnreadable(path, "not a note of this vault")
        if path not in self.files:
            raise NoteUnreadable(path, "no such file")
        content = self.files[path]
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise NoteUnreadable(path, "not UTF-8 text") from None
        content = content.removeprefix("﻿").replace("\r\n", "\n")
        return NoteFile(path=path, text=content, mtime=self._mtime)
