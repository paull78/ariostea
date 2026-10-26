from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class NoteFile:
    """A note as the vault holds it: vault-relative POSIX path, text, mtime."""

    path: str
    text: str
    mtime: float


class VaultUnavailable(Exception):
    """The vault itself cannot be listed: missing, not a folder, unreadable."""

    def __init__(self, location: str, reason: str) -> None:
        super().__init__(f"vault {location} is unavailable: {reason}")
        self.location = location
        self.reason = reason


class NoteUnreadable(Exception):
    """One note cannot be read: not a note, gone, not UTF-8, no permission."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


@runtime_checkable
class Vault(Protocol):
    """The notes Ariostea indexes. The adapter decides what counts as a note,
    so indexing, `get_note` and `watch` cannot disagree about it."""

    @property
    def location(self) -> str:
        """Where the vault is, for messages."""
        ...

    def notes(self) -> list[str]:
        """Every note's vault-relative POSIX path, sorted. Raises
        VaultUnavailable."""
        ...

    def read(self, path: str) -> NoteFile:
        """The note at `path`. Raises NoteUnreadable, also for a path that is
        not a note of this vault."""
        ...

    def changes(self, stop_event: threading.Event | None = None) -> Iterator[frozenset[str]]:
        """Block, yielding a batch of vault-relative paths each time something
        that may hold notes changes ("." is the vault itself). Changes that
        cannot touch a note, such as `.obsidian/` churn, yield nothing; a
        batch is never empty. Ends when `stop_event` is set."""
        ...
