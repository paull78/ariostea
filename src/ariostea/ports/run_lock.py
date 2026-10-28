from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RunLock(Protocol):
    """One index run at a time, across every process that writes the index.
    A holder is one instance; a second instance on the same lock is another
    holder, as another process would be."""

    def try_acquire(self) -> bool:
        """Take the lock if it is free and say whether this holder got it. Never waits."""
        ...

    def acquire(self) -> None:
        """Wait until the lock is free, then take it."""
        ...

    def release(self) -> None:
        """Give the lock up; nothing happens if this holder does not have it."""
        ...

    def held(self) -> bool:
        """Whether any holder has the lock, this one included."""
        ...
