from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from watchfiles import watch as _default_watch

from ariostea.indexing.index_vault import IndexVault

# A watcher: given a path (and stop_event), yield a batch per change.
WatchFn = Callable[..., Iterable[object]]


class WatchVault:
    """Keep the index current: do an initial incremental index, then re-index
    on every filesystem change batch. The watch function is injected so the
    loop is testable without real filesystem events."""

    def __init__(
        self,
        indexer: IndexVault,
        root: str | Path,
        ignore: Sequence[str] = (),
        watch_fn: WatchFn = _default_watch,
    ) -> None:
        self._indexer = indexer
        self._root = root
        self._ignore = tuple(ignore)
        self._watch_fn = watch_fn

    def run(self, stop_event: object | None = None) -> None:
        self._indexer.index(self._root, ignore=self._ignore)  # initial sync
        for _changes in self._watch_fn(self._root, stop_event=stop_event):
            self._indexer.index(self._root, ignore=self._ignore)
