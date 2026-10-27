from __future__ import annotations

import threading
from collections.abc import Callable

from ariostea.domain.models import IndexReport, Progress
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.vault import Vault, VaultUnavailable


def _ignore(_: object) -> None:
    pass


class WatchVault:
    """Keep the index current: one pass at start, then one per batch of
    changes the vault reports. What each pass did goes to `on_report`, and
    what the user should know goes to `on_warning` as one-line messages."""

    def __init__(self, indexer: IndexVault, vault: Vault) -> None:
        self._indexer = indexer
        self._vault = vault
        self._warned: set[str] = set()

    def run(
        self,
        stop_event: threading.Event | None = None,
        on_report: Callable[[IndexReport], None] = _ignore,
        on_warning: Callable[[str], None] = _ignore,
        progress: Callable[[Progress], None] | None = None,
    ) -> None:
        """Watch until `stop_event` is set. A vault that fails its guard at
        start raises VaultUnavailable; on a later pass it is a warning, the
        index stays as it was, and the next change tries again. `progress`
        hears about the first pass, the only long one."""
        self._pass(on_report, on_warning, progress)
        for _batch in self._vault.changes(stop_event):
            try:
                self._pass(on_report, on_warning)
            except VaultUnavailable as exc:
                on_warning(str(exc))

    def _pass(
        self,
        on_report: Callable[[IndexReport], None],
        on_warning: Callable[[str], None],
        progress: Callable[[Progress], None] | None = None,
    ) -> None:
        report = self._indexer.index(progress=progress)
        # Warn about a bad file once, not on every pass while it stays bad.
        for path, reason in report.skipped:
            if path not in self._warned:
                on_warning(f"skipped {path}: {reason}")
        self._warned = {path for path, _ in report.skipped}
        on_report(report)
