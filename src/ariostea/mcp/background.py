from __future__ import annotations

import logging
import threading

from ariostea.domain.models import Progress
from ariostea.indexing.index_runs import IndexRuns, Run

logger = logging.getLogger(__name__)


class BackgroundRuns:
    """Index runs started by an MCP client, each on a thread this adapter
    owns, so the tool call answers at once and the server keeps answering
    searches (server-robustness.09). It keeps the latest progress of its own
    run for `status`; runs in other processes report only that they run."""

    def __init__(self, runs: IndexRuns) -> None:
        self._runs = runs
        self._progress: Progress | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start a run, or say False when one is already going on anywhere."""
        run = self._runs.begin(wait=False)
        if run is None:
            return False
        # A daemon thread: when the client closes the server, the process
        # ends, the operating system releases the run lock, and the note
        # being written rolls back as one transaction.
        self._thread = threading.Thread(
            target=self._work, args=(run,), name="ariostea-index", daemon=True
        )
        self._thread.start()
        return True

    def progress(self) -> Progress | None:
        return self._progress

    def join(self, timeout: float | None = None) -> None:
        """Wait for the current run's thread to end; for tests and shutdown."""
        if self._thread is not None:
            self._thread.join(timeout)

    def _keep(self, progress: Progress) -> None:
        self._progress = progress

    def _work(self, run: Run) -> None:
        try:
            run.execute(progress=self._keep)
        except Exception:
            # Stdout belongs to the protocol on stdio; logging goes to stderr.
            logger.exception("index run failed")
        finally:
            self._progress = None
