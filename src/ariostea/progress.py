from __future__ import annotations

import time
from collections.abc import Callable

from ariostea.domain.models import Progress


class ProgressPrinter:
    """Turns the indexer's per-note progress into a line every `every` notes
    or every `interval` seconds, whichever comes first, and one at the end.
    The clock lives here, on the driving side, not in the use case."""

    def __init__(
        self,
        write: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
        every: int = 100,
        interval: float = 10.0,
    ) -> None:
        self._write = write
        self._clock = clock
        self._every = every
        self._interval = interval
        self._last = clock()

    def __call__(self, progress: Progress) -> None:
        now = self._clock()
        if (
            progress.done % self._every == 0
            or progress.done == progress.total
            or now - self._last >= self._interval
        ):
            self._last = now
            self._write(
                f"{progress.done}/{progress.total:,} notes · {progress.embedded} embedded"
                f" · {progress.skipped} skipped"
            )
