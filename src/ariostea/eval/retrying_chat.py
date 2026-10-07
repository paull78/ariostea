"""Retry a flaky `ChatProvider` before its caller (or the cache in front of
it) ever sees a failure.

A 15-hour unattended run over 4,246 chunks will hit the odd transient
endpoint hiccup; without a retry, `LLMChunkContextualizer` degrades that one
chunk to plain text and the coverage gate then aborts the whole run at the
end. Retrying here, inside the cache boundary, means a transient failure
costs a short wait instead of the run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from ariostea.adapters.chat.openai_compat import ChatError
from ariostea.ports.chat import ChatProvider

logger = logging.getLogger(__name__)


class RetryingChat(ChatProvider):
    """Wrap a `ChatProvider`, retrying on `ChatError` or a blank reply.

    A blank reply is treated as a failure worth retrying rather than a valid
    (if useless) answer, and after the last attempt it is raised as
    `ChatError` too, so it can never reach the cache and be written down as
    the permanent answer for that chunk.
    """

    def __init__(
        self,
        inner: ChatProvider,
        attempts: int = 3,
        backoff_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._attempts = attempts
        self._backoff_s = backoff_s
        self._sleep = sleep

    def complete(self, system: str, user: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                reply = self._inner.complete(system, user)
            except ChatError as exc:
                last_error = exc
            else:
                if (reply or "").strip():
                    return reply
                last_error = ChatError("chat reply was empty or whitespace")
            if attempt < self._attempts:
                logger.warning(
                    "chat call failed (attempt %d/%d): %s; retrying",
                    attempt,
                    self._attempts,
                    last_error,
                )
                self._sleep(self._backoff_s * attempt)
        raise ChatError(f"chat call failed after {self._attempts} attempts: {last_error}")
