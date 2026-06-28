from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...
