from typing import Protocol, runtime_checkable

from ariostea.domain.models import RetrievedChunk


@runtime_checkable
class Fuser(Protocol):
    def fuse(
        self, dense: list[RetrievedChunk], sparse: list[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...
