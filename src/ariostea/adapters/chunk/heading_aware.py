from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ariostea.domain.models import Chunk, Note
from ariostea.ports.pipeline import Chunker

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    text: str


def _split_sections(body: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[str] = []  # current heading path by level
    buffer: list[str] = []
    current_path: tuple[str, ...] = ()

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append(_Section(current_path, text))
        buffer.clear()

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            stack[:] = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            current_path = tuple(s for s in stack if s)
            buffer.append(line)
        else:
            buffer.append(line)
    flush()
    return sections


def _token_count(text: str) -> int:
    return len(text.split())


class HeadingAwareChunker(Chunker):
    """Split at every heading, then cut long sections into windows.

    `max_tokens` caps each piece's total cost. A word costs 1 unless `count`
    is given, in which case it costs `count(word)` -- the embedding model's
    tokenizer, so the cap means what the model actually reads. `overlap`
    repeats up to that much cost from the end of one piece at the start of the
    next, so a span cut at a boundary is still whole in one chunk. Windows
    never cross a heading: the heading split is the part of the policy that
    already follows the document's structure.

    With the defaults the output is identical to the chunker before overlap
    and cost functions existed; a test pins that on the whole wiki corpus.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap: int = 0,
        count: Callable[[str], int] | None = None,
        unit: str = "words",
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap = overlap
        self._count = count
        self._unit = unit

    @property
    def fingerprint(self) -> str:
        """Empty for the default policy, so an index built before chunking was
        configurable keeps its stored fingerprint instead of being forced into
        a full reindex it does not need. Any other policy names itself, so
        changing it re-chunks every note."""
        if (self.max_tokens, self.overlap, self._unit, self._count) == (512, 0, "words", None):
            return ""
        return f"chunk:heading_aware:{self.max_tokens}:{self.overlap}:{self._unit}"

    def chunk(self, note: Note, body: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0
        for section in _split_sections(body):
            for piece in self._fit(section.text):
                chunks.append(
                    Chunk(
                        note_path=note.path,
                        ordinal=ordinal,
                        heading_path=section.heading_path,
                        text=piece,
                        token_count=_token_count(piece),
                    )
                )
                ordinal += 1
        return chunks

    def _fit(self, text: str) -> list[str]:
        words = text.split()
        costs = [1] * len(words) if self._count is None else [self._count(w) for w in words]
        if sum(costs) <= self.max_tokens:
            return [text]  # untouched, newlines and all, as before

        pieces: list[str] = []
        start = 0
        while start < len(words):
            end, total = start, 0
            # Always take at least one word, so a word costlier than the whole
            # budget becomes its own piece instead of stalling the window.
            while end < len(words) and (end == start or total + costs[end] <= self.max_tokens):
                total += costs[end]
                end += 1
            pieces.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            # Step back to repeat up to `overlap` of cost, but always leave the
            # next window starting at least one word further on.
            back, repeated = end, 0
            while back > start + 1 and repeated + costs[back - 1] <= self.overlap:
                repeated += costs[back - 1]
                back -= 1
            start = back
        return pieces
