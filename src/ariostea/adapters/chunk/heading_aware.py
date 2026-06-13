from __future__ import annotations

import re
from dataclasses import dataclass

from ariostea.domain.models import Note, Chunk
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
    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

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
        if len(words) <= self.max_tokens:
            return [text]
        pieces: list[str] = []
        for i in range(0, len(words), self.max_tokens):
            pieces.append(" ".join(words[i : i + self.max_tokens]))
        return pieces
