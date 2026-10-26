"""Markdown structure shared by the parser and the chunker, so a note's title
and its sections cannot disagree about what is a heading."""

from __future__ import annotations

import re
from collections.abc import Iterator

HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
# CommonMark fences: up to three spaces, then three or more backticks or
# tildes. A backtick fence's info string may not hold a backtick.
_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def heading_lines(body: str) -> Iterator[tuple[str, re.Match[str] | None]]:
    """Each line of `body` with its heading match, or None for a line that
    is not a heading. Lines inside a fenced code block are never headings
    (vault-hardening.07). A fence closes on the same character repeated at
    least as many times with nothing after it; an unclosed fence runs to the
    end."""
    fence = ""  # the opening run while inside a fence
    for line in body.splitlines():
        if fence:
            stripped = line.strip()
            if (
                len(line) - len(line.lstrip(" ")) <= 3
                and stripped
                and set(stripped) == {fence[0]}
                and len(stripped) >= len(fence)
            ):
                fence = ""
            yield line, None
            continue
        opening = _OPEN.match(line)
        if opening and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            fence = opening.group(1)
            yield line, None
            continue
        yield line, HEADING.match(line)
