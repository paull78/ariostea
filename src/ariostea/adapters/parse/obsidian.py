from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from ariostea.domain.models import Note
from ariostea.ports.pipeline import MarkdownParser

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_TAG = re.compile(r"(?:^|\s)#([A-Za-z0-9_\-/]+)")
_WIKILINK = re.compile(r"\[\[([^\]|#]+)")
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    block, body = m.group(1), raw[m.end() :]
    fm: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm, body


class ObsidianMarkdownParser(MarkdownParser):
    def parse(self, path: str, raw: str, mtime: float) -> tuple[Note, str]:
        frontmatter, body = _parse_frontmatter(raw)
        heading = _H1.search(body)
        title = heading.group(1).strip() if heading else PurePosixPath(path).stem
        tags = tuple(sorted(set(_TAG.findall(body))))
        wikilinks = tuple(sorted(set(link.strip() for link in _WIKILINK.findall(body))))
        content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        note = Note(
            path=path,
            title=title,
            frontmatter=frontmatter,
            tags=tags,
            wikilinks=wikilinks,
            content_hash=content_hash,
            mtime=mtime,
        )
        return note, body
