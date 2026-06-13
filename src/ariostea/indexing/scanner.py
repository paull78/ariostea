from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class ScannedFile:
    rel_path: str
    raw: str
    mtime: float
    content_hash: str


def _is_ignored(rel_path: str, ignore: Sequence[str]) -> bool:
    return any(rel_path == pat.rstrip("/") or rel_path.startswith(pat) for pat in ignore)


def scan_vault(root: str | Path, ignore: Sequence[str] = ()) -> Iterator[ScannedFile]:
    root = Path(root)
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if _is_ignored(rel, ignore):
            continue
        raw = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        yield ScannedFile(
            rel_path=rel,
            raw=raw,
            mtime=path.stat().st_mtime,
            content_hash=digest,
        )
