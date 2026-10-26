from __future__ import annotations

from collections.abc import Sequence


def is_note(path: str, ignore: Sequence[str] = ()) -> bool:
    """Whether a vault-relative POSIX path is a note.

    A note is a `.md` file outside any dot-folder (`.obsidian`, `.git`,
    `.trash`) and not itself hidden, that is not an Excalidraw drawing and
    does not fall under an `ignore` prefix. "Templates/" also matches a file
    named "Templates".
    """
    parts = path.split("/")
    if not path.endswith(".md") or path.endswith(".excalidraw.md"):
        return False
    if any(part.startswith(".") or part == "" for part in parts):
        return False
    return not any(path == pat.rstrip("/") or path.startswith(pat) for pat in ignore)
