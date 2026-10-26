from __future__ import annotations

from collections.abc import Sequence


def is_note(path: str, ignore: Sequence[str] = ()) -> bool:
    """Whether a vault-relative POSIX path is a note.

    A note is a `.md` file outside any dot-folder (`.obsidian`, `.git`,
    `.trash`) and not itself hidden, that is not an Excalidraw drawing and
    does not fall under an `ignore` prefix. "Templates/" also matches a file
    named "Templates".
    """
    return path.endswith(".md") and "" not in path.split("/") and may_hold_notes(path, ignore)


def may_hold_notes(path: str, ignore: Sequence[str] = ()) -> bool:
    """Whether a change at `path` can affect the notes: `is_note` without the
    `.md` test. A moved folder or a vanished vault is reported only by its
    own path ("." for the root), so those must pass (vault-hardening.24)."""
    if path == ".":
        return True
    if any(part.startswith(".") for part in path.split("/")):
        return False
    if path.endswith(".excalidraw.md"):
        return False
    return not any(path == pat.rstrip("/") or path.startswith(pat) for pat in ignore)
