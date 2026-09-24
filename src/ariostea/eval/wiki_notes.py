"""Read the committed Wikipedia corpus back off disk.

Gold generation and validation both need the note *body* -- the text a span
must appear in -- not the frontmatter that `wiki_corpus.render_note` wrapped
it in. This module is the one place that knows how to undo that wrapping, so
the two consumers cannot drift apart on what "the note text" means.
"""

from __future__ import annotations

import re
from pathlib import Path

_FENCE = "---\n"
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def strip_frontmatter(raw: str) -> str:
    """Drop a leading `---\\n...\\n---\\n` block and the blank lines after it.

    Returns `raw` unchanged when there is no frontmatter, and -- deliberately
    -- also when the opening fence is never closed. A truncated note is a
    damaged note; returning its whole text keeps the damage visible to the
    caller (the H1 check in `note_titles`, the verbatim check in the
    validation gate) instead of silently returning an empty body that would
    just look like an article with nothing in it.
    """
    if not raw.startswith(_FENCE):
        return raw
    end = raw.find("\n---\n", len(_FENCE) - 1)
    if end == -1:
        return raw
    return raw[end + len("\n---\n") :].lstrip("\n")


def load_corpus_notes(wiki_dir: Path) -> dict[str, str]:
    """Every `<cluster>/<slug>.md` under `wiki_dir`, keyed by that relative
    path, frontmatter stripped.

    Globbed exactly one level deep on purpose: the corpus layout is one
    directory per cluster, and a recursive glob would also sweep up whatever a
    future subdirectory holds. Paths are sorted so every consumer -- passage
    selection above all -- iterates in the same order on every machine.
    """
    return {
        f"{path.parent.name}/{path.name}": strip_frontmatter(path.read_text(encoding="utf-8"))
        for path in sorted(wiki_dir.glob("*/*.md"))
    }


def note_titles(notes: dict[str, str]) -> dict[str, str]:
    """Note path -> article title, read from each note's H1.

    Raises on a note with no H1 rather than falling back to the path stem. The
    title is what the validation gate uses to reject queries answerable from
    the title alone, so a wrong title silently weakens a gate -- that failure
    has to be loud.
    """
    titles: dict[str, str] = {}
    for path, text in notes.items():
        match = _H1.search(text)
        if match is None:
            raise ValueError(f"note {path!r} has no H1 heading to read a title from")
        titles[path] = match.group(1).strip()
    return titles


def cluster_of(note_path: str) -> str:
    """`string-instruments/violin.md` -> `string-instruments`."""
    return note_path.split("/", 1)[0]
