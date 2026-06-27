from __future__ import annotations


def recall_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """1.0 if any expected note appears in the top-k ranked notes, else 0.0.

    `ranked` is a list of note paths in rank order (best first), already
    deduplicated to one entry per note.
    """
    top = ranked[:k]
    return 1.0 if any(path in top for path in expected) else 0.0


def reciprocal_rank(expected: set[str], ranked: list[str]) -> float:
    """Reciprocal of the 1-based rank of the first expected note; 0.0 if none."""
    for index, path in enumerate(ranked):
        if path in expected:
            return 1.0 / (index + 1)
    return 0.0
