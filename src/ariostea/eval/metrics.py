"""Retrieval metrics for the eval harness.

These assume **one correct note per query** (the gold set's normal case). Under
that assumption both metrics are exact and simple:

- ``recall_at_k`` is "did we find it in the top k" (1.0/0.0). With a single
  relevant note, fraction-found can only be 0/1 or 1/1, so the binary form *is*
  true recall@k (a.k.a. hit@k).
- ``reciprocal_rank`` looks only at the first relevant hit; with one correct
  note there is nothing later to miss.

If a query ever has multiple acceptable notes, these degrade gracefully but stop
being exact: ``recall_at_k`` returns 1.0 on *any* hit (not the found-fraction),
and ``reciprocal_rank`` still scores only the earliest hit. Reach for MAP/nDCG
if multi-answer queries become common.
"""

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
