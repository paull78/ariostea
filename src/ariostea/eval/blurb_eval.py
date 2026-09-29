"""Pure helpers for the blurb-aware reranking experiment (run_blurb_eval.py).

In the package rather than the runner so they are testable without a model,
an LLM or a database. See docs/design/2026-09-29-blurb-aware-reranking.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.contextual import find_uncontextualized_notes

HYBRID_CONTROL = "2026-09-24-chunk-160t+40-dense-hybrid-sparse"
FUSED_CONTROL = "2026-09-28-chunk-160t+40-fused"


@dataclass(frozen=True)
class Arm:
    """One log entry: which channels it records and the run it is compared to.

    `source` names the channel each logged channel is scored with, so arm 3
    can log its blurb-aware hybrid under "HYBRID" and be compared cell for
    cell with the control's HYBRID.
    """

    key: str
    label: str
    channels: tuple[str, ...]
    control: str
    source: dict[str, str]


ARMS = (
    Arm(
        key="raw-rerank",
        label="Blurbs, reranker scores raw text",
        channels=("DENSE", "SPARSE", "HYBRID"),
        control=HYBRID_CONTROL,
        source={"DENSE": "DENSE", "SPARSE": "SPARSE", "HYBRID": "HYBRID"},
    ),
    Arm(
        key="fused",
        label="Blurbs, no reranker",
        channels=("FUSED",),
        control=FUSED_CONTROL,
        source={"FUSED": "FUSED"},
    ),
    Arm(
        key="context-rerank",
        label="Blurbs, reranker scores blurb and text",
        channels=("HYBRID",),
        control=HYBRID_CONTROL,
        source={"HYBRID": "HYBRID+CONTEXT"},
    ),
)


def blurb_run_id(date: str, arm: Arm) -> str:
    return f"{date}-blurbs-{arm.key}"


class BlurbCoverageError(RuntimeError):
    """The blurbed index is missing blurbs; scoring it would understate them."""


def require_full_coverage(rows: list[tuple[str, str | None]]) -> int:
    """Return the number of blurbed notes, or raise naming every note that
    fell back to plain text. `rows` is (note_path, context_blurb) per chunk."""
    if not rows:
        raise BlurbCoverageError("the index has no chunks")
    missing = find_uncontextualized_notes(rows)
    if missing:
        raise BlurbCoverageError(f"{len(missing)} note(s) have no blurb: {', '.join(missing)}")
    return len({path for path, _ in rows})
