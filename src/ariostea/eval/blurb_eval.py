"""Pure helpers for the blurb-aware reranking experiment (run_blurb_eval.py).

In the package rather than the runner so they are testable without a model,
an LLM or a database. See docs/design/2026-09-29-blurb-aware-reranking.md.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ariostea.eval.contextual import find_uncontextualized_notes

# Production hybrid search with the reranker scoring blurb plus chunk text.
HYBRID_CONTEXT = "HYBRID+CONTEXT"
# Channels whose ranking is the reranker's: a silent fallback to fused order
# would log fused numbers under their names.
RERANKED_CHANNELS = frozenset({"HYBRID", HYBRID_CONTEXT})

HYBRID_CONTROL = "2026-09-24-chunk-160t+40-dense-hybrid-sparse"
FUSED_CONTROL = "2026-09-28-chunk-160t+40-fused"


@dataclass(frozen=True)
class Arm:
    """One log entry: which channels it records and the run it is compared to.

    With `use_context` the arm's HYBRID is scored by the blurb-aware channel
    but logged under "HYBRID", so arm 3 compares cell for cell with the
    control's HYBRID.
    """

    key: str
    label: str
    channels: tuple[str, ...]
    control: str
    use_context: bool = False

    def source(self, channel: str) -> str:
        """The channel that scores the logged `channel`."""
        return HYBRID_CONTEXT if self.use_context and channel == "HYBRID" else channel


ARMS = (
    Arm(
        key="raw-rerank",
        label="Blurbs, reranker scores raw text",
        channels=("DENSE", "SPARSE", "HYBRID"),
        control=HYBRID_CONTROL,
    ),
    Arm(
        key="fused",
        label="Blurbs, no reranker",
        channels=("FUSED",),
        control=FUSED_CONTROL,
    ),
    Arm(
        key="context-rerank",
        label="Blurbs, reranker scores blurb and text",
        channels=("HYBRID",),
        control=HYBRID_CONTROL,
        use_context=True,
    ),
)


# Per granularity: the log's experiment name and the tag in its run ids. The
# tags differ so a per-chunk run cannot collide with a note-level run of the
# same day; "blurbs" is what the note-level runs were logged under.
_EXPERIMENTS = {"note": "Contextual blurbs", "chunk": "Per-chunk context"}
_RUN_TAGS = {"note": "blurbs", "chunk": "chunkctx"}
# The contextualizer's part of the index fingerprint, per granularity.
_MARKERS = {"note": "llm", "chunk": "llm-chunk"}


def _check_granularity(granularity: str) -> None:
    if granularity not in _EXPERIMENTS:
        raise ValueError(f"unknown granularity {granularity!r}; choose note or chunk")


def blurb_run_id(date: str, arm: Arm, granularity: str = "note") -> str:
    _check_granularity(granularity)
    return f"{date}-{_RUN_TAGS[granularity]}-{arm.key}"


def experiment_name(granularity: str) -> str:
    """The experiment the runs of this granularity are grouped under in the log."""
    _check_granularity(granularity)
    return _EXPERIMENTS[granularity]


def arm_label(arm: Arm, granularity: str) -> str:
    """`arm.label`, naming per-chunk contexts rather than blurbs in chunk mode."""
    _check_granularity(granularity)
    if granularity == "note":
        return arm.label
    return arm.label.replace("Blurbs", "Per-chunk context", 1)


def index_marker(granularity: str, model: str) -> str:
    """The fingerprint component an index contextualized by `model` at this
    granularity carries (see the contextualize adapters' `fingerprint`)."""
    _check_granularity(granularity)
    return f"{_MARKERS[granularity]}:{model}"


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


def select_arms(spec: str) -> tuple[Arm, ...]:
    """The arms named in a comma-separated `spec`, in ARMS order.

    Raises ValueError naming every unknown key, or when `spec` names none.
    """
    keys = [key.strip() for key in spec.split(",") if key.strip()]
    known = {arm.key for arm in ARMS}
    unknown = [key for key in keys if key not in known]
    if unknown:
        raise ValueError(
            f"unknown arm(s): {', '.join(unknown)}; choose from {', '.join(sorted(known))}"
        )
    if not keys:
        raise ValueError("no arms selected")
    return tuple(arm for arm in ARMS if arm.key in keys)


def needed_channels(arms: Iterable[Arm]) -> set[str]:
    """Every channel the `arms` are scored with."""
    return {arm.source(name) for arm in arms for name in arm.channels}


def control_mismatch(control_run: dict, chunking: dict, embedding: str) -> str | None:
    """Why `control_run` cannot stand as the no-blurb control for a run with
    this chunking and embedding, or None when it can. The runner reuses the
    control's `reachable` count, which is only valid under the same chunking."""
    config = control_run.get("config", {})
    problems = []
    if config.get("chunking") != chunking:
        problems.append(f"chunking {config.get('chunking')!r} != {chunking!r}")
    if config.get("embedding") != embedding:
        problems.append(f"embedding {config.get('embedding')!r} != {embedding!r}")
    if not problems:
        return None
    return f"control {control_run.get('id')!r}: " + "; ".join(problems)


def blurbs_by_note(rows: list[tuple[str, str | None]]) -> dict[str, str]:
    """{note_path: blurb} from (note_path, context_blurb) chunk rows.

    Blurbs are written once per note, so two different blurbs for one note
    mean the index is not what this experiment assumes: raise. Rows without a
    blurb are skipped; `require_full_coverage` is what rejects them.
    """
    blurbs: dict[str, str] = {}
    for path, blurb in rows:
        if not blurb:
            continue
        seen = blurbs.setdefault(path, blurb)
        if seen != blurb:
            raise ValueError(f"{path} has more than one blurb")
    return blurbs


def contexts_by_chunk(rows: Iterable[tuple[str, int, str | None]]) -> dict[str, dict[int, str]]:
    """{note_path: {ordinal: context}} from (note_path, ordinal, context) chunk
    rows. Rows without a context are skipped; `require_full_coverage` is what
    rejects them."""
    contexts: dict[str, dict[int, str]] = {}
    for path, ordinal, context in rows:
        if context:
            contexts.setdefault(path, {})[ordinal] = context
    return contexts


def lmstudio_load_problem(info: dict, model: str, min_context: int) -> str | None:
    """Why the model LM Studio describes in `info` (its /api/v0/models/<id>
    entry) is not ready for a run, or None when it is.

    LM Studio loads a model that is not loaded on the first request, with its
    own defaults (an 8k context, several parallel slots) rather than the ones
    the run needs, so a plain probe would pass and the run would then truncate
    long articles. A missing `loaded_context_length` is not held against it.
    """
    state = info.get("state")
    if state != "loaded":
        return f"{model} is not loaded in LM Studio (state {state!r})"
    loaded = info.get("loaded_context_length")
    if isinstance(loaded, int) and loaded < min_context:
        return f"{model} is loaded with a {loaded}-token context, below {min_context}"
    return None


class WarnedError(RuntimeError):
    """A block run under `fail_on_warnings` logged a warning."""


@contextmanager
def fail_on_warnings(logger_name: str) -> Iterator[None]:
    """Raise WarnedError (a RuntimeError) on exit if `logger_name` logged WARNING or above
    inside the block. For degradations the product deliberately survives,
    such as the reranker falling back to fused order, that would silently
    invalidate a measurement."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=logging.WARNING)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
    if records:
        messages = "; ".join(record.getMessage() for record in records)
        raise WarnedError(f"{logger_name} warned: {messages}")
