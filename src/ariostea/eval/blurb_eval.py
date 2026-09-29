"""Pure helpers for the blurb-aware reranking experiment (run_blurb_eval.py).

In the package rather than the runner so they are testable without a model,
an LLM or a database. See docs/design/2026-09-29-blurb-aware-reranking.md.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from ariostea.eval.contextual import find_uncontextualized_notes
from ariostea.ports.chat import ChatProvider

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


def require_full_coverage(rows: list[tuple[str, str | None]], granularity: str = "note") -> int:
    """Return the number of contextualized notes, or raise naming every note
    that fell back to plain text. `rows` is (note_path, context) per chunk.

    In note mode every chunk of a note shares one blurb, so naming the notes
    is the whole story. In chunk mode a note can have only some of its chunks
    fail, so the message also counts chunks: "12 of 4246 chunks have no
    context, in 3 note(s): a.md, b.md, c.md".
    """
    _check_granularity(granularity)
    if not rows:
        raise BlurbCoverageError("the index has no chunks")
    missing_notes = find_uncontextualized_notes(rows)
    if missing_notes:
        if granularity == "chunk":
            missing_chunks = sum(1 for _, context in rows if not context)
            raise BlurbCoverageError(
                f"{missing_chunks} of {len(rows)} chunks have no context, in "
                f"{len(missing_notes)} note(s): {', '.join(missing_notes)}"
            )
        raise BlurbCoverageError(
            f"{len(missing_notes)} note(s) have no blurb: {', '.join(missing_notes)}"
        )
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


@dataclass(frozen=True)
class Preflight:
    """Outcome of `lmstudio_preflight`.

    `abort` names why the run must not start at all. `fallback_to_probe`
    means the endpoint does not look like LM Studio, so the caller must fall
    back to a plain chat probe (and warn loudly that the load state -- context
    window, parallel slots -- could not be verified that way)."""

    abort: str | None = None
    fallback_to_probe: bool = False


def lmstudio_preflight(
    models_endpoint_responded: bool, info: dict | None, model: str, min_context: int
) -> Preflight:
    """Decide whether the run can proceed against LM Studio, given whether its
    `/api/v0/models` endpoint answered and, when it did, what the per-model
    lookup at `/api/v0/models/<model>` returned.

    A server that answers `/api/v0/models` is LM Studio, so a failed or
    missing per-model lookup means the named model is not loaded as this run
    needs -- that aborts, naming the model, rather than falling back to the
    plain chat probe, which only catches a fully dead endpoint and would
    silently let a wrongly-loaded model (8k context, several parallel slots)
    through. Only when the models endpoint itself does not answer, so the
    server may not be LM Studio at all, does the chat probe take over.
    """
    if not models_endpoint_responded:
        return Preflight(fallback_to_probe=True)
    if info is None:
        return Preflight(abort=f"{model} is not loaded in LM Studio (not listed by /api/v0/models)")
    problem = lmstudio_load_problem(info, model, min_context)
    return Preflight(abort=problem)


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


class _HitMissCounts(Protocol):
    """What `NoteProgressChat` needs from the cache in front of it, so it is
    not coupled to `CachingChat` itself."""

    hits: int
    misses: int


def format_note_progress(
    path: str, chunk_count: int, hits: int, misses: int, note_elapsed: float, run_elapsed: float
) -> str:
    """One progress line printed as chunk-mode contextualization starts a note:
    which note and how many chunks it has, the cache's hit/miss tally so far,
    and how long the previous note took plus the run's total elapsed time."""
    return (
        f"{path} ({chunk_count} chunks): {hits} hits, {misses} misses so far, "
        f"{note_elapsed:.1f}s for the previous note, {run_elapsed:.1f}s run so far"
    )


class NoteProgressChat(ChatProvider):
    """A pass-through `ChatProvider` that prints one line per note as chunk
    mode reaches it.

    A new note is detected by a change in `system`: `LLMChunkContextualizer`
    puts the whole note's text there, identical for every chunk of that note
    and different for the next one, so watching for the change costs no LLM
    calls and never touches the prompt (the cache key depends on it). `notes`
    supplies the path and chunk count for each note, in the order the indexer
    will reach them (`scan_vault`'s sort order); once exhausted, further notes
    print as "?" rather than raising, so a mismatch never aborts the run over
    a progress cosmetic.

    Place this outside the cache (wrapping it, not wrapped by it) so `hits`
    and `misses` read from `cache` reflect every chunk seen so far, including
    the one that triggered this note's line.
    """

    def __init__(
        self,
        inner: ChatProvider,
        notes: Sequence[tuple[str, int]],
        cache: _HitMissCounts,
        clock: Callable[[], float] = time.monotonic,
        out: Callable[[str], None] = print,
    ) -> None:
        self._inner = inner
        self._notes = iter(notes)
        self._cache = cache
        self._clock = clock
        self._out = out
        self._last_system: str | None = None
        self._run_start = clock()
        self._note_start = self._run_start

    def complete(self, system: str, user: str) -> str:
        if system != self._last_system:
            self._last_system = system
            now = self._clock()
            path, chunk_count = next(self._notes, ("?", 0))
            self._out(
                format_note_progress(
                    path,
                    chunk_count,
                    self._cache.hits,
                    self._cache.misses,
                    now - self._note_start,
                    now - self._run_start,
                )
            )
            self._note_start = now
        return self._inner.complete(system, user)
