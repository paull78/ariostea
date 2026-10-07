"""Measure contextual indexing, and blurb-aware reranking, on the wiki gold set.

Usage:
    uv run python eval/run_blurb_eval.py                      # build the blurbed index, score, log
    uv run python eval/run_blurb_eval.py --reuse-index        # score the index built last time
    uv run python eval/run_blurb_eval.py --reuse-index --arms context-rerank   # resume one arm
    uv run python eval/run_blurb_eval.py --granularity chunk  # per-chunk contexts instead
    uv run python eval/run_blurb_eval.py --granularity chunk \
        --preview coffee/caffe-espresso-it.md                 # print one note's contexts only

`--granularity note` (the default) writes one blurb per note and prepends it
to every chunk; `--granularity chunk` asks the LLM once per chunk, each call
seeing the whole note (docs/design/2026-09-29-per-chunk-context.md). The two
keep separate indexes and log under different experiments and run ids.

Indexes the corpus once at the production chunking default with contextual
indexing on, aborts unless every chunk got a context, then scores:
  raw-rerank      arm 2  DENSE, SPARSE, HYBRID with the reranker on raw text
  fused           arm 2  FUSED (no reranker)
  context-rerank  arm 3  HYBRID with the reranker scoring blurb plus text
Arm 1 (no blurbs) is already logged; each entry names it as its control, and
the control must have been measured with the same chunking and embedding.
`--arms` picks a comma-separated subset; only the channels those arms need
are built. See docs/design/2026-09-29-blurb-aware-reranking.md.

Safety:
  - Before building, one short probe checks the blurb LLM answers; a dead
    endpoint aborts before the index is touched. The previous index is moved
    to <index>.prev.db, never deleted.
  - --reuse-index opens the kept index with the current config and brings it
    up to date (a no-op when nothing changed). The stored fingerprint is
    printed before and after, so an unexpected rebuild is visible, and logged.
  - If the reranker model fails to load, the product falls back to fused
    order with a warning; here that warning aborts any arm that reranks.

The blurb LLM comes from:
    ARIOSTEA_CTX_BASE_URL  (default http://localhost:1234/v1, LM Studio)
    ARIOSTEA_CTX_MODEL     (default qwen2.5-14b-instruct-mlx)
    ARIOSTEA_CTX_API_KEY   (default empty)
    ARIOSTEA_CTX_TIMEOUT   (seconds, default 300 for note, 900 for chunk:
                            whole articles, local model)
Load the model with a context window of at least 32k tokens first.

Blurbing takes 20 to 40 minutes and each HYBRID pass about an hour, so the
blurbed index is kept under eval/results/indexes/ for --reuse-index. The
blurbs themselves are written to eval/results/logs/<date>-blurbs.json.

Per-chunk contexts take about 15 hours, unattended. Load the model in LM
Studio with
    lms load qwen2.5-14b-instruct-mlx --context-length 32768 --parallel 1
(with more than one parallel slot the server stops reusing the note's
prompt prefix across its chunks, and every call pays for the whole note),
then check `lms ps` shows PARALLEL 1 and CONTEXT 32768 before starting. Then:
    nohup caffeinate -i uv run python eval/run_blurb_eval.py \
        --granularity chunk > eval/results/logs/chunk-context-run.out 2>&1 & disown
`caffeinate -i` keeps the Mac from sleeping mid-run; `nohup ... & disown`
keeps the run alive after the terminal closes. Each chunk call retries
transient failures on its own (`RetryingChat`), and a progress line prints
per note as chunk mode reaches it.
Every answer is cached in eval/results/indexes/chunk-context-cache.jsonl, so
after a crash rerun the same command: regenerated contexts come from the
cache for free, and a rerun without --reuse-index rebuilds the index from it.
Resuming on a later day must pass --arms naming only the arms not yet logged
today, or the run aborts on an arm already in runs.jsonl.
The contexts are written to eval/results/logs/<date>-chunk-contexts.json.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ariostea.adapters.chat.openai_compat import OpenAICompatChat
from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.config.container import (
    Container,
    _build_contextualizer,
    build_chunker,
    build_container,
)
from ariostea.config.schema import ChunkingCfg, ContextualCfg
from ariostea.eval.blurb_eval import (
    ARMS,
    HYBRID_CONTEXT,
    BlurbCoverageError,
    NoteProgressChat,
    WarnedError,
    arm_label,
    blurb_run_id,
    blurbs_by_note,
    contexts_by_chunk,
    control_mismatch,
    experiment_name,
    fail_on_warnings,
    index_marker,
    lmstudio_preflight,
    needed_channels,
    require_full_coverage,
    select_arms,
)
from ariostea.eval.chat_cache import CachingChat
from ariostea.eval.chunk_sweep import load_discriminated
from ariostea.eval.contextual import read_chunk_context_rows
from ariostea.eval.harness import SpanSearchFn
from ariostea.eval.results_log import (
    append_run,
    code_commit,
    load_runs,
    make_run,
    render_html,
    report_to_dict,
)
from ariostea.eval.retrying_chat import RetryingChat
from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import load_wiki_gold
from ariostea.eval.wiki_index import (
    CHUNK_POOL,
    MULTILINGUAL_MODEL,
    context_rerank_channel,
    fused_channel,
    wiki_channels,
    wiki_config,
)
from ariostea.mcp.handlers import reindex_payload
from ariostea.ports.chat import ChatProvider

EVAL = Path(__file__).resolve().parent
WIKI = EVAL / "wiki"
RESULTS = EVAL / "results"
RUNS = RESULTS / "runs.jsonl"
LOGS = RESULTS / "logs"
INDEXES = RESULTS / "indexes"
INDEX = {
    "note": INDEXES / "blurbs-160t+40.db",
    "chunk": INDEXES / "chunk-context-160t+40.db",
}
CHUNK_CACHE = INDEXES / "chunk-context-cache.jsonl"
K = 5
# `_build_reranker` warns here when it falls back to fused order.
CONTAINER_LOGGER = "ariostea.config.container"
PROBE_TIMEOUT = 30.0
# Whole articles go into every prompt.
MIN_CONTEXT = 32768


def _contextual(granularity: str) -> ContextualCfg:
    common = {
        "enabled": True,
        "base_url": os.environ.get("ARIOSTEA_CTX_BASE_URL", "http://localhost:1234/v1"),
        "model": os.environ.get("ARIOSTEA_CTX_MODEL", "qwen2.5-14b-instruct-mlx"),
        "api_key": os.environ.get("ARIOSTEA_CTX_API_KEY", ""),
    }
    if granularity == "chunk":
        # The first long-article call took 570 s; the pilot's contexts reached
        # 135 tokens.
        return ContextualCfg(
            **common,
            granularity="chunk",
            timeout=float(os.environ.get("ARIOSTEA_CTX_TIMEOUT", "900")),
            max_tokens=200,
        )
    return ContextualCfg(**common, timeout=float(os.environ.get("ARIOSTEA_CTX_TIMEOUT", "300")))


def _note_queue() -> list[tuple[str, int]]:
    """(note_path, chunk_count) for every note under WIKI, in `scan_vault`'s
    sort order -- the order `IndexVault.index` reaches them once the whole
    corpus is freshly indexed, which is what happens after `_move_aside`
    empties the store. Used only to label `NoteProgressChat`'s lines; a
    mismatch against the real run (e.g. a note with no chunks) only makes a
    progress line wrong, never the run."""
    parser = ObsidianMarkdownParser()
    chunker = build_chunker(ChunkingCfg(), FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL))
    queue: list[tuple[str, int]] = []
    for path in sorted(p.relative_to(WIKI).as_posix() for p in WIKI.rglob("*.md")):
        note, body = parser.parse(path, (WIKI / path).read_text(encoding="utf-8"), 0.0)
        chunks = chunker.chunk(note, body)
        if chunks:
            queue.append((path, len(chunks)))
    return queue


def _chat_cache(
    ctx: ContextualCfg,
) -> tuple[Callable[[ChatProvider], ChatProvider] | None, list[CachingChat]]:
    """A `wrap_chat` that retries transient failures, caches every per-chunk
    answer, and prints one progress line per note; and the list of caches it
    builds (for hit and miss counts). Note mode is neither cached nor
    progress-printed.

    The cache key is `label` (the model name) plus the exact `system`/`user`
    text; it does not include `max_tokens`, `temperature` or `base_url`, so a
    change to any of those over the model's answers is invisible to the
    cache -- change the model name (or clear the cache file) to force a
    re-blurb after such a change.
    """
    caches: list[CachingChat] = []
    if ctx.granularity != "chunk":
        return None, caches

    def wrap(chat: ChatProvider) -> ChatProvider:
        cache = CachingChat(RetryingChat(chat), CHUNK_CACHE, label=ctx.model)
        caches.append(cache)
        return NoteProgressChat(cache, _note_queue(), cache)

    return wrap, caches


def _report_cache(caches: list[CachingChat]) -> None:
    if caches:
        hits, misses = sum(c.hits for c in caches), sum(c.misses for c in caches)
        print(f"context cache: {hits} hits, {misses} misses ({CHUNK_CACHE})", flush=True)


def _preflight(ctx: ContextualCfg) -> str | None:
    """Why the blurb LLM cannot be used, or None when it is ready. Run before
    the index is touched: a dead or wrongly-loaded endpoint would otherwise
    only show up as a failed coverage gate after the whole corpus.

    Prefers LM Studio's native API, which reports the loaded context window
    and parallel-slot count the plain chat probe below cannot see: a model
    LM Studio auto-loads on first request gets its own defaults (8k context,
    several parallel slots), which a probe alone would not catch."""
    responded, info = _lmstudio_models(ctx)
    outcome = lmstudio_preflight(responded, info, ctx.model, MIN_CONTEXT)
    if outcome.abort:
        return (
            f"{outcome.abort}; load it first: lms load {ctx.model} "
            f"--context-length {MIN_CONTEXT} --parallel 1"
        )
    if outcome.fallback_to_probe:
        print(
            f"WARNING: could not verify the model's load state "
            f"({ctx.base_url} does not answer LM Studio's /api/v0/models); "
            "falling back to a plain chat probe",
            file=sys.stderr,
            flush=True,
        )
    else:
        return None
    chat = OpenAICompatChat(
        base_url=ctx.base_url,
        model=ctx.model,
        api_key=ctx.api_key,
        timeout=PROBE_TIMEOUT,
        max_tokens=ctx.max_tokens,
    )
    try:
        reply = chat.complete(system="Reply with the single word OK.", user="ping")
    except Exception as exc:  # ChatError, or anything else the probe trips on
        return f"blurb LLM {ctx.model} at {ctx.base_url} failed: {exc}"
    if not (reply or "").strip():
        return f"blurb LLM {ctx.model} at {ctx.base_url} returned empty text"
    return None


def _lmstudio_models(ctx: ContextualCfg) -> tuple[bool, dict | None]:
    """Whether LM Studio's `/api/v0/models` endpoint answered, and, when it
    did, its description of `ctx.model` from `/api/v0/models/<model>` (or None
    when the lookup failed or the model is not listed)."""
    if not ctx.base_url.rstrip("/").endswith("/v1"):
        return False, None
    root = ctx.base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/v0/models", timeout=PROBE_TIMEOUT):
            pass
    except OSError:  # not LM Studio, or down
        return False, None
    try:
        with urllib.request.urlopen(
            f"{root}/api/v0/models/{ctx.model}", timeout=PROBE_TIMEOUT
        ) as response:
            info = json.loads(response.read())
    except (OSError, ValueError):  # unknown model, or a bad response
        return True, None
    return True, (info if isinstance(info, dict) else None)


def _move_aside(index: Path) -> None:
    """Move `index` and its SQLite sidecars to `<stem>.prev.db`, replacing an
    older copy. A stale sidecar of the older copy is removed rather than left
    to pair with the new one."""
    if not index.exists():
        return
    prev = index.with_name(f"{index.stem}.prev{index.suffix}")
    for suffix in ("", "-wal", "-shm"):
        src, dst = Path(f"{index}{suffix}"), Path(f"{prev}{suffix}")
        if src.exists():
            src.replace(dst)
        else:
            dst.unlink(missing_ok=True)
    print(f"moved the previous index to {prev}", flush=True)


def _guard(reranks: bool) -> contextlib.AbstractContextManager[None]:
    """Abort on a reranker fallback while building a container that ranks with it."""
    return fail_on_warnings(CONTAINER_LOGGER) if reranks else contextlib.nullcontext()


def _open_index(
    ctx: ContextualCfg,
    index: Path,
    reranks: bool,
    wrap_chat: Callable[[ChatProvider], ChatProvider] | None,
) -> Container:
    with _guard(reranks):
        return build_container(wiki_config(WIKI, str(index), contextual=ctx), wrap_chat=wrap_chat)


def _index_up_to_date(container: Container) -> str:
    """Index (or refresh) the corpus and return the stored fingerprint,
    printing it before and after."""
    before = container.admin.stats().config_fingerprint
    print(f"  fingerprint before: {before or '(none)'}", flush=True)
    stats = reindex_payload(container)
    after = container.admin.stats().config_fingerprint
    print(f"  fingerprint after:  {after}", flush=True)
    if before and before != after:
        print("  NOTE: the fingerprint changed, so the whole index was rebuilt", flush=True)
    print(f"  {stats['notes']} notes, {stats['chunks']} chunks", flush=True)
    return after


def _build_channels(container: Container, index: Path, needed: set[str]) -> dict[str, SpanSearchFn]:
    """Only the channels in `needed`, built up front so a failure shows before
    hours of scoring rather than between arms."""
    builders: dict[str, Callable[[], SpanSearchFn]] = {
        "FUSED": lambda: fused_channel(container),
        HYBRID_CONTEXT: lambda: context_rerank_channel(container),
    }
    channels: dict[str, SpanSearchFn] = {}
    if needed & {"DENSE", "SPARSE", "HYBRID"}:
        # One call builds all three; HYBRID reuses `container`'s reranker,
        # which was checked when the container was built.
        base = wiki_channels(str(index), container)
        channels.update({name: fn for name, fn in base.items() if name in needed})
    for name in sorted(needed - channels.keys()):
        with _guard(name == HYBRID_CONTEXT):
            channels[name] = builders[name]()
    return channels


def _preview(ctx: ContextualCfg, notes: list[str]) -> int:
    """Contextualize only `notes` (through the cache in chunk mode) and print
    each chunk's context, without indexing or logging."""
    problem = _preflight(ctx)
    if problem:
        print(f"ABORT: {problem}", file=sys.stderr)
        return 1
    wrap, caches = _chat_cache(ctx)
    try:
        with fail_on_warnings(CONTAINER_LOGGER):  # a fallback to plain chunks
            contextualizer = _build_contextualizer(ctx, wrap_chat=wrap)
    except WarnedError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    parser = ObsidianMarkdownParser()
    chunker = build_chunker(ChunkingCfg(), FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL))
    for rel in notes:
        note, body = parser.parse(rel, (WIKI / rel).read_text(encoding="utf-8"), 0.0)
        chunks = chunker.chunk(note, body)
        print(f"\n##### {rel} ({len(chunks)} chunks)", flush=True)
        for cchunk in contextualizer.contextualize(note, body, chunks):
            print(f"{cchunk.chunk.ordinal}: {cchunk.context_blurb or '(no context)'}", flush=True)
    _report_cache(caches)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reuse-index", action="store_true", help="score the kept index")
    parser.add_argument(
        "--arms",
        default=",".join(arm.key for arm in ARMS),
        help=f"comma-separated subset of {', '.join(arm.key for arm in ARMS)} (default all)",
    )
    parser.add_argument(
        "--granularity",
        choices=("note", "chunk"),
        default="note",
        help="one blurb per note, or one context per chunk (default note)",
    )
    parser.add_argument(
        "--experiment",
        help="group name in the log (default: Contextual blurbs, or Per-chunk context)",
    )
    parser.add_argument("--no-log", action="store_true", help="print only, record nothing")
    parser.add_argument(
        "--preview",
        metavar="NOTE[,NOTE]",
        help="only print the contexts of these notes (paths under eval/wiki); no index, no log",
    )
    args = parser.parse_args(argv)
    try:
        arms = select_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))
    granularity = args.granularity
    ctx = _contextual(granularity)
    if args.preview is not None:
        notes = [n.strip() for n in args.preview.split(",") if n.strip()]
        if not notes:
            parser.error("--preview names no note")
        unknown = [n for n in notes if not (WIKI / n).is_file()]
        if unknown:
            parser.error(f"not a note under {WIKI}: {', '.join(unknown)}")
        return _preview(ctx, notes)
    experiment = args.experiment or experiment_name(granularity)
    index = INDEX[granularity]

    today = dt.date.today().isoformat()
    commit = code_commit()
    chunking = ChunkingCfg().model_dump()
    runs = {run["id"]: run for run in load_runs(RUNS)}
    if not args.no_log:
        for arm in arms:
            if arm.control not in runs:
                parser.error(f"control run {arm.control!r} is not in {RUNS}")
            mismatch = control_mismatch(runs[arm.control], chunking, MULTILINGUAL_MODEL)
            if mismatch:
                parser.error(mismatch)
            run_id = blurb_run_id(today, arm, granularity)
            if run_id in runs:
                parser.error(f"already logged today: {run_id}")

    needed = needed_channels(arms)
    wrap, caches = _chat_cache(ctx)
    try:
        if args.reuse_index:
            if not index.exists():
                parser.error(f"no kept index at {index}; run without --reuse-index first")
            print(f"reusing {index}", flush=True)
        else:
            problem = _preflight(ctx)
            if problem:
                print(f"ABORT: {problem}; the index was not touched", file=sys.stderr)
                return 1
            index.parent.mkdir(parents=True, exist_ok=True)
            _move_aside(index)
            print(f"indexing with {granularity} contexts from {ctx.model} ...", flush=True)
        # Only the plain HYBRID channel ranks with this container's reranker.
        container = _open_index(ctx, index, reranks="HYBRID" in needed, wrap_chat=wrap)
        if args.reuse_index:
            # Reuse skips the LLM probe, so it must not trigger a rebuild that
            # needs the LLM: a kept index blurbed by another model would be
            # re-blurbed in place, or flattened to plain chunks if the LLM is down.
            stored = container.admin.stats().config_fingerprint
            marker = index_marker(granularity, ctx.model)
            if marker not in stored.split("|"):
                print(
                    f"ABORT: the kept index was not contextualized as {marker} "
                    f"(fingerprint {stored!r}); rebuild without --reuse-index",
                    file=sys.stderr,
                )
                return 1
        fingerprint = _index_up_to_date(container)
        _report_cache(caches)

        chunk_rows = read_chunk_context_rows(str(index))
        rows = [(path, context) for path, _, context in chunk_rows]
        try:
            blurbed = require_full_coverage(rows, granularity)
        except BlurbCoverageError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 1
        print(f"context coverage {blurbed}/{blurbed} notes, {len(rows)} chunks", flush=True)
        if granularity == "chunk":
            dump_name, contexts = "chunk-contexts", contexts_by_chunk(chunk_rows)
        else:
            dump_name, contexts = "blurbs", blurbs_by_note(rows)
        if not args.no_log:
            LOGS.mkdir(parents=True, exist_ok=True)
            dump = LOGS / f"{today}-{dump_name}.json"
            dump.write_text(
                json.dumps(contexts, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {dump}", flush=True)

        channels = _build_channels(container, index, needed)
    except WarnedError as exc:  # e.g. the reranker fell back to fused order
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1

    cases = load_wiki_gold(WIKI / "gold.json")
    easy = load_discriminated(WIKI / "gold_rejected.json")
    rerank = container.config.rerank
    for arm in arms:
        lines: list[str] = []

        def say(text: str) -> None:
            print(text, flush=True)
            lines.append(text)

        label = arm_label(arm, granularity)
        say(f"\n##### {label}")
        scores: dict[str, dict] = {}
        dropped: dict[str, dict] = {}
        for name in arm.channels:
            fn = channels[arm.source(name)]
            say(f"  scoring {name} ({arm.source(name)}) ...")
            report = evaluate_spans(cases, fn, k=K, pool=CHUNK_POOL)
            scores[name] = report_to_dict(report)
            dropped[name] = report_to_dict(evaluate_spans(easy, fn, k=K, pool=CHUNK_POOL))[
                "overall"
            ]
            say(f"=== {name} ===\n{format_span_report(report)}")
            say(
                f"  discriminated-out cases ({len(easy)}): span recall "
                f"{dropped[name]['span_recall']:.3f}"
            )
        if args.no_log:
            continue

        run_id = blurb_run_id(today, arm, granularity)
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"{run_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run = make_run(
            run_id=run_id,
            experiment=experiment,
            label=label,
            date=today,
            commit=commit,
            control=arm.control,
            config={
                "k": K,
                "embedding": MULTILINGUAL_MODEL,
                "chunking": chunking,
                "contextual": {
                    "granularity": granularity,
                    "model": ctx.model,
                    "base_url": ctx.base_url,
                    "timeout": ctx.timeout,
                    "max_tokens": ctx.max_tokens,
                    "blurbed_notes": blurbed,
                    "contextualized_chunks": len(rows),
                },
                "index_fingerprint": fingerprint,
                "rerank_model": rerank.model,
                "rerank_pool": rerank.pool,
                "rerank_use_context": arm.use_context,
            },
            cases=len(cases),
            # Blurbs do not change chunking, so the control's ceiling holds
            # (control_mismatch checked the chunking matches).
            reachable=runs[arm.control]["reachable"],
            channels=scores,
            notes=f"Channels measured: {', '.join(arm.channels)}.",
        )
        run["discriminated"] = dropped
        append_run(RUNS, run)
        print(f"  logged {run_id}", flush=True)

    if not args.no_log:
        page = RESULTS / "experiment_log.html"
        template = (RESULTS / "template.html").read_text(encoding="utf-8")
        page.write_text(render_html(load_runs(RUNS), template), encoding="utf-8")
        print(f"\nrendered {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
