"""Measure note-level context blurbs, and blurb-aware reranking, on the wiki gold set.

Usage:
    uv run python eval/run_blurb_eval.py                      # build the blurbed index, score, log
    uv run python eval/run_blurb_eval.py --reuse-index        # score the index built last time
    uv run python eval/run_blurb_eval.py --reuse-index --arms context-rerank   # resume one arm

Indexes the corpus once at the production chunking default with contextual
indexing on, aborts unless every note got a blurb, then scores:
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
    to blurbs-160t+40.prev.db, never deleted.
  - --reuse-index opens the kept index with the current config and brings it
    up to date (a no-op when nothing changed). The stored fingerprint is
    printed before and after, so an unexpected rebuild is visible, and logged.
  - If the reranker model fails to load, the product falls back to fused
    order with a warning; here that warning aborts any arm that reranks.

The blurb LLM comes from:
    ARIOSTEA_CTX_BASE_URL  (default http://localhost:1234/v1, LM Studio)
    ARIOSTEA_CTX_MODEL     (default qwen2.5-14b-instruct-mlx)
    ARIOSTEA_CTX_API_KEY   (default empty)
    ARIOSTEA_CTX_TIMEOUT   (seconds, default 300: whole articles, local model)
Load the model with a context window of at least 32k tokens first.

Blurbing takes 20 to 40 minutes and each HYBRID pass about an hour, so the
blurbed index is kept under eval/results/indexes/ for --reuse-index. The
blurbs themselves are written to eval/results/logs/<date>-blurbs.json.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from ariostea.adapters.chat.openai_compat import OpenAICompatChat
from ariostea.config.container import Container, build_container
from ariostea.config.schema import ChunkingCfg, ContextualCfg
from ariostea.eval.blurb_eval import (
    ARMS,
    HYBRID_CONTEXT,
    BlurbCoverageError,
    WarnedError,
    blurb_run_id,
    blurbs_by_note,
    control_mismatch,
    fail_on_warnings,
    needed_channels,
    require_full_coverage,
    select_arms,
)
from ariostea.eval.chunk_sweep import load_discriminated
from ariostea.eval.contextual import read_blurb_rows
from ariostea.eval.harness import SpanSearchFn
from ariostea.eval.results_log import (
    append_run,
    code_commit,
    load_runs,
    make_run,
    render_html,
    report_to_dict,
)
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

EVAL = Path(__file__).resolve().parent
WIKI = EVAL / "wiki"
RESULTS = EVAL / "results"
RUNS = RESULTS / "runs.jsonl"
LOGS = RESULTS / "logs"
INDEX = RESULTS / "indexes" / "blurbs-160t+40.db"
K = 5
# `_build_reranker` warns here when it falls back to fused order.
CONTAINER_LOGGER = "ariostea.config.container"
PROBE_TIMEOUT = 30.0


def _contextual() -> ContextualCfg:
    return ContextualCfg(
        enabled=True,
        base_url=os.environ.get("ARIOSTEA_CTX_BASE_URL", "http://localhost:1234/v1"),
        model=os.environ.get("ARIOSTEA_CTX_MODEL", "qwen2.5-14b-instruct-mlx"),
        api_key=os.environ.get("ARIOSTEA_CTX_API_KEY", ""),
        timeout=float(os.environ.get("ARIOSTEA_CTX_TIMEOUT", "300")),
    )


def _preflight(ctx: ContextualCfg) -> str | None:
    """Why the blurb LLM cannot be used, or None when one short probe gets an
    answer. Run before the index is touched: a dead endpoint would otherwise
    only show up as a failed coverage gate after the whole corpus."""
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


def _open_index(ctx: ContextualCfg, reranks: bool) -> Container:
    with _guard(reranks):
        return build_container(wiki_config(WIKI, str(INDEX), contextual=ctx))


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


def _build_channels(container: Container, needed: set[str]) -> dict[str, SpanSearchFn]:
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
        base = wiki_channels(str(INDEX), container)
        channels.update({name: fn for name, fn in base.items() if name in needed})
    for name in sorted(needed - channels.keys()):
        with _guard(name == HYBRID_CONTEXT):
            channels[name] = builders[name]()
    return channels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reuse-index", action="store_true", help="score the kept index")
    parser.add_argument(
        "--arms",
        default=",".join(arm.key for arm in ARMS),
        help=f"comma-separated subset of {', '.join(arm.key for arm in ARMS)} (default all)",
    )
    parser.add_argument("--experiment", default="Contextual blurbs", help="group name in the log")
    parser.add_argument("--no-log", action="store_true", help="print only, record nothing")
    args = parser.parse_args(argv)
    try:
        arms = select_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))

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
            if blurb_run_id(today, arm) in runs:
                parser.error(f"already logged today: {blurb_run_id(today, arm)}")

    ctx = _contextual()
    needed = needed_channels(arms)
    try:
        if args.reuse_index:
            if not INDEX.exists():
                parser.error(f"no kept index at {INDEX}; run without --reuse-index first")
            print(f"reusing {INDEX}", flush=True)
        else:
            problem = _preflight(ctx)
            if problem:
                print(f"ABORT: {problem}; the index was not touched", file=sys.stderr)
                return 1
            INDEX.parent.mkdir(parents=True, exist_ok=True)
            _move_aside(INDEX)
            print(f"indexing with blurbs from {ctx.model} ...", flush=True)
        # Only the plain HYBRID channel ranks with this container's reranker.
        container = _open_index(ctx, reranks="HYBRID" in needed)
        if args.reuse_index:
            # Reuse skips the LLM probe, so it must not trigger a rebuild that
            # needs the LLM: a kept index blurbed by another model would be
            # re-blurbed in place, or flattened to plain chunks if the LLM is down.
            stored = container.admin.stats().config_fingerprint
            if f"llm:{ctx.model}" not in stored.split("|"):
                print(
                    f"ABORT: the kept index was not blurbed by {ctx.model} "
                    f"(fingerprint {stored!r}); rebuild without --reuse-index",
                    file=sys.stderr,
                )
                return 1
        fingerprint = _index_up_to_date(container)

        try:
            rows = read_blurb_rows(str(INDEX))
            blurbed = require_full_coverage(rows)
        except BlurbCoverageError as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 1
        print(f"blurb coverage {blurbed}/{blurbed} notes", flush=True)
        blurbs = blurbs_by_note(rows)
        if not args.no_log:
            LOGS.mkdir(parents=True, exist_ok=True)
            dump = LOGS / f"{today}-blurbs.json"
            dump.write_text(
                json.dumps(blurbs, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {dump}", flush=True)

        channels = _build_channels(container, needed)
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

        say(f"\n##### {arm.label}")
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

        run_id = blurb_run_id(today, arm)
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"{run_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run = make_run(
            run_id=run_id,
            experiment=args.experiment,
            label=arm.label,
            date=today,
            commit=commit,
            control=arm.control,
            config={
                "k": K,
                "embedding": MULTILINGUAL_MODEL,
                "chunking": chunking,
                "contextual": {
                    "model": ctx.model,
                    "base_url": ctx.base_url,
                    "timeout": ctx.timeout,
                    "max_tokens": ctx.max_tokens,
                    "blurbed_notes": blurbed,
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
