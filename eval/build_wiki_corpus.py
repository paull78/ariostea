"""Fetch the pinned Wikipedia snapshot for the eval corpus and convert it to
Obsidian-flavored Markdown under eval/wiki/<cluster>/.

Usage:  uv run python eval/build_wiki_corpus.py [--cluster NAME] [--refresh]

Reads eval/wiki/clusters.json. Articles that already carry a `revid` are
fetched at that exact revision, so a re-run reproduces the committed snapshot;
--refresh re-pins every article to the current latest revision. The manifest is
written back with whatever revisions were pinned, and the attribution table in
eval/wiki/NOTICE is regenerated from the files on disk.

Politeness and abort policy: every fetch attempt (success or failure) is
followed by exactly one delay before the next request, `DELAY_S` normally, or a
growing backoff when the attempt just failed in a way
`wiki_fetch.WikiFetchError.retryable` marks transient (rate-limited, blocked,
or a transport error) -- the delay always runs via a `finally`, specifically so
a retryable failure never leads into a burst of immediate retries against an
endpoint that just asked us to slow down. The backoff grows geometrically
(5s, 15s, 45s) because Wikimedia's throttle, once tripped, does not clear in
the handful of seconds a fixed delay would spend; `MAX_CONSECUTIVE_RETRYABLE`
failures in a row (reset by any success or permanent failure) then abort the
run rather than grinding through the rest of a ~79-article batch against what
is probably a block, not noise.

An abort does not roll anything back, and neither does an unexpected crash:
the manifest and NOTICE are written in a `finally`, so whatever notes reached
disk are always described by both. Only the articles the run never got to keep
their pre-run manifest entry. Files on disk with no manifest or NOTICE record
of them would be the worst outcome available -- it is the state that silently
breaks reproducibility, since the next run re-pins those articles to newer
revisions.

Dropped-template report: every article's conversion accumulates, into one
Counter shared across the whole run, the name of every template
`wikitext.expand_templates` left for `strip_templates` to remove whole, plus
every argument a handler had to discard. Most of that tally is
citation/navigation chrome (cite web, isbn, main, ...) and is expected. The
prefixed entries are not: BLOCKED means an allowlisted template's expansion
was prevented by an unallowlisted one nested inside it, UNKNOWN-* means a
handler met a shape it could not render, EMPTY-EXPANSION means a handler
turned a template with arguments into nothing. Those are printed in their own
section, because a corpus this size cannot be proofread by eye and every one
of them is text that silently left the article.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ariostea.eval.wiki_corpus import notice_row, render_note, write_notice_rows
from ariostea.eval.wiki_fetch import WikiFetchError, fetch_article
from ariostea.eval.wiki_manifest import (
    ArticleSpec,
    Cluster,
    link_targets,
    load_manifest,
    note_path,
    save_manifest,
)
from ariostea.eval.wikitext import wikitext_to_markdown

WIKI_DIR = Path(__file__).resolve().parent / "wiki"
MANIFEST = WIKI_DIR / "clusters.json"
NOTICE = WIKI_DIR / "NOTICE"
# Wikimedia asks for serial, unhurried API access from batch clients.
DELAY_S = 0.2
# First backoff after a retryable failure; each further one triples it.
RETRYABLE_BACKOFF_S = 5.0
MAX_CONSECUTIVE_RETRYABLE = 3
# Report lines a human has to look at, as opposed to expected chrome.
REPORT_PREFIXES = ("BLOCKED:", "UNKNOWN-", "EMPTY-EXPANSION:")


def backoff_delay(consecutive_retryable: int) -> float:
    """Seconds to wait after `consecutive_retryable` failures in a row."""
    if consecutive_retryable <= 0:
        return DELAY_S
    return RETRYABLE_BACKOFF_S * 3 ** (consecutive_retryable - 1)


@dataclass
class BuildResult:
    clusters: tuple[Cluster, ...] = ()
    failures: list[str] = field(default_factory=list)
    dropped_templates: Counter[str] = field(default_factory=Counter)
    aborted: bool = False


def fetch_and_write(
    client: httpx.Client,
    cluster: str,
    article: ArticleSpec,
    targets: dict[str, str],
    wiki_dir: Path,
    dropped: Counter[str],
    refresh: bool = False,
) -> ArticleSpec:
    """Fetch one article, convert it, write its note; return the pinned spec.

    Raises `WikiFetchError` — the caller owns the retry/abort policy, so this
    function stays a straight line from a manifest entry to a file on disk.
    """
    fetched = fetch_article(
        client,
        lang=article.lang,
        title=article.title,
        revid=None if refresh else article.revid,
    )
    if fetched.title != article.title:
        # The manifest title stays authoritative: it is what the slug, the
        # note path and the NOTICE row were all derived from.
        print(f"  note: {article.title!r} redirects to {fetched.title!r}")
    pinned = ArticleSpec(title=article.title, lang=article.lang, revid=fetched.revid)
    body = wikitext_to_markdown(
        fetched.wikitext, title=article.title, targets=targets[article.lang], dropped=dropped
    )
    path = wiki_dir / note_path(cluster, pinned)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_note(cluster, pinned, body), encoding="utf-8")
    print(f"  {note_path(cluster, pinned)}  rev {fetched.revid}  {len(body):,} chars")
    return pinned


def build(
    client: httpx.Client,
    clusters: tuple[Cluster, ...],
    wiki_dir: Path,
    only_cluster: str | None = None,
    refresh: bool = False,
    sleep: object = time.sleep,
) -> BuildResult:
    """Fetch and convert every requested article, honouring the abort policy.

    `sleep` is injectable purely so the abort policy can be tested without a
    test suite that takes a minute to run.
    """
    langs = {a.lang for c in clusters for a in c.articles}
    targets = {lang: link_targets(clusters, lang) for lang in langs}
    result = BuildResult()
    built: list[Cluster] = []
    consecutive_retryable = 0

    for cluster in clusters:
        if result.aborted or (only_cluster and cluster.name != only_cluster):
            built.append(cluster)
            continue
        print(f"\n=== {cluster.name} ===")
        articles: list[ArticleSpec] = []
        for article in cluster.articles:
            if result.aborted:
                articles.append(article)
                continue
            try:
                try:
                    articles.append(
                        fetch_and_write(
                            client,
                            cluster.name,
                            article,
                            targets,
                            wiki_dir,
                            result.dropped_templates,
                            refresh=refresh,
                        )
                    )
                    consecutive_retryable = 0
                except WikiFetchError as exc:
                    result.failures.append(f"{cluster.name}/{article.title}: {exc}")
                    articles.append(article)
                    if exc.retryable:
                        consecutive_retryable += 1
                        print(
                            f"  retryable failure "
                            f"({consecutive_retryable}/{MAX_CONSECUTIVE_RETRYABLE}): {exc}",
                            file=sys.stderr,
                        )
                        if consecutive_retryable >= MAX_CONSECUTIVE_RETRYABLE:
                            print(
                                "  aborting: too many consecutive retryable failures "
                                "(rate-limited or blocked)",
                                file=sys.stderr,
                            )
                            result.aborted = True
                    else:
                        consecutive_retryable = 0
                        print(f"  skipped (permanent failure): {exc}", file=sys.stderr)
            finally:
                # Always runs, including on the failure path above -- that is
                # the path that most needs the delay.
                sleep(backoff_delay(consecutive_retryable))  # type: ignore[operator]
        built.append(Cluster(name=cluster.name, articles=tuple(articles)))

    result.clusters = tuple(built)
    return result


def reconcile(clusters: tuple[Cluster, ...], wiki_dir: Path, manifest: Path, notice: Path) -> int:
    """Write the manifest and regenerate NOTICE from the notes on disk."""
    save_manifest(manifest, clusters)
    rows = [
        notice_row(c.name, a)
        for c in clusters
        for a in c.articles
        if a.revid is not None and (wiki_dir / note_path(c.name, a)).exists()
    ]
    notice.write_text(write_notice_rows(notice.read_text(encoding="utf-8"), rows), encoding="utf-8")
    return len(rows)


def print_report(dropped: Counter[str]) -> None:
    flagged = {n: c for n, c in dropped.items() if n.startswith(REPORT_PREFIXES)}
    chrome = {n: c for n, c in dropped.items() if n not in flagged}
    if chrome:
        print(f"\n{sum(chrome.values())} template invocations removed as chrome, by name:")
        for name, count in Counter(chrome).most_common():
            print(f"  {count:5d}  {name}")
    if flagged:
        print("\nTEXT LOST TO UNHANDLED MARKUP -- each of these is prose that left an article:")
        for name, count in Counter(flagged).most_common():
            print(f"  {count:5d}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", help="build only this cluster")
    parser.add_argument(
        "--refresh", action="store_true", help="re-pin every article to the latest revision"
    )
    args = parser.parse_args()

    clusters = load_manifest(MANIFEST)
    known = {c.name for c in clusters}
    if args.cluster and args.cluster not in known:
        # Without this, a typo builds nothing, rewrites both files, and exits 0.
        print(
            f"unknown cluster {args.cluster!r}; manifest has {', '.join(sorted(known))}",
            file=sys.stderr,
        )
        return 2

    result = BuildResult(clusters=clusters)
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            result = build(
                client, clusters, WIKI_DIR, only_cluster=args.cluster, refresh=args.refresh
            )
    finally:
        # In a `finally` so a crash mid-run still leaves the manifest and
        # NOTICE describing whatever actually reached disk.
        written = reconcile(result.clusters, WIKI_DIR, MANIFEST, NOTICE)
        print(f"\n{written} articles in the snapshot; NOTICE updated.")
        print_report(result.dropped_templates)

    for failure in result.failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    if result.aborted:
        print("ABORTED: run stopped early after repeated retryable failures.", file=sys.stderr)
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
