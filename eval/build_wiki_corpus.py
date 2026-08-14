# eval/build_wiki_corpus.py
"""Fetch the pinned Wikipedia snapshot for the eval corpus and convert it to
Obsidian-flavored Markdown under eval/wiki/<cluster>/.

Usage:  uv run python eval/build_wiki_corpus.py [--cluster NAME] [--refresh]

Reads eval/wiki/clusters.json. Articles that already carry a `revid` are
fetched at that exact revision, so a re-run reproduces the committed snapshot;
--refresh re-pins every article to the current latest revision. The manifest is
written back with whatever revisions were pinned, and the attribution table in
eval/wiki/NOTICE is regenerated from the files on disk.

Politeness and abort policy: every fetch attempt (success or failure) is
followed by exactly one delay before the next request, `DELAY_S` normally, or
the longer `RETRYABLE_BACKOFF_S` when the attempt just failed in a way
`wiki_fetch.WikiFetchError.retryable` marks transient (rate-limited, blocked,
or a transport error) -- the delay always runs via a `finally`, specifically
so a retryable failure never leads into a burst of immediate retries against
an endpoint that just asked us to slow down. `MAX_CONSECUTIVE_RETRYABLE`
retryable failures in a row (reset by any success or permanent failure) abort
the run outright rather than grinding through the rest of a ~75-article batch
against what is probably a block, not noise.

An abort does not roll anything back: whatever notes were already fetched and
written stay on disk, the manifest and NOTICE are still saved/regenerated
exactly as they would be on a clean run, and only the articles the run never
got to keep their pre-run manifest entry (unpinned, or pinned to whatever
revid they were pinned to before this run started). That keeps disk, manifest,
and NOTICE mutually consistent for whatever fraction of the corpus was
actually built -- a stale NOTICE describing files that were never written, or
files on disk with no manifest/NOTICE record of them, would be worse than a
run that stops early but leaves a coherent partial result.

Dropped-template report: every article's conversion accumulates, into one
Counter shared across the whole run, the name of every template
`wikitext.expand_templates` left for `strip_templates` to remove whole. Most
of that tally is citation/navigation chrome (cite web, isbn, main, ...) and
is expected -- printed at the end, frequency-sorted, specifically so a name
that *isn't* chrome (a display template not yet on `wikitext.py`'s
allowlist) is a visible line in the run's output instead of something that
has to be noticed by reading 79 converted files by eye.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
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
# Applied instead of DELAY_S after a retryable failure. Simple fixed backoff,
# not exponential -- MAX_CONSECUTIVE_RETRYABLE aborts the run long before a
# smarter schedule would matter.
RETRYABLE_BACKOFF_S = 5.0
# Wikimedia telling us to slow down (429/503/blocked) rarely clears up inside
# one batch run; grinding through ~70 more doomed requests just to log ~70
# more identical failures wastes the run and is exactly the discourteous
# batch behavior Wikimedia's API etiquette asks clients to avoid.
MAX_CONSECUTIVE_RETRYABLE = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", help="build only this cluster")
    parser.add_argument(
        "--refresh", action="store_true", help="re-pin every article to the latest revision"
    )
    args = parser.parse_args()

    clusters = load_manifest(MANIFEST)
    langs = {a.lang for c in clusters for a in c.articles}
    targets = {lang: link_targets(clusters, lang) for lang in langs}

    built: list[Cluster] = []
    failures: list[str] = []
    consecutive_retryable = 0
    aborted = False
    dropped_templates: Counter[str] = Counter()

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for cluster in clusters:
            if aborted or (args.cluster and cluster.name != args.cluster):
                built.append(cluster)
                continue
            print(f"\n=== {cluster.name} ===")
            articles: list[ArticleSpec] = []
            for article in cluster.articles:
                if aborted:
                    articles.append(article)
                    continue
                try:
                    try:
                        fetched = fetch_article(
                            client,
                            lang=article.lang,
                            title=article.title,
                            revid=None if args.refresh else article.revid,
                        )
                    except WikiFetchError as exc:
                        failures.append(f"{cluster.name}/{article.title}: {exc}")
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
                                    "in a row (rate-limited or blocked)",
                                    file=sys.stderr,
                                )
                                aborted = True
                        else:
                            consecutive_retryable = 0
                            print(f"  skipped (permanent failure): {exc}", file=sys.stderr)
                        continue

                    consecutive_retryable = 0
                    if fetched.title != article.title:
                        print(f"  note: {article.title!r} redirects to {fetched.title!r}")
                    pinned = ArticleSpec(
                        title=article.title, lang=article.lang, revid=fetched.revid
                    )
                    body = wikitext_to_markdown(
                        fetched.wikitext,
                        title=article.title,
                        targets=targets[article.lang],
                        dropped=dropped_templates,
                    )
                    path = WIKI_DIR / note_path(cluster.name, pinned)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(render_note(cluster.name, pinned, body), encoding="utf-8")
                    articles.append(pinned)
                    print(
                        f"  {note_path(cluster.name, pinned)}  "
                        f"rev {fetched.revid}  {len(body):,} chars"
                    )
                finally:
                    # Always runs, including on the retryable-failure `continue`
                    # above -- that is the path that most needs the delay, and
                    # a `finally` is what keeps it from being skipped.
                    time.sleep(RETRYABLE_BACKOFF_S if consecutive_retryable else DELAY_S)
            built.append(Cluster(name=cluster.name, articles=tuple(articles)))

    save_manifest(MANIFEST, tuple(built))
    rows = [
        notice_row(c.name, a)
        for c in built
        for a in c.articles
        if a.revid is not None and (WIKI_DIR / note_path(c.name, a)).exists()
    ]
    NOTICE.write_text(write_notice_rows(NOTICE.read_text(encoding="utf-8"), rows), encoding="utf-8")
    print(f"\n{len(rows)} articles in the snapshot; NOTICE updated.")

    if dropped_templates:
        print(f"\n{sum(dropped_templates.values())} template invocations stripped, by name:")
        for name, count in dropped_templates.most_common():
            print(f"  {count:5d}  {name}")

    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    if aborted:
        print("ABORTED: run stopped early after repeated retryable failures.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
