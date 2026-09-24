"""Evaluate retrieval against the span-anchored Wikipedia gold set.

Usage:  uv run python eval/run_wiki_eval.py [k]

Indexes eval/wiki/ into a throwaway database, then prints, per retrieval
channel, note-level and span-level recall@k / MRR / nDCG broken down by query
type -- plus the per-cluster dense-only difficulty guard.

The per-type breakdown is the point. An aggregate number cannot tell you that
a tokenizer change helped `exact_term` and hurt nothing else, and that
attribution is the whole reason this corpus exists.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ariostea.eval.difficulty import (
    EASY_THRESHOLD,
    cluster_baselines,
    flag_easy_clusters,
    format_baselines,
)
from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import load_wiki_gold, validate_wiki_gold
from ariostea.eval.wiki_index import CHUNK_POOL, index_wiki_corpus, wiki_channels
from ariostea.eval.wiki_notes import load_corpus_notes

WIKI_DIR = Path(__file__).resolve().parent / "wiki"
GOLD = WIKI_DIR / "gold.json"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    k = int(argv[0]) if argv else 5

    if not GOLD.exists():
        print(
            f"no gold set at {GOLD}; run `uv run python eval/generate_gold.py` first",
            file=sys.stderr,
        )
        return 2

    notes = load_corpus_notes(WIKI_DIR)
    cases = load_wiki_gold(GOLD)
    if not cases:
        # Zeros across the board read as a broken retrieval stack rather than
        # as a generation run that produced nothing.
        print(f"gold set at {GOLD} has no cases; nothing to evaluate", file=sys.stderr)
        return 2

    errors = validate_wiki_gold(cases, notes)
    if errors:
        # Refuse to score gold the corpus no longer supports: a span that has
        # drifted out of its note scores zero on every channel, which reads as
        # a retrieval regression rather than as stale data.
        print(f"gold set does not match the corpus ({len(errors)} problems):", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        return 3

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "eval.db")
        print(f"indexing {len(notes)} notes, evaluating {len(cases)} cases at k={k} ...")
        container = index_wiki_corpus(WIKI_DIR, db)
        channels = wiki_channels(db, container)

        for label, span_fn in channels.items():
            print(f"\n=== {label} ===")
            print(format_span_report(evaluate_spans(cases, span_fn, k=k, pool=CHUNK_POOL)))

        print(f"\n=== difficulty guard (dense-only baseline, k={k}) ===")
        baselines = cluster_baselines(cases, channels["DENSE"], k=k, pool=CHUNK_POOL)
        print(format_baselines(baselines, threshold=EASY_THRESHOLD))
        easy = flag_easy_clusters(baselines, threshold=EASY_THRESHOLD)
        if easy:
            print(
                f"\n{len(easy)} cluster(s) at or above {EASY_THRESHOLD}: {', '.join(easy)}. "
                f"No experiment can show an improvement there -- densify the cluster or "
                f"regenerate its queries."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
