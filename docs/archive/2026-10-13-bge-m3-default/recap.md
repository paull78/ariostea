# Recap: bge-m3-default

Feature cycle on `feat/bge-m3-default`, started and finished on 2026-10-13. Follows the embedding-sweep cycle, which chose bge-m3 (embedding-sweep.19).

## What was built

The default embedding model is now `BAAI/bge-m3`, with chunks of 256 model tokens and overlap 64. On the wiki gold set that configuration finds 157 of 167 cases on hybrid, against 147 for the previous defaults. The eval pins the same model, so a plain eval run measures what ships (bge-m3-default.03).

For existing vaults, the switch needed these fixes:

- The store rebuilds its vector table when the configured size differs from the one on disk, and clears the fingerprint so every note re-embeds (bge-m3-default.01). Before, an index built at 768 dimensions rejected the first 1024-dimension vector.
- The store adds `chunks` columns that an older index lacks (bge-m3-default.09). Before, an index built before the blurb work failed on `context_blurb`.

Both fastembed adapters now keep their models under `~/.cache/fastembed`, or `FASTEMBED_CACHE_PATH`, instead of the system temp directory macOS purges. That closes embedding-sweep.10.

The README has a fourth learning-journey trial, a section on the current defaults (renamed from "Where it stands", bge-m3-default.12) and an upgrade note: reindex once after a change of model or chunking. `docs/retrieval-tuning.md` has a section on the switch, and the log artifact is at version 10.

## How it went

Three milestones, all gates passed on 2026-10-13. Milestone 1 made indexes survive a change of vector size and moved the model cache. Milestone 2 measured the chunk size and switched the defaults. Milestone 3 wrote the docs.

The spec already carried one migration bug, found while reading the store before the first question: the vector table keeps its size forever (bge-m3-default.01). The second one surfaced only at the milestone 2 gate, when the owner's real vault was reindexed on the new default. The index predated the blurb column and also had no FTS or meta table; the schema code creates missing tables but never adds a column, so the reindex failed (bge-m3-default.09). Step 01's rebuild had already run, so the owner's config and index were restored from scratchpad backups. Step 07 added the column migration, a rehearsal on a copy of the index passed, and the real reindex then indexed 88 notes into 952 chunks (bge-m3-default.10, .11). The old index had held 59 notes, so it was out of date.

The chunk-size measurement took one HYBRID run: 157 hits at 256t+64 against 155 at 160t+40, with buried and paraphrase each one case better and nothing worse (bge-m3-default.06).

## Decisions that stand

- bge-m3-default.02 · Adopt 256t+64 if HYBRID finds at least 155 of 167 with no type down more than 2 cases · the model switch already reindexes every vault, so a larger chunk costs users nothing extra.
- bge-m3-default.03 · The eval's pinned model is bge-m3 · a plain eval run measures the configuration users get.
- bge-m3-default.06 · Default chunking is 256 model tokens with overlap 64 · 157 against 155 hits, MRR 0.867 against 0.846.
- bge-m3-default.12 · README section "Where it stands" renamed "The defaults today" · the old name is a stock status heading.

## Discoveries

- sqlite-vec fixes a vector table's size at creation; the store reads it back from the table's DDL, since an empty table has no rows to measure (bge-m3-default.01, .04).
- The store drops a mismatched vector table when it opens, so `serve` and `status` also trigger it. A server on the new model answers from BM25 alone until `reindex` runs (bge-m3-default.07).
- The full suite now takes about 23 minutes against 12, because the integration tests embed with the larger default (bge-m3-default.08).
- The chunker records 254 tokens for the 256-token policy, since two go to the model's start and end markers (bge-m3-default.10).

## Open issues and known limitations

- A server started on a new model before any reindex serves BM25 only, silently. Reindexing automatically on `serve` when the fingerprint is empty was left out of scope (bge-m3-default.07).
- Old model copies remain in fastembed's temp cache, about 2.3 GB for bge-m3 alone; nothing reads them any more.
- The column migration lists one column. A future column on `chunks` has to be added to that list, or old indexes break again.
- From the previous cycle, still open: with bge-m3, fusing BM25 lowers the first stage on cross-lingual queries, and 10 of 167 cases are still missed.

## Takeaways

The hands-on gate on the owner's real vault found the bug every test had missed, because no test had built an index from an earlier release. A fixture copied from a real old database closed that gap.

Taking backups of the owner's config and index before running the check turned a failed migration into a two-command restore.
