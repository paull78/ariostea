# Issues

Open issues found during the cycle, one entry each, updated at every gate.

### server-robustness.14 · issue · 2026-10-27 · step 08
On an index migrated from the previous release, `status.config_fingerprint` and `last_indexed` are null until the first run under this release completes, since the migration keeps no run record. `eval/run_blurb_eval.py --reuse-index` reads that fingerprint, so on such an index it aborts with its "rebuild without --reuse-index" message until one `reindex` has run (which embeds nothing if the settings are unchanged).
**Why:** the spec says a migrated index has no completed run yet (AC-09, AC-10: null until one completes). Accepted: one `reindex` clears it; synthesising a run record at migration would report a finish time that never happened. Open, low.
