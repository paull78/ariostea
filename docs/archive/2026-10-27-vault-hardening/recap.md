# Recap: vault-hardening

Feature cycle on `feature/vault-hardening`, started on 2026-10-26 and finished on 2026-10-27. Source: the usability assessment of 2026-10-26, a hands-on test on a 54-note synthetic vault plus a code audit.

## What was built

Ariostea can now index a vault that is not the owner's without losing the index, stopping on one bad file, or spending tens of gigabytes on one note. All nine fixes in the spec were built (vault-hardening.01):

- A missing vault path, a path that is a file, or a scan that finds no notes while the index holds some stops the run with an error naming the path, and the index is left alone (vault-hardening.02). In `watch`, the same failure on a later pass prints a warning and the watcher carries on (vault-hardening.03, .18).
- A file that is not UTF-8, or cannot be read, is skipped with a warning naming it. The summary counts it, the MCP `reindex` result lists it, and an entry indexed earlier keeps its chunks (vault-hardening.04). A leading BOM is dropped, so frontmatter after one is recognised.
- A word costlier than the chunk budget is cut into pieces that fit, so no chunk exceeds `chunking.max_tokens`. A note with a 20,000-character base64 image now indexes into 61 chunks of at most 254 tokens, with 211 MB of memory growth against the 20 to 25 GB the hands-on test saw (vault-hardening.05, .26).
- Paths with a dot segment at any depth and `*.excalidraw.md` drawings are always skipped; `vault.ignore` only adds prefixes (vault-hardening.06). `watch` drops `.obsidian/` churn.
- Lines inside fenced code blocks are never headings, by the CommonMark fence rule, and the note title follows the same rule (vault-hardening.07, .27).
- `get_note` checks the path against the index, then returns the file as written, frontmatter included (vault-hardening.09, .28).
- The unused `openai_compat` embedding fields are gone; a config that still sets one fails with a message naming the key. Every known CLI error is one line on stderr with exit status 1 (vault-hardening.10).
- `reindex` and the first pass of `watch` print a progress line every 100 notes or 10 seconds, and `watch` prints one line for each pass that changed something (vault-hardening.16).
- The README covers the 3.2 GB first-run download, running `reindex` from a terminal before connecting a client, the config lookup, what gets skipped, and the measured memory and search time (vault-hardening.12, .29).

The vault now sits behind a `Vault` port with a filesystem adapter and an in-memory one, which share one note rule and one contract suite. `IndexVault`, `WatchVault` and the new `GetNote` use case read notes only through it, and a test fails on any import of an adapter or vendor library from the inside of the app (vault-hardening.13, .15, .17, .19). The hands-on test's awkward vault is a permanent end-to-end test that drives the real CLI and stdio server (vault-hardening.14, .31).

Deviations from the spec:

- The README gives about 5 GB and about 2.5 s per search, not the spec's 4.3 GB: the server grows past that after a few searches. It also gives the figures with the reranker off, 1.7 GB and about 20 ms (vault-hardening.29).
- The awkward vault has 55 notes: a web clip with an inline base64 image was added, since the original generator had no long unbroken string outside the Excalidraw drawing the vault now skips (vault-hardening.22).
- The end-to-end test leaves out two of the hands-on test's queries: the Latin-1 note is now skipped by design, and symlinked folders are a non-goal.
- `Vault.changes()` passes every path that may hold notes, folders and the vault root included, not only note paths, because a folder moved to `.trash` or a vanished vault is reported only by its own path (vault-hardening.24).

## How it went

Four milestones. Milestone 1 (the index cannot be lost), Milestone 2 (watch and progress) and Milestone 3 (correct content) passed their gates on 2026-10-26; Milestone 4 (ready for a new user) passed the final gate on 2026-10-27. The full suite at the final gate: 1,258 passed and 1 skipped, in 25 minutes.

The design moved twice before the build started. The spec's first progress design had a refreshing line in a terminal and plain lines elsewhere. The challenge round cut it to one plain mode, with the clock in the CLI's printer and the use case reporting plain values (vault-hardening.11, superseded by .16). The same round moved change notification behind the `Vault` port, so `WatchVault` no longer imports `watchfiles` (vault-hardening.15). Probing `watchfiles` on this Mac then showed that a folder move reports only the folder, so the change filter had to pass folders and the root through (vault-hardening.24).

The Milestone 1 gate found two things outside the plan. A test built its reference embedder in fastembed's temp cache, which macOS had purged, and failed on a Hugging Face 401; the fix was one argument (vault-hardening.23). And the awkward-vault generator had no note that exercised the oversized-word path end to end, so a plan update added the base64 web clip to step 10 (vault-hardening.22).

Several tests were born green, because they pin behaviour that already held or that earlier steps had built. Each was shown able to fail. The end-to-end test failed once with the dot rule disabled by hand, and once on a genuine fixture bug: porting the generator had indented the text inside its strings, so frontmatter no longer opened the file (vault-hardening.25, .26, .27, .31).

At the final gate, CI's ruff checks failed on the hands-on scripts kept in `docs/current/assets/`. Ruff now skips `docs/`, so the archived scripts stay as they ran (vault-hardening.30).

## Decisions that stand

- vault-hardening.01 · Nine fixes from the assessment, server issues deferred · covers every way a generic vault loses its index, stalls the machine or fills with junk, at half the size of fixing everything.
- vault-hardening.02 · Two guards: the path must be a folder, and a scan that finds no notes cannot empty a non-empty index · covers the iCloud case without a ratio threshold.
- vault-hardening.03 · `watch` exits on a guard failure at startup, warns and carries on afterwards · a brief unmount should not kill an unattended watcher.
- vault-hardening.04 · UTF-8 only, BOM dropped; unreadable files are skipped and count as present · matches what Obsidian shows, and a sync hiccup never deletes chunks.
- vault-hardening.05 · An oversized word is cut to fit, and windows always move forward · one rule with no threshold bounds memory for any input.
- vault-hardening.06 · Dot segments and Excalidraw are always skipped; `vault.ignore` only adds prefixes · matches what Obsidian hides, and a user list cannot drop the defaults.
- vault-hardening.07 · Code fences follow CommonMark · the rule the Obsidian editor highlights by.
- vault-hardening.08 · The chunker fingerprint is unchanged; affected notes re-chunk when edited · the owner chose no forced reindex.
- vault-hardening.09 · `get_note` matches the index exactly, then returns the file as written · only the file has the note whole; the index match keeps traversal safety.
- vault-hardening.10 · `openai_compat` removed, a leftover key fails loudly; CLI errors are one line · the setting never worked, and the user should learn so.
- vault-hardening.13 · The vault is a driven actor behind a `Vault` port · indexing, `get_note` and `watch` cannot disagree on what a note is.
- vault-hardening.14 · The awkward vault is a permanent end-to-end fixture · it found every finding of the cycle.
- vault-hardening.15 · Change notification sits behind the port · no vendor library inside a use case.
- vault-hardening.16 · Progress is one plain mode; the CLI owns the clock · one behaviour to build and test, and time stays outside the app.
- vault-hardening.17 · An `ast` test enforces the dependency rule · a boundary nothing checks erodes.
- vault-hardening.19 · The note rule is one function on the adapter side, shared by both adapters · the fake cannot drift from the real one.
- vault-hardening.24 · `Vault.changes()` passes anything that may hold notes, folders and root included · folder moves and a vanished vault are reported only by their own path.
- vault-hardening.28 · `GetNote` is the use case behind `get_note`; `DocumentReader.read_note` and `NoteDocument` are removed · the Container exposes use cases, and the old code was dead.
- vault-hardening.30 · Ruff skips `docs/` · archived scripts stay as they ran, and CI stays green.

## Discoveries

- `watchfiles` on macOS survives its root folder being deleted and recreated; Linux's inotify drops the watch, not checked (vault-hardening.18).
- `rglob` order sorts by path segment, not by string, and the eval relies on it; `issubclass()` refuses runtime-checkable protocols with properties (vault-hardening.20).
- The delete sweep also removes a note that now yields no chunks; a test pins it, since a loop refactor can drop the rule silently (vault-hardening.21).
- The memory blow-up came from bge-m3 truncating only at 8,192 tokens and from a note's chunks being embedded as one batch padded to the longest (vault-hardening.05, .26).
- The parser's old title pattern could match across a line break, taking the line after a bare `#` as the title (vault-hardening.27).
- A running server holds 4.5 to 5.1 GB and a search takes 2.3 to 3.0 s; with the reranker off, 1.7 GB and about 20 ms (vault-hardening.29).

## Open issues and known limitations

- Notes whose chunks the fixes would change keep their old chunks until they are next edited, since the fingerprint did not change; a note whose only `#` lines sit in a fence keeps its old title until then. Deleting the index file rebuilds everything (vault-hardening.08, .27).
- The non-goals stay open: the MCP `reindex` tool blocks the server, an interrupted first index restarts from zero, concurrent writers hit "database is locked", `status.last_indexed` reports the current time, symlinked folders are not followed, `.canvas` and `.base` files are not indexed, and `search.top_k`, `store.backend` and the query filters are unused.
- `watch` surviving a vanished vault was checked on macOS only (vault-hardening.18).
- fastembed warns that `paraphrase-multilingual-mpnet-base-v2` now uses mean pooling; it shows in the test run and affects only the old control model.

## Takeaways

The hands-on test on a deliberately awkward vault found every finding of the cycle, and turning it into a permanent test made the gates cheap to trust. Checking that fixture byte for byte against the generator it came from caught a port that would have tested the wrong files.

Probing the real library before settling a design (`watchfiles` on a deleted root, on a moved folder) changed two decisions before any code depended on them.

Running the CI commands themselves, not only on `src` and `tests`, would have found the asset lint failure at gate 0 instead of at the final gate.
