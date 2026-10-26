# Spec: vault-hardening

Cycle: vault-hardening · Type: feature · Branch: feature/vault-hardening · Status: approved

## Goal
Make Ariostea safe and predictable on a vault that is not the owner's: a vault of any size, synced by iCloud or similar, holding imported notes, plugin files and pasted data. Indexing must never lose the index by accident, never stop on one bad file, and never spend minutes and tens of gigabytes on one note. Retrieval quality and the measured defaults stay as they are.

Source: the usability assessment of 2026-10-26, a hands-on test on a 54-note synthetic vault plus a code audit.

## Scope and non-goals
In scope, nine fixes:
1. A missing or empty vault path never wipes the index.
2. A file that cannot be read is skipped with a warning naming it; the run carries on.
3. A word longer than the chunk budget is split, so no chunk exceeds `chunking.max_tokens`.
4. Hidden folders, `.trash/` and Excalidraw drawings are left out by default.
5. Lines inside fenced code blocks are never read as headings.
6. `get_note` returns the note as written in the vault.
7. The `openai_compat` embedding provider is removed from the config.
8. Indexing prints progress.
9. The README tells a new user about the first-run model downloads.

Added on the way, because the fixes above need the same code: frontmatter after a BOM is recognised (fix 2), every CLI error is one line with no traceback (fixes 1 and 7), and a test enforces the dependency rule (challenge round).

Non-goals (deferred, recorded in the assessment): the MCP `reindex` tool blocking the server, resuming an interrupted first index, concurrent writers ("database is locked"), `status.last_indexed`, symlinked folders, `.canvas` and `.base` files, unused config keys (`search.top_k`, `store.backend`, query filters), Claude Desktop setup docs, packaging for PyPI. Retrieval changes of any kind (models, fusion, chunk sizes, the gold set).

## Behaviour
### Fix 1: the index survives a vault it cannot trust
Before indexing, the vault path must exist and be a directory. If the scan then finds no notes while the index holds some (iCloud not synced yet, files evicted, the wrong folder), the run stops too. In both cases the run reports an error naming the path and changes nothing in the index; notes that disappeared are deleted only after a scan that found at least one note. To empty an index on purpose, delete the index file.

```
vault path --exists and is a directory?--no--> error, index untouched
     | yes
   scan --0 notes found while the index has notes?--yes--> error, index untouched
     | no
   index changed notes, delete notes no longer there (as today)
```

The same guard applies to `ariostea reindex`, `ariostea watch` and the MCP `reindex` tool. The CLI exits with a non-zero status and a one-line message; the MCP tool returns the error to the client. `watch` exits with the error when a guard fails at startup; once running, a failed guard prints a warning naming the path, leaves the index alone, and the next change tries again.

### Existing indexes
The chunker fingerprint does not change. Notes whose chunks fixes 2, 3 and 5 would change are re-chunked when they are next edited; a new index gets every fix at once. Nobody is forced into a full reindex. Deleting the index file rebuilds it in full.

### Fix 2: one bad file never stops a run
Notes are read as UTF-8, and a leading byte-order mark is dropped (so frontmatter after a BOM is now recognised). A file that cannot be read (not valid UTF-8, permission denied, gone between listing and reading) is skipped with a warning naming its vault-relative path, and the run carries on. A skipped file counts as still present: if the index already holds it, its old entry stays, and a file that has really gone is deleted by the next run that cannot find it. The CLI summary line adds the number of skipped files; the MCP `reindex` result lists their paths. In `watch`, a skipped file never stops the watcher.

### Fix 3: no chunk exceeds the budget
A word that costs more than `chunking.max_tokens` on its own (a pasted base64 image, a minified blob) is cut into pieces that each fit, so every chunk stays within `max_tokens` whatever the input. Windows always move forward: each chunk of a section adds text the previous one did not have, and consecutive chunks repeat at most `overlap` of cost. This also removes the run of near-duplicate chunks, each one word shorter, that the old chunker emitted in front of an oversized word.

### Fix 4: what Obsidian hides is not indexed
The scanner always skips a path with any segment starting with a dot (`.obsidian/`, `.git/`, `.trash/`, the same folders nested deeper, hidden files) and every `*.excalidraw.md` drawing. `vault.ignore` adds prefixes on top of that rule and defaults to an empty list, so a user's own list can no longer drop the built-in exclusions. Notes already in an index that the rule now excludes are deleted by the next run, like any note that disappeared. `watch` reports only changes that pass the same rule, so Obsidian rewriting `.obsidian/workspace.json` no longer triggers a pass.

### Fix 5: code is not structure
A fenced code block opens with a line of three or more backticks or tildes and closes with a line of the same character at least as long; an unclosed fence runs to the end of the note (the CommonMark rule). Lines inside a fence are never headings: they neither split sections nor enter `heading_path`, and the note title comes from the first `#` heading outside any fence, else the file name.

### Fix 6: `get_note` returns the file
`get_note` first looks the path up in the index by exact match, so only indexed notes can be returned and a path such as `../../etc/passwd` is never read. It then reads that file from the vault and returns its text as written, frontmatter included, with the title from the index. A path the index does not hold returns `found: false`; an indexed note whose file is gone or unreadable returns `found: false` with a reason.

### Fix 7: one embedding provider
`[embedding]` loses `provider`, `base_url`, `api_key` and `model`; Ariostea embeds locally with fastembed. A config that still sets any of them fails to load with a message naming the key and pointing to `embedding.local_model`. The example config and README drop them.

### Errors reach the user as one line
Every CLI command prints a known error (missing or invalid config, a removed key, a vault guard from fix 1) as one line on stderr and exits with status 1, with no traceback. This is the same path fixes 1 and 7 need, so raw tracebacks on config errors move from non-goal to in scope.

### Fix 8: indexing shows progress
`reindex` and the first pass of `watch` print plain progress lines on stderr, such as `412/3,410 notes · 37 embedded · 2 skipped`, every 100 notes or every 10 seconds, whichever comes first, and once at the end; the same output in a terminal, a log file or a client's stderr. `IndexVault` reports every note to the callback as a plain value; the CLI's printer owns the clock and decides when to print. The total comes from listing the vault's files before reading them. After that, `watch` prints one line for each pass that changed something, such as `+2 updated, -1 deleted`, and nothing for passes that changed nothing. The MCP tools print nothing, so stdout stays the protocol stream.

### Fix 9: the README prepares a new user
The quick start says that the first command downloads about 3.2 GB of models (bge-m3 2.1 GB, the jina reranker 1.1 GB) into `~/.cache/fastembed`; that `ariostea reindex` should run once from a terminal before an MCP client starts the server, so the client does not time out during the download; that the config is read from `ariostea.toml` in the current directory unless `--config` says otherwise; what the scanner skips (fix 4) and how unreadable files are reported (fix 2); and the measured cost on the test Mac: about 4.3 GB of memory for a running server and about 2.5 s per search on CPU, most of it the reranker. The 15 ms figure is labelled as query embedding time.

## Architecture
The vault becomes a driven actor behind a port, like the store and the embedder.

```
CLI / MCP --> IndexVault ---+--> Vault port: notes() -> paths, read(path) -> text,
              GetNote ------+               changes() -> batches of note changes
              WatchVault ---+    FilesystemVault adapter owns every file rule:
                                 dot segments, *.excalidraw.md, vault.ignore,
                                 UTF-8 with BOM dropped, path exists and is a folder,
                                 unreadable file -> NoteUnreadable; wraps watchfiles
IndexVault owns policy: the 0-notes guard, skipping and recording unreadable notes,
the delete sweep, progress reported as plain values to a callback the CLI prints.
GetNote: exact lookup in the index (DocumentReader), then Vault.read.
```

- `ports/vault.py` defines the `Vault` port and its errors (`VaultUnavailable`, `NoteUnreadable`); `adapters/vault/filesystem.py` implements it and subclasses the port (setup.02). `indexing/scanner.py` goes away; its rules move into the adapter.
- `IndexVault.index()` takes no path: the vault is injected. It takes an optional progress callback receiving a plain value (done, total, embedded, skipped).
- `GetNote` is a new use case; the `Container` exposes it in place of the bare `reader` for `get_note` (setup.01: config, use cases, port-typed admin).
- Change notification is the vault's job too: `Vault.changes()` yields batches that already follow the note rule, so `WatchVault` no longer imports `watchfiles` and no filter passes between objects. Indexing, `get_note` and `watch` cannot disagree on what a note is.
- A dependency-rule test parses the imports of `domain/`, `ports/`, `indexing/` and `search/` with the standard `ast` module and fails on any import of `ariostea.adapters`, `ariostea.config`, `ariostea.mcp`, `watchfiles`, `fastembed`, `sqlite3`, `sqlite_vec`, `mcp` or `httpx`. Today only `watch_vault.py`'s `watchfiles` import breaks it, and this cycle removes that.
- The CLI maps the known errors (config errors, removed keys, `VaultUnavailable`) to one line and exit status 1.

## Acceptance criteria
- AC-01 · Indexing a vault path that does not exist, or is a file, fails with an error naming the path, and the index keeps every note and chunk it had (CLI exit status non-zero).
- AC-02 · Indexing an existing folder with no notes, while the index holds notes, fails with an error naming the path and leaves the index unchanged; on an empty index the same folder indexes to 0 notes without error.
- AC-03 · With `watch` running, a guard failure on a later pass prints a warning naming the path, leaves the index unchanged, and does not stop the watcher; the next pass over a restored vault indexes normally.
- AC-04 · A vault holding a Latin-1 note, an unreadable (permission-denied) note and valid notes indexes every valid note; each bad file is named in a warning, the summary counts 2 skipped files, and the exit status is zero. A note indexed earlier that becomes unreadable keeps its chunks.
- AC-05 · A note starting with a byte-order mark followed by frontmatter is indexed without the frontmatter in its chunks, and its title comes from the body or the file name, never from the frontmatter text.
- AC-06 · With `watch` running, adding a non-UTF-8 note prints a warning naming it and the watcher keeps indexing later changes.
- AC-07 · A note holding a 20,000-character string with no spaces indexes into chunks that each cost at most `max_tokens` under the model's tokenizer, and the whole note embeds within 2 GB of peak memory above the loaded model's.
- AC-08 · For any section text and any `max_tokens` > `overlap` (property test over random words of random cost, including words costlier than the budget), the chunker emits chunks within the budget, consecutive chunks share at most `overlap` of cost, every chunk adds at least one word the previous lacked, and the chunks together cover every word.
- AC-09 · With `vault.ignore = ["Templates/"]`, a vault holding notes under `.git/`, `.trash/`, `sub/.obsidian/`, `Templates/`, a hidden `.draft.md` and `Board.excalidraw.md` indexes none of them and indexes every other note; an index that held them before loses them on the next run.
- AC-10 · With `watch` running, a change under `.obsidian/` alone triggers no indexing pass.
- AC-11 · A note whose only `#` lines sit inside ```` ``` ```` and `~~~` fences (one of them unclosed, one fence of four backticks containing a three-backtick line) gets its title from the file name and every chunk has an empty `heading_path`; adding a real `# Title` above the code makes that the title.
- AC-12 · `get_note` on an indexed note long enough to need overlapping windows returns exactly the file's text read as UTF-8 (frontmatter, line breaks and code included; a BOM dropped and line endings read as `\n`, as Python's text mode does).
- AC-13 · `get_note` with `../../etc/passwd`, an absolute path, or a vault file the index does not hold (a hidden or ignored note) returns `found: false` and reads no file outside the index; an indexed note deleted from disk returns `found: false` with a reason.
- AC-14 · A config with `[embedding] provider = "openai_compat"` (or `base_url`, `api_key`, `model`) makes every CLI command exit with status 1 and a one-line message naming the key, with no traceback and no model loaded.
- AC-15 · A missing config file, a config without `vault.path` and a vault guard failure each print one line on stderr naming the problem and exit with status 1, with no traceback.
- AC-16 · `reindex` on a vault of 250 notes prints progress lines carrying the done/total count after notes 100 and 200 and at the end, then the summary; with a fake clock advancing 10 seconds per note, the printer prints a line for every note; stdout of `serve` over stdio carries nothing but protocol messages during an MCP `reindex`.
- AC-17 · With `watch` running, editing one note and deleting another prints one summary line reporting 1 updated and 1 deleted; a pass that changes nothing prints nothing.
- AC-18 · The README quick start states the 3.2 GB first-run download and where it goes, says to run `reindex` from a terminal before connecting a client, names the config lookup in the current directory, lists what the scanner skips, and gives the measured memory and per-search time.
- AC-19 · At the default chunking settings, the chunker's output on every note of the wiki eval corpus is identical before and after the cycle (no eval rerun needed).
- AC-21 · The dependency-rule test passes on the cycle's code and fails when an import of `watchfiles` or `ariostea.adapters` is added to any module under `indexing/` (checked once by adding one).
- AC-20 · An end-to-end test builds the synthetic awkward vault (54 notes: bad encodings, BOM, CRLF, dot folders, Excalidraw, inline base64, fences, non-ASCII names, a 20,000-word note) and runs `reindex`, `watch` and every MCP tool on it: indexing completes with the bad files reported, no excluded file is searchable, and every known-answer query from the hands-on test ranks its note first.

## Testing approach
- Use cases (`IndexVault`, `GetNote`, `WatchVault`) run against an in-memory `Vault` fake, which also yields scripted change batches: the guards, skipping, the delete sweep, progress values and watch summaries. A contract test runs the same cases on the fake and on `FilesystemVault`.
- `FilesystemVault` runs on real files in temporary folders: Latin-1, permission-denied, vanishing and BOM files, dot folders at any depth, Excalidraw, `vault.ignore`, a missing path and a path that is a file.
- The dependency-rule test (AC-21) in the fast suite.
- The chunker and parser: fence cases (AC-11) and the property test of AC-08 as a seeded random loop over random words and costs, no new dependency.
- CLI commands through Typer's test runner: exit status, one-line errors, progress lines (AC-14 to AC-17).
- Integration (marked `integration`, real bge-m3): the memory bound of AC-07, and AC-19, which shows the default chunker output on the wiki corpus is unchanged, so no eval rerun is needed.
- End to end (integration): the hands-on test's synthetic-vault generator (kept in `docs/current/assets/make_vault.py`, with its MCP client and queries) becomes a test fixture; one test builds the 54-note awkward vault and drives `reindex`, `watch` and every MCP tool on it (AC-20). Milestone gates rerun it by hand as the hands-on check.

Test command: `uv run pytest` (full suite, about 23 minutes); `uv run pytest -m "not integration"` for the fast suite.

## Open questions
- none
