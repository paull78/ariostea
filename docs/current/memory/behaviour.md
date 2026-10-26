# Behaviour

How each fix behaves.

### vault-hardening.02 · decision · 2026-10-26 · spec
Two guards protect the index: the vault path must exist and be a directory, and a scan that finds no notes while the index holds some stops the run. Either way the run errors with the path and the index is unchanged. To empty an index on purpose, delete the index file.
**Why:** the second guard covers the iCloud case, where the folder exists but evicted files show up as `.Name.md.icloud` and the scan finds nothing. Rejected: the path check alone (an existing empty folder still wipes the index); a ratio guard refusing scans that would delete more than some share (needs a threshold and an override flag, cleverer than the problem asks).

### vault-hardening.03 · decision · 2026-10-26 · spec
In `watch`, a guard failing at startup exits with the error, like `reindex`. A guard failing on a later pass prints a warning naming the path, leaves the index alone, and the watcher keeps going; the next change tries again.
**Why:** a brief unmount or an iCloud eviction should not kill a long-running watcher that nobody is looking at. Rejected: exiting on any guard failure (simplest code, but the watcher stays dead until restarted by hand). Open for the build: how watchfiles behaves when the watched folder itself disappears is not checked yet.

### vault-hardening.04 · decision · 2026-10-26 · spec
Notes are read as UTF-8 with a leading BOM dropped. A file that cannot be read (bad encoding, permission, vanished) is skipped with a warning naming it; the run continues, the summary counts skipped files and the MCP `reindex` result lists them. A skipped file counts as present, so an entry indexed earlier stays; a file that has truly gone is deleted by a later run. Dropping the BOM fixes the BOM-frontmatter leak, which moves from non-goal to in scope at no cost.
**Why:** Obsidian itself assumes UTF-8 and shows such files garbled, so skipping matches what the user sees in the app, with no encoding guess. Rejected: falling back to cp1252 (never fails, so UTF-16 or other encodings would be indexed as garbage text). Treating a skipped file as present was chosen without asking: deleting a note's chunks because of a momentary permission or sync error would lose work for nothing.

### vault-hardening.05 · decision · 2026-10-26 · spec
A word costlier than `max_tokens` is cut into pieces that fit, so no chunk ever exceeds the budget. Windows always move forward, which also removes the near-duplicate shrinking chunks in front of an oversized word.
**Why:** one rule with no threshold bounds memory for any input. The memory blow-up came from bge-m3 truncating only at 8,192 tokens (fastembed enables truncation at the tokenizer's `model_max_length`) and from embedding a note's chunks as one batch padded to the longest. Rejected: dropping words over N characters (needs a threshold, long URLs vanish from search, small `max_tokens` could still be exceeded); both rules (cleaner index, two rules and a threshold). Junk blobs still yield a few junk chunks; fix 4 removes the largest source, Excalidraw.

### vault-hardening.06 · decision · 2026-10-26 · spec
The scanner always skips paths with a segment starting with a dot and every `*.excalidraw.md`; `vault.ignore` only adds prefixes and defaults to empty. `watch` filters its events by the same rule (added without asking: same rule, and it removes the `.obsidian/workspace.json` churn the assessment found).
**Why:** Obsidian itself hides dot-folders, so one rule matches what the user sees, nested folders included, and a custom `ignore` list can no longer drop the defaults by replacing them. Rejected: a longer default `ignore` list (a user list still replaces it, nested dot-folders still indexed, Excalidraw needs glob support).

### vault-hardening.07 · decision · 2026-10-26 · spec
Fix 5 uses the CommonMark fence rule (three or more backticks or tildes, closed by the same character at least as long, an unclosed fence running to the end); the parser's title follows the same rule. Chosen without asking, as the standard.
**Why:** the documented rule the Obsidian editor follows; anything else would disagree with what the user sees highlighted as code.

### vault-hardening.08 · decision · 2026-10-26 · spec
The chunker fingerprint stays as it is. Notes whose chunks change under fixes 2, 3 and 5 are re-chunked when next edited; new indexes get every fix at once; deleting the index file forces a full rebuild.
**Why:** setup.04, no forced reindex without the owner's say, and the owner chose not to: the affected notes are few, and their old chunks are no worse than today. Rejected: bumping the fingerprint (every vault re-embeds once, about an hour for a large vault on bge-m3).

### vault-hardening.09 · decision · 2026-10-26 · spec
`get_note` checks the path against the index by exact match, then reads the file from the vault and returns it as written, frontmatter included; a missing or unreadable file returns `found: false` with a reason.
**Why:** the client asked for the note, and only the file has it whole: rebuilding from chunks repeats the overlap (24,535 words against 20,162 on the long test note) and loses line breaks and frontmatter. The exact index match keeps the traversal safety the database-only version had. Rejected: stripping frontmatter (loses properties a client may want); rebuilding from chunks without overlap (still loses formatting). Reading the vault from the search side needs a vault-reading port, so the use case does not touch the filesystem (architecture, to settle).

### vault-hardening.10 · decision · 2026-10-26 · spec
The `openai_compat` embedding fields are removed, and a config still setting any of them fails to load with a message naming the key. The CLI turns known errors (config, removed key, vault guard) into one line on stderr and exit status 1, so raw config tracebacks move into scope (added without asking: the same code path fixes 1 and 7 need).
**Why:** a user who configured Ollama embeddings learns the setting never worked instead of silently downloading bge-m3; remote embedding was already rejected in the embedding-sweep design. Rejected: ignoring the old keys silently, which repeats today's silent failure.

### vault-hardening.11 · decision · 2026-10-26 · spec
Progress: a stderr counter refreshed at most once a second (`412/3,410 notes · 37 embedded · 2 skipped`), plain lines every 100 notes when stderr is not a terminal, and one summary line per `watch` pass that changed something. MCP tools stay silent.
**Why:** an hour of silence on a first index looks like a hang, and `watch` today never says what it did. Rejected: one line per embedded note (10,000 lines on a first index); summary only (keeps the silence).
Superseded by vault-hardening.16 (output mode).

### vault-hardening.12 · decision · 2026-10-26 · spec
Fix 9's README content was filled without asking: the 3.2 GB download and its cache path, running `reindex` before connecting a client, config lookup in the current directory, the scanner's exclusions and skipped-file warnings, and the measured memory (4.3 GB server) and search time (about 2.5 s), with the 15 ms figure labelled as query embedding only.
**Why:** each item is a fact the hands-on test hit as a new user; the 15 ms figure read as search latency.

### vault-hardening.16 · pivot · 2026-10-26 · challenge round
Supersedes vault-hardening.11 for the output mode. Progress is one plain mode everywhere: a line every 100 notes or every 10 seconds, whichever comes first, and once at the end. `IndexVault` reports every note as a plain value; the CLI printer owns the clock and the throttling. The `watch` summaries of .11 stand.
**Why:** simplicity lens, the old design braided output format with terminal detection into two behaviours to build and test. Hexagonal lens, time is outside the app, so the clock lives in the driving adapter (the CLI), not in the use case. Rejected: keeping the in-place terminal line (nicer to watch, twice the paths).

### vault-hardening.18 · discovery · 2026-10-26 · self-review
Checked on this Mac: `watchfiles.watch` survives its root folder being deleted and recreated. Deleting the folder yields one batch (`added vault`, `deleted vault`), recreating it and adding `b.md` yields `added b.md`, and later changes keep arriving. So a vanished vault reaches `WatchVault` as an ordinary pass whose guard fails (vault-hardening.03), and the watcher carries on. Linux uses inotify, which drops a watch on a deleted folder; not checked, and the project runs on macOS.
**Why:** resolves the open point in vault-hardening.03 before Gate 0.

### vault-hardening.21 · discovery · 2026-10-26 · step 02
The delete sweep removes every stored note not marked seen in the pass, and a note that now yields no chunks is never marked: emptying a note removes it from the index. A first rewrite swept against the vault's note list instead, which would have kept an emptied note's old chunks; `test_a_note_emptied_of_chunks_leaves_the_index` pins the behaviour. Seen = unchanged, re-embedded, or skipped as unreadable.
**Why:** the old loop's `seen` set carried this rule implicitly, and any refactor of the loop can drop it without a failing test.

### vault-hardening.26 · discovery · 2026-10-26 · step 06
Measured after the split-to-fit change: a note with a 20,000-character base64 image indexes into 61 chunks, the largest 254 bge-m3 tokens (the 256 budget less two special tokens), with 211 MB of peak-memory growth against the 20 to 25 GB the hands-on test saw. The AC-19 digest over the wiki corpus at the defaults (2,858 chunks, `198542d4…`) was computed on the old chunker and still matches. Three tests were born green because they pin behaviour that already held: `test_short_sections_untouched`, `test_words_unit_never_splits_a_word`, and the tightened `test_a_cost_function_caps_each_piece`. The property test gives every character in a case its own CJK code point: with short ASCII units, a unit could end one piece and start the next by chance and read as overlap. The memory test was not run on the old chunker, which would take 20 to 25 GB of RAM.
**Why:** records the evidence for AC-07, AC-08 and AC-19, and why the property test's input looks odd.

### vault-hardening.27 · discovery · 2026-10-26 · step 07
The parser's old title pattern, `^#\s+(.+)$` with `re.MULTILINE`, could match across a line break: a line holding only `#` took the next line as the title. The shared `heading_lines` helper matches one line at a time, so that case now falls back to the file name, as it should. `test_real_title_above_code_wins` was born green: the old first-H1 search already found a title above the code. A note whose only `#` lines sit in a fence keeps its old wrong title until it is edited (vault-hardening.08, no fingerprint bump); the manual check needs a fresh store.
**Why:** a small behaviour change outside the fence fix, and a trap for the gate's hands-on check.
