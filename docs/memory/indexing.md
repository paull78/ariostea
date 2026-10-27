# Indexing

Chunking, parsing, progress output, `get_note` and CLI errors.

### vault-hardening.05 · decision · 2026-10-26 · spec
A word costlier than `max_tokens` is cut into pieces that fit, so no chunk ever exceeds the budget. Windows always move forward, which also removes the near-duplicate shrinking chunks in front of an oversized word.
**Why:** one rule with no threshold bounds memory for any input. The memory blow-up came from bge-m3 truncating only at 8,192 tokens (fastembed enables truncation at the tokenizer's `model_max_length`) and from embedding a note's chunks as one batch padded to the longest. Rejected: dropping words over N characters (needs a threshold, long URLs vanish from search, small `max_tokens` could still be exceeded); both rules (cleaner index, two rules and a threshold). Junk blobs still yield a few junk chunks; fix 4 removes the largest source, Excalidraw.

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

### vault-hardening.16 · pivot · 2026-10-26 · challenge round
Supersedes vault-hardening.11 for the output mode. Progress is one plain mode everywhere: a line every 100 notes or every 10 seconds, whichever comes first, and once at the end. `IndexVault` reports every note as a plain value; the CLI printer owns the clock and the throttling. The `watch` summaries of .11 stand.
**Why:** simplicity lens, the old design braided output format with terminal detection into two behaviours to build and test. Hexagonal lens, time is outside the app, so the clock lives in the driving adapter (the CLI), not in the use case. Rejected: keeping the in-place terminal line (nicer to watch, twice the paths).

### vault-hardening.21 · discovery · 2026-10-26 · step 02
The delete sweep removes every stored note not marked seen in the pass, and a note that now yields no chunks is never marked: emptying a note removes it from the index. A first rewrite swept against the vault's note list instead, which would have kept an emptied note's old chunks; `test_a_note_emptied_of_chunks_leaves_the_index` pins the behaviour. Seen = unchanged, re-embedded, or skipped as unreadable.
**Why:** the old loop's `seen` set carried this rule implicitly, and any refactor of the loop can drop it without a failing test.

### vault-hardening.26 · discovery · 2026-10-26 · step 06
Measured after the split-to-fit change: a note with a 20,000-character base64 image indexes into 61 chunks, the largest 254 bge-m3 tokens (the 256 budget less two special tokens), with 211 MB of peak-memory growth against the 20 to 25 GB the hands-on test saw. The AC-19 digest over the wiki corpus at the defaults (2,858 chunks, `198542d4…`) was computed on the old chunker and still matches. Three tests were born green because they pin behaviour that already held: `test_short_sections_untouched`, `test_words_unit_never_splits_a_word`, and the tightened `test_a_cost_function_caps_each_piece`. The property test gives every character in a case its own CJK code point: with short ASCII units, a unit could end one piece and start the next by chance and read as overlap. The memory test was not run on the old chunker, which would take 20 to 25 GB of RAM.
**Why:** records the evidence for AC-07, AC-08 and AC-19, and why the property test's input looks odd.

### vault-hardening.27 · discovery · 2026-10-26 · step 07
The parser's old title pattern, `^#\s+(.+)$` with `re.MULTILINE`, could match across a line break: a line holding only `#` took the next line as the title. The shared `heading_lines` helper matches one line at a time, so that case now falls back to the file name, as it should. `test_real_title_above_code_wins` was born green: the old first-H1 search already found a title above the code. A note whose only `#` lines sit in a fence keeps its old wrong title until it is edited (vault-hardening.08, no fingerprint bump); the manual check needs a fresh store.
**Why:** a small behaviour change outside the fence fix, and a trap for the gate's hands-on check.

### vault-hardening.28 · decision · 2026-10-26 · step 08
`GetNote(reader, vault)` is the use case behind `get_note`, exposed on the Container as `notes` in place of `reader`. With it in place nothing called `read_note` any more, so `DocumentReader.read_note`, its SQLite implementation (chunks joined back together) and the `NoteDocument` model were removed; `DocumentReader` keeps only `note_titles`, which both `GetNote` and `SearchSources` use. `NoteMissing.reason` is `None` for a path the index does not hold and the read error otherwise, and the MCP payload adds `reason` only when it has one.
**Why:** setup.01 (the Container exposes use cases and ports, not adapters) and dead code removal; a missing note and an unreadable one need different answers for a client.
