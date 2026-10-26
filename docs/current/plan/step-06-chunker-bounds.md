# Step 06 · Chunker bounds oversized words

**Milestone:** 3 · **Needs:** none · **Covers:** AC-07, AC-08, AC-19

## Goal
No chunk costs more than `max_tokens`, whatever the input, and every window adds new text. The default output on the wiki corpus is unchanged.

## Files touched
- `src/ariostea/adapters/chunk/heading_aware.py`: `_fit` splits a word costing more than `max_tokens` into character pieces that each fit, and steps back for overlap only as far as still lets the next window take the next new word.
- `tests/adapters/chunk/test_heading_aware.py`: new property and case tests.
- `tests/eval/test_wiki_chunks_unchanged.py` (new, integration): AC-19.
- `tests/test_chunk_memory.py` (new, integration): AC-07.

## Tests to write first
- First, on the unchanged code, compute the AC-19 golden value: chunk every note of the wiki eval corpus with the default `ChunkingCfg` and bge-m3's tokenizer (as `build_chunker` does) and record a sha256 over `(path, ordinal, heading_path, text)`; hard-code it in `test_wiki_chunks_unchanged`. Commit nothing else before it passes on the old code.
- `test_fit_property` (AC-08): seeded `random.Random(seed)` over 500 cases: random words with random costs 1 to 3×`max_tokens`, `max_tokens` 2 to 40, `overlap` 0 to `max_tokens - 1`; with a cost function that also prices substrings (cost = len(word) // k + 1). Assert: every piece's cost ≤ `max_tokens`; consecutive pieces share at most `overlap` of cost; each piece holds a word or word-part the previous lacked; concatenating the non-overlapping parts rebuilds the input words in order.
- `test_oversized_word_is_split_to_fit`: one 20,000-character token-like string with `max_tokens=256` → every piece ≤ 256.
- `test_no_shrinking_duplicates_before_an_oversized_word`: the web-clip case from the hands-on test (`# Web clip Article ...` then a 4,500-token word) → no two chunks where one is a suffix of the other.
- `test_short_sections_untouched`: the existing "untouched, newlines and all" behaviour stays.
- `test_wiki_chunks_unchanged` (AC-19, integration).
- `test_long_string_note_memory_bound` (AC-07, integration): in a subprocess, load bge-m3, record `ru_maxrss`, index a note holding a 20,000-character string with no spaces, record again; the difference is under 2 GB, and every chunk's `count_tokens` ≤ `max_tokens`.

## Implementation notes
- Splitting a word: greedy by characters, longest prefix whose `cost(prefix)` ≤ budget (binary search; at least one character per piece). With `count=None` (words unit), a word costs 1 and is never split.
- Forward progress (vault-hardening.05): step back while `repeated + costs[back-1] + costs[end] <= max_tokens`, so the next window always reaches word `end`.
- The chunker fingerprint does not change (vault-hardening.08).
- The model's tokenizer may count a substring differently from the same characters inside a longer word; the budget is checked with `count` on each piece, so the bound holds for the piece as embedded.

## Verify
`uv run pytest tests/adapters/chunk -q` → all pass; `uv run pytest tests/eval/test_wiki_chunks_unchanged.py tests/test_chunk_memory.py -q` → pass.

## Done when
- AC-19's golden value was computed on the code before this step's change.
