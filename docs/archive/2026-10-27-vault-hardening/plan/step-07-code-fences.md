# Step 07 · Code fences in chunker and parser

**Milestone:** 3 · **Needs:** 06 · **Covers:** AC-11

## Goal
Lines inside fenced code blocks never split sections, never enter `heading_path`, and never become the note title.

## Files touched
- `src/ariostea/adapters/markdown.py` (new): `heading_lines(body) -> Iterator[tuple[str, re.Match | None]]`, each line with its heading match, `None` inside a fence. CommonMark fence rule (vault-hardening.07): opening line of 3+ backticks or tildes (up to 3 leading spaces), closed by the same character at least as long; unclosed runs to the end.
- `src/ariostea/adapters/chunk/heading_aware.py`: `_split_sections` uses it.
- `src/ariostea/adapters/parse/obsidian.py`: the title is the first `#` heading outside fences.
- `tests/adapters/test_markdown.py` (new), `tests/adapters/chunk/test_heading_aware.py`, `tests/adapters/parse/test_obsidian.py`.

## Tests to write first
- `test_hash_lines_inside_backtick_fence_are_not_headings`, `..._tilde_fence_...`.
- `test_unclosed_fence_runs_to_the_end`.
- `test_longer_fence_is_not_closed_by_shorter`: a ```` fence containing a ``` line stays open.
- `test_fence_closed_by_other_char_stays_open`: ``` closed only by ```, not ~~~.
- `test_note_with_only_fenced_hashes_titles_from_file_name` (AC-11), `test_real_title_above_code_wins` (AC-11).
- `test_chunks_inside_fences_have_empty_heading_path` (AC-11).
- Re-run AC-19 (`test_wiki_chunks_unchanged`): fences must not change the wiki output.

## Implementation notes
- One helper used by both adapters, so title and sections cannot disagree.
- Tags and wikilinks extraction are untouched (non-goal).

## Verify
`uv run pytest tests/adapters -q -m "not integration"` → all pass; `uv run pytest tests/eval/test_wiki_chunks_unchanged.py -q` → pass.

## Done when
- manual: in the synthetic vault, `Code first.md` gets its file name as title through `search_sources`.
