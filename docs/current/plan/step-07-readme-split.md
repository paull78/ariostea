# Step 07 · README split

**Milestone:** 3 · **Needs:** 06 · **Covers:** AC-05, AC-06

## Goal
`README.md` reads as a product page and `LEARNING.md` holds the learning journey, both correct against the code and the logged runs (search-speed.25).

## Files touched
- `README.md`: product page only, link to `LEARNING.md` and `ARCHITECTURE.md`.
- `LEARNING.md`: new; How it is built (diagram, what Clean Architecture buys), the instrument, baseline, Trials 1 to 5, the defaults today.
- `docs/retrieval-tuning.md`: a search-speed section and its reproduction command.
- `ariostea.example.toml`, `ARCHITECTURE.md`: links that named a README section.

## Tests to write first
Exception: documentation; the check is the review list below and the link check.

## Implementation notes
Corrections from the review: p95 707 ms is "about 0.7 s"; jina never gets under 1 s; name the 50-note vault (2.5 s) and the eval set (4.0 s); Trial 5 used its own pre-set bar, not the +0.03 rule; Trial 5's scratch figures are not from runs.jsonl; "bge-m3's cost"; bge-m3 is 2.27 GB and the first download about 2.8 GB; reranker scores are unbounded raw scores. Neutral register, VOICE.md through human-write-time.

## Verify
`grep -o '([A-Za-z./-]*\.md[#a-z-]*)' README.md LEARNING.md | sort -u` → every target exists; `uv run ruff check . && uv run ruff format --check .` → passes.

## Done when
- README holds no trial; LEARNING.md holds all five; the eight corrections are in.
- manual: the owner reads README.md and LEARNING.md.
