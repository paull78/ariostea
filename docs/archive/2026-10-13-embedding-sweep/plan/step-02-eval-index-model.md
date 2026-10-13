# Step 02 · Eval index takes a model

**Milestone:** 1 · **Needs:** none · **Covers:** AC-05

## Goal
`wiki_config` and `index_wiki_corpus` accept a `model` that defaults to `MULTILINGUAL_MODEL`, and `wiki_channels` builds its dense embedder from the container's configured model instead of the constant.

## Files touched
- `src/ariostea/eval/wiki_index.py`: `model` keyword on `wiki_config` and `index_wiki_corpus`; `wiki_channels` reads `container.config.embedding.local_model`.
- `tests/eval/test_wiki_index.py`: new tests.

## Tests to write first
- `test_wiki_config_defaults_to_the_pinned_model`: `wiki_config(CORPUS, "x.db") == wiki_config(CORPUS, "x.db", model=MULTILINGUAL_MODEL)`.
- `test_wiki_config_takes_a_model`: `wiki_config(CORPUS, "x.db", model="m").embedding.local_model == "m"`.
- `test_wiki_channels_queries_with_the_indexed_model`: monkeypatch `wiki_index.FastEmbedEmbeddings` and `wiki_index.SqliteStore` with fakes; pass a stub container whose `config.embedding.local_model == "m"`; assert the fake embedder was built with `model_name="m"`.

## Implementation notes
- `model` goes last as a keyword with a default, so every existing positional call still works.
- Callers (`run_chunk_sweep.py`, `run_blurb_eval.py`, `generate_gold.py`, `run_wiki_eval.py`) need no change.
- The module docstring's point stands: indexing and channels must use the same model. Reading it from the container enforces that.

## Verify
`uv run pytest tests/eval/test_wiki_index.py -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
