# Step 10 · Normalise setting

**Milestone:** 2 · **Needs:** 04 · **Covers:** AC-10, AC-11

## Goal
`[embedding] normalize = true` makes the local embedder return unit-length document and query vectors, so the store's Euclidean ranking equals cosine ranking. Off by default, with production vectors and fingerprints unchanged (embedding-sweep.07, embedding-sweep.08).

## Files touched
- `src/ariostea/config/schema.py`: `EmbeddingCfg.normalize: bool = False`.
- `src/ariostea/adapters/embedding/fastembed_local.py`: `normalize` constructor keyword; divide each vector by its length when on; fingerprint gains `|unit`.
- `src/ariostea/config/container.py`: pass `config.embedding.normalize` to `FastEmbedEmbeddings`.
- `src/ariostea/eval/wiki_index.py`: `normalize` keyword on `wiki_config` and `index_wiki_corpus`, default `False`; `wiki_channels` builds its embedder with the container's `normalize`.
- `ariostea.example.toml`: commented `normalize` line with one sentence on what it does.
- Tests: `tests/adapters/embedding/test_fastembed_local.py`, `tests/config/test_schema.py`, `tests/config/test_container.py`, `tests/eval/test_wiki_index.py`.

## Tests to write first
- `test_normalize_off_returns_the_raw_vectors` (adapter, fake returning `[3, 4, 0]`): `embed_documents` and `embed_query` return `[3, 4, 0]`; fingerprint unchanged.
- `test_normalize_on_returns_unit_vectors`: with `normalize=True`, both return `[0.6, 0.8, 0.0]` within 1e-6.
- `test_normalize_on_marks_the_fingerprint`: fingerprint ends in `|unit` for the control and for e5 (after the prefixes).
- `test_embedding_normalize_defaults_off` (schema): `EmbeddingCfg().normalize is False`.
- `test_embedding_normalize_loads_from_toml`: `[embedding]\nnormalize = true` loads as `True`.
- `test_container_passes_normalize_to_the_embedder`: monkeypatch `container.FastEmbedEmbeddings` with a recorder; `build_container` with `normalize=True` builds it with `normalize=True`.
- `test_wiki_config_takes_normalize` and `test_wiki_channels_embeds_with_the_indexed_normalize` (wiki_index): mirror the step 02 model tests.

## Implementation notes
- Normalise in float64 with numpy, then `tolist()`. A zero vector stays as is rather than dividing by zero.
- `dimension` probes through `embed_query`, unaffected.
- bge-m3 and jina-v3 already return unit vectors; normalising them again is a no-op, which is the point of measuring candidates with the flag on.
- Memory: the container exposes ports only, and adapters subclass their port. Neither changes here.

## Verify
`uv run pytest tests/adapters/embedding tests/config tests/eval/test_wiki_index.py -m "not integration" -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
