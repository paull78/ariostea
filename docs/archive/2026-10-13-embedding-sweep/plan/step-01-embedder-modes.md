# Step 01 · Embedder query/passage modes and prefixes

**Milestone:** 1 · **Needs:** none · **Covers:** AC-01, AC-02, AC-03

## Goal
`FastEmbedEmbeddings` embeds documents through fastembed's `passage_embed` and queries through `query_embed`, prepends the prefixes a model requires (only `intfloat/multilingual-e5-large`: `query: ` and `passage: `), and keeps the control model's fingerprint unchanged so no vault reindexes.

## Files touched
- `src/ariostea/adapters/embedding/fastembed_local.py`: prefix table, `passage_embed`/`query_embed` calls, fingerprint.
- `tests/adapters/embedding/test_fastembed_local.py`: unit tests with a fake `TextEmbedding`, one new integration test.

## Tests to write first
Unit tests monkeypatch `ariostea.adapters.embedding.fastembed_local.TextEmbedding` with a fake that records the texts passed to `embed`, `query_embed` and `passage_embed` and returns fixed numpy vectors.
- `test_documents_go_through_passage_embed`: `embed_documents(["a", "b"])` calls the fake's `passage_embed` with `["a", "b"]` and never `embed`.
- `test_queries_go_through_query_embed`: `embed_query("q")` calls `query_embed` with `["q"]`.
- `test_e5_prefixes_passages_and_queries`: with model `intfloat/multilingual-e5-large`, the fake receives `["passage: a"]` and `["query: q"]`.
- `test_models_without_prefixes_get_the_text_unchanged`: the control model sends `"a"` and `"q"` unchanged.
- `test_control_fingerprint_is_unchanged`: fingerprint equals `fastembed:sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- `test_prefixed_model_fingerprint_names_the_prefixes`: e5's fingerprint is not `fastembed:intfloat/multilingual-e5-large` and contains both prefixes.
- `test_count_tokens_ignores_the_prefix`: with a fake tokenizer that counts characters, `count_tokens("abc")` is 3 for e5 and for the control.
- Integration, `test_control_vectors_match_plain_embed`: for the control model, `embed_documents([t])[0]` and `embed_query(t)` equal `TextEmbedding(model).embed([t])` within 1e-6.

## Implementation notes
- A module-level `_PREFIXES: dict[str, tuple[str, str]] = {"intfloat/multilingual-e5-large": ("query: ", "passage: ")}`, looked up once in `__init__`.
- Fingerprint: `fastembed:{name}` when there is no prefix; otherwise `fastembed:{name}|query={q!r}|passage={p!r}`. The control's string must not change (spec Behaviour, AC-01).
- `count_tokens` stays on the raw text. `dimension` probes via `embed_query`, which is fine.
- jina-v3 needs no prefix entry: fastembed's `query_embed`/`passage_embed` select its task ids.
- Keep the class subclassing `EmbeddingProvider` (memory: adapters subclass ports). The port does not change.

## Verify
`uv run pytest tests/adapters/embedding -m "not integration" -q` → all pass; then `uv run pytest tests/adapters/embedding -q` → all pass, including the integration test (control model is cached).

## Done when
- All listed tests pass, and the fast suite passes.
