# Plan: Chunking policy

Implements `docs/design/2026-09-24-chunking-policy.md`. Branch
`feat/chunking-policy`. Every task is test-first: write the test, see it fail,
make it pass, run the fast suite, commit.

## Correction to the design

The design says the `Chunker` port does not change. It has to. The index
fingerprint that forces a full reindex covers the embedding model and the
contextualizer, not the chunker, so changing `[chunking]` would leave every
unchanged note with its old chunks and no warning. The port gains a
`fingerprint` property, like `Contextualizer` already has.

The default policy's fingerprint is the empty string, and the index fingerprint
omits empty parts. Existing indexes therefore keep their stored fingerprint and
are not forced into a pointless reindex on upgrade; any non-default policy
changes it.

## Task 1: `ChunkingCfg`

`src/ariostea/config/schema.py`: add

```python
class ChunkingCfg(BaseModel):
    max_tokens: int = 512
    overlap: int = 0
    unit: Literal["words", "model_tokens"] = "words"
```

with a validator: `max_tokens > 0`, `0 <= overlap < max_tokens`. `Config` gains
`chunking: ChunkingCfg = ChunkingCfg()`.

Tests (`tests/config/`): defaults match today's chunker; a TOML `[chunking]`
section loads; `overlap >= max_tokens` is rejected; an unknown unit is rejected.

## Task 2: overlap and a cost function in `HeadingAwareChunker`

`HeadingAwareChunker(max_tokens=512, overlap=0, count=None)`. Each word in a
section gets a cost: 1 when `count` is `None`, else `count(word)`. A section
whose total cost fits returns its original text unchanged, as today. Otherwise
a window walks the words: fill up to `max_tokens` of cost (always at least one
word), emit the piece, then start the next piece far enough back to repeat at
most `overlap` of cost, always advancing by at least one word.

Tests:
- with default arguments, output is identical to today's on a long fixture
  (pin the current behaviour before touching the code);
- overlap repeats the tail of each piece at the head of the next;
- overlap never crosses a heading;
- a span straddling a boundary with overlap 0 is contained whole with overlap;
- a cost function is respected: pieces never exceed `max_tokens` of cost;
- a single word costing more than `max_tokens` still makes progress;
- `fingerprint` is `""` for the defaults and encodes size, overlap and unit
  otherwise.

## Task 3: chunker fingerprint in the index

`ports/pipeline.py`: `Chunker` gains `fingerprint`. `IndexVault._fingerprint`
joins the embedding, contextualizer and chunker fingerprints, skipping empty
parts.

Tests (`tests/indexing/`): the default chunker leaves the stored fingerprint
string exactly as before; a non-default chunker changes it and so re-chunks
unchanged notes.

## Task 4: model-token counting and container wiring

`FastEmbedEmbeddings.count_tokens(text)` returns the model tokenizer's count
without special tokens. Not on the `EmbeddingProvider` port: only the local
adapter has a tokenizer.

`config/container.py`: `build_chunker(cfg: ChunkingCfg, embeddings)` builds the
chunker, passing `count_tokens` when `unit == "model_tokens"`. It raises a clear
error when the unit is `model_tokens` and the embeddings adapter has no
tokenizer (the OpenAI-compatible provider). `build_container` uses it.

When counting model tokens, the budget reserves two tokens for the start and end
markers the model adds, so `max_tokens = 128` fits a 128-token window.

Tests: `build_chunker` with words ignores the tokenizer; with model tokens it
counts through it; with a tokenizer-less adapter it raises. One integration
test, marked `integration`, checks `count_tokens` on the real model.

## Task 5: the sweep runner

`eval/wiki_index.py`: `wiki_config` and `index_wiki_corpus` take an optional
`ChunkingCfg`.

`eval/run_chunk_sweep.py`:

```
uv run python eval/run_chunk_sweep.py 512w 128t 128t+32 --channels DENSE,SPARSE
```

A spec is a size, a unit (`w` words, `t` model tokens) and an optional `+overlap`.
Per spec it indexes the corpus, counts reachable spans with the same chunker,
scores the requested channels on the gold set and on the 29 cases the
discrimination gate dropped, prints the tables, and appends a run to
`eval/results/runs.jsonl` with the baseline as control. It re-renders the page
at the end. The parsing, run-id and record-building pieces live in
`src/ariostea/eval/chunk_sweep.py` so they are testable.

Tests: spec parsing (valid, invalid, overlap too large); run ids are stable and
distinct per spec and channel set; the discriminated cases load from
`gold_rejected.json`.

## Task 6: document it

README: the `[chunking]` section, what each key does, and that changing it
forces a reindex. `ariostea.example.toml`: the section, commented, with defaults.

## Task 7: run the sweep

1. Cheap pass, dense and sparse, nine configurations: `512w`, `96t`, `128t`,
   `160t`, `256t`, and each token size with overlap at a quarter. Logged under
   "Chunk sweep: dense and sparse".
2. Pick the three best by mean of dense and sparse overall span recall, among
   those with at least 160 reachable spans.
3. Full pass on those three, all channels, logged under "Chunk sweep: full".
   The control's hybrid numbers are the committed baseline, already logged.
4. Republish the log artifact after each pass.
5. Report against the decision rule. Do not change the default: the owner
   decides once the gain is known.

## Task 8: finish

Full suite including integration tests, lint, format, scope check (only
`config`, `adapters/chunk`, `adapters/embedding`, `ports/pipeline`,
`indexing` and `eval` touched), push, open the PR with the results table and
the log link.
