# Per-chunk context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Contextualize each chunk separately (Anthropic's Contextual Retrieval) with the local LLM, and measure it on the wiki gold set.

**Architecture:** A new `LLMChunkContextualizer` behind the existing `Contextualizer` port, selected by `contextual.granularity = "chunk"`. `build_container` gains an optional `wrap_chat` hook so the eval can put `CachingChat` in front of the LLM. The existing blurb runner gains `--granularity chunk` and `--preview`.

**Tech Stack:** Python 3.12, pydantic config, httpx chat adapter, pytest, LM Studio.

Implements `docs/design/2026-09-29-per-chunk-context.md`. Branch `feat/blurb-aware-rerank` (continues it).

**Fast suite:** `uv run pytest -m "not integration" -q` before every commit.

---

### Task 1: `ContextualCfg.granularity` and the `wrap_chat` hook

**Files:** `src/ariostea/config/schema.py`, `src/ariostea/config/container.py`, `tests/config/test_schema.py`, `tests/config/test_container.py`

- [ ] Tests first:
  - `ContextualCfg().granularity == "note"`; `granularity = "chunk"` loads from TOML; an unknown value is rejected.
  - `_build_contextualizer(ContextualCfg(enabled=True, granularity="chunk"))` returns an `LLMChunkContextualizer`; with `"note"` an `LLMContextualizer` (as today).
  - `_build_contextualizer(cfg, wrap_chat=f)` passes the built `OpenAICompatChat` through `f` and uses what `f` returns (assert with a recording fake).
  - `build_container(config, wrap_chat=f)` forwards `f` to the contextualizer (fake `f` records that it was called when `contextual.enabled`).
- [ ] Implement: `granularity: Literal["note", "chunk"] = "note"` on `ContextualCfg`. `_build_contextualizer(cfg, wrap_chat=None)`: build the chat, apply `wrap_chat` if given, then choose the adapter by granularity. `build_container(config, *, wrap_chat=None)` passes it on. Add `granularity = "note"` with a comment to `ariostea.example.toml` under `[contextual]` if that section exists.
- [ ] Task 2 creates the adapter; for this task create `src/ariostea/adapters/contextualize/llm_chunk.py` with the class skeleton from Task 2 so imports resolve, or do Tasks 1 and 2 together in one commit.
- [ ] Commit: `feat(config): contextual.granularity and a chat hook for the contextualizer`

### Task 2: `LLMChunkContextualizer`

**Files:** `src/ariostea/adapters/contextualize/llm_chunk.py`, `tests/adapters/contextualize/test_llm_chunk.py`

- [ ] Tests first, with a fake `ChatProvider` that records `(system, user)` and returns scripted answers:
  - Every call for one note has a byte-identical `system`, which contains the full document inside `<document>…</document>`; each `user` contains that chunk's text inside `<chunk>…</chunk>` and the same-language instruction.
  - Calls happen in chunk order, one per chunk.
  - `context_blurb` is the stripped answer; `embedding_text == contextual_text(answer, chunk.text)`.
  - A chunk whose call raises, or returns empty/whitespace, gets `context_blurb=None`, `embedding_text=chunk.text`; the other chunks keep theirs; a warning is logged naming the note and ordinal.
  - `fingerprint == "llm-chunk:<model>"`.
  - It subclasses the `Contextualizer` port (project rule: adapters subclass their port).
- [ ] Implement:

```python
_SYSTEM = (
    "You situate chunks of a document for search retrieval.\n\n"
    "<document>\n{doc}\n</document>"
)
_USER = (
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. Write it in "
    "the same language as the chunk. Answer only with the succinct context and nothing else."
)


class LLMChunkContextualizer(Contextualizer):
    """Anthropic-style Contextual Retrieval: one LLM call per chunk, each seeing
    the whole note. The note goes in the system message, identical for every
    chunk, so a server with prompt caching processes it once per note. A failed
    call degrades only its own chunk to plain text."""

    def __init__(self, chat: ChatProvider, model_name: str) -> None: ...
    def contextualize(self, note, full_doc, chunks) -> list[ContextualizedChunk]: ...
    @property
    def fingerprint(self) -> str:
        return f"llm-chunk:{self._model_name}"
```

  Follow the error-handling and logging style of `LLMContextualizer` (`adapters/contextualize/llm.py`).
- [ ] Commit: `feat(contextualize): per-chunk contexts, Anthropic's method`

### Task 3: runner support

**Files:** `eval/run_blurb_eval.py`, `src/ariostea/eval/blurb_eval.py`, `src/ariostea/eval/wiki_index.py`, `tests/eval/test_blurb_eval.py`, `tests/eval/test_wiki_index.py`

- [ ] `wiki_config`/`index_wiki_corpus` pass `wrap_chat` through to `build_container` (test the pass-through with a fake).
- [ ] Pure helpers with tests in `blurb_eval.py`:
  - `contexts_by_chunk(rows) -> dict[str, dict[int, str]]` from `(note_path, ordinal, context)` rows, skipping empty contexts. Add a reader for those rows next to `read_blurb_rows` in `src/ariostea/eval/contextual.py` (the existing reader returns no ordinal).
  - `index_marker(granularity, model) -> str`: `"llm:<model>"` or `"llm-chunk:<model>"`, used by the `--reuse-index` guard.
  - `experiment_name(granularity)`: `"Contextual blurbs"` / `"Per-chunk context"`; run ids must differ by granularity (e.g. `...-chunkctx-raw-rerank`), so a per-chunk run on the same day as a note-level run cannot collide. Test that.
- [ ] Runner:
  - `--granularity {note,chunk}` (default `note`, behaviour unchanged).
  - Chunk mode: index path `chunk-context-160t+40.db`; `ContextualCfg(granularity="chunk", timeout=900 default, max_tokens=200)`, still overridable by `ARIOSTEA_CTX_TIMEOUT`; `wrap_chat` wraps in `CachingChat(chat, INDEXES / "chunk-context-cache.jsonl", label=ctx.model)` (keep a reference to report hits/misses at the end of indexing).
  - Dump `logs/<today>-chunk-contexts.json` via `contexts_by_chunk` in chunk mode; note mode keeps `blurbs_by_note`.
  - `--preview NOTE[,NOTE]`: preflight the LLM, build the chunk contextualizer (through the cache), chunk each listed note with the production chunker, contextualize, print `ordinal: context` per chunk, report cache hits/misses, exit 0. No index, no log.
  - Record `granularity` in each run's config.
- [ ] Smoke checks: `--help`; `--granularity chunk --reuse-index --no-log` fails on the missing index without creating files; `--preview` with LM Studio down aborts with the preflight message.
- [ ] Commit: `feat(eval): run the blurb experiment with per-chunk context`

### Task 4: full suite

- [ ] `uv run pytest -q` (about 14 minutes). All pass.

### Task 5: language check (manual)

- [ ] `lms unload --all && lms load qwen2.5-14b-instruct-mlx --context-length 32768 --parallel 1 -y`
- [ ] `uv run python eval/run_blurb_eval.py --granularity chunk --preview coffee/caffe-espresso-it.md,string-instruments/violin-es.md`
- [ ] Contexts must be in the chunk's language and situate the chunk (name the article's subject). If the language is wrong, fix the prompt before the full run; the cache key includes the prompt, so changed prompts regenerate.

### Task 6: the measurement (manual, overnight)

- [ ] `nohup uv run python eval/run_blurb_eval.py --granularity chunk > eval/results/logs/chunk-context-run.out 2>&1 & disown`; monitor with `kill -0` on the saved PID. Resume after a crash with the same command (cache) or `--reuse-index --arms ...`.
- [ ] Unload the model, commit `runs.jsonl`, the rendered page, the logs and the contexts dump; republish the log artifact.

### Task 7: write-up

- [ ] Apply the decision rule against the no-blurb controls; compare with the note-level arms.
- [ ] Add a "Per-chunk context" section to `docs/retrieval-tuning.md` with the results table and a finding/evidence/reason table like the note-level one; update memories `contextual-lift-measured` and `per-chunk-context-pilot`.
- [ ] Ask the owner before changing any default.
