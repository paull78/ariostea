# Ariostea

[![CI](https://github.com/paull78/ariostea/actions/workflows/ci.yml/badge.svg)](https://github.com/paull78/ariostea/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Ariostea is a local-first RAG server for an Obsidian vault. Point it at the vault and it
indexes the notes incrementally, then exposes retrieval to any MCP client, Claude included,
through five tools. Everything runs on the machine with local, keyless models. An optional LLM
step can add context to each chunk before indexing, following Anthropic's
[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) method. Measured
on the eval corpus it did not pay, so it is off by default.

Two retrieval modes:

- `search_knowledge` returns the most relevant passages. Dense and BM25 search run in
  parallel, RRF fuses them, and a cross-encoder reranks the pool.
- `search_sources` returns which notes a concept appears in, and `get_note` fetches one in
  full.

> This is a personal learning project: a RAG pipeline (hybrid retrieval, contextual
> retrieval, cross-encoder reranking) built from scratch with few dependencies, on a strict
> Clean Architecture skeleton.

## How it is built

Ariostea follows Clean Architecture. Source-code dependencies point only inward, toward the
more abstract and more stable code. The domain has no framework imports, use cases depend on
ports (Protocols) rather than concrete adapters, and every concrete detail, SQLite, the
embedding model, the MCP framework, is wired in one place, the composition root
`config/container.py`.

```mermaid
flowchart TB
    subgraph FD["🖥️ Frameworks & Drivers"]
        FDex["Typer CLI · FastMCP server · composition root"]
        subgraph IA["🔌 Interface Adapters — adapters/*"]
            IAex["FilesystemVault · ObsidianMarkdownParser · SqliteStore · FastEmbedEmbeddings"]
            subgraph UC["⚙️ Use Cases — indexing/ · search/"]
                UCex["IndexVault · WatchVault · SearchKnowledge · SearchSources · GetNote"]
                subgraph PO["🧩 Ports — ports/* (abstractions)"]
                    POex["Vault · EmbeddingProvider · ChunkRetriever · Fuser · Reranker"]
                    DM["💎 Domain — domain/models.py<br/>Note · Chunk · Query · RetrievedChunk"]
                end
            end
        end
    end

    classDef fd fill:#fde8e8,stroke:#d33,color:#000;
    classDef ia fill:#fef3c7,stroke:#d97706,color:#000;
    classDef uc fill:#dcfce7,stroke:#16a34a,color:#000;
    classDef po fill:#dbeafe,stroke:#2563eb,color:#000;
    classDef dm fill:#ede9fe,stroke:#7c3aed,color:#000;
    classDef ex fill:transparent,stroke-width:0px,color:#334155,font-size:13px;
    class FD fd; class IA ia; class UC uc; class PO po; class DM dm;
    class FDex,IAex,UCex,POex ex;
```

Each layer may depend only on the layers inside it. The innermost `domain` knows nothing
about anything outside it, and the outer shell knows about everything.

What this buys:

- The vault, the embedding model, store, fusion, reranking and contextualisation all sit
  behind ports.
  Swapping the English embedding model for a multilingual one, or adding the whole BM25 + RRF
  + reranking stack, touched no use-case code.
- Use cases run against in-memory fakes, so the fast test suite needs no SQLite file and no
  model download.
- `config/container.py` is the only module that imports concrete adapters, and it injects
  each one into a use case as its narrow role.

A search runs dense retrieval and BM25 in parallel, RRF fuses them into a high-recall pool,
and a cross-encoder reranker picks the final results by query-passage relevance.

[ARCHITECTURE.md](ARCHITECTURE.md) covers every port and adapter, the indexing and search
flows, the config-fingerprint guard, and the reason for each ranking stage.

## MCP tools

| Tool               | Returns                                                            |
| ------------------ | ----------------------------------------------------------------- |
| `status`           | Index health: `notes` and `chunks`; `last_indexed`, when an index run last completed (ISO 8601 UTC, or null), and `config_fingerprint`, the settings it ran with; `last_run`, the last run that ended, with `finished_at`, `notes`, `chunks`, `skipped` and `error`; `running`, whether an index run is going on in any process; and `progress` (done and total notes, embedded and skipped so far) for a run this server started. |
| `reindex`          | Starts indexing the configured vault in the background and returns `{"started": true}` at once, or `{"started": false, "reason": ...}` while another index run is going on. Follow the run with `status`. |
| `search_knowledge` | Most relevant passages with their source notes.                   |
| `search_sources`   | Which notes a concept appears in: hit count, best score, snippets. |
| `get_note`         | A note's title and its text as written in the file, by vault-relative path, for notes the index holds. |

## Example

Once connected, an MCP client calls the tools. Asking for passages about a concept:

```jsonc
// search_knowledge(query="how does reciprocal rank fusion combine results", k=3)
{
  "results": [
    {
      "note_path": "notes/retrieval/hybrid-search.md",
      "heading_path": ["Fusion", "RRF"],
      "text": "Reciprocal Rank Fusion merges two ranked lists by summing 1/(k + rank) for each result across lists, so an item ranked highly by either channel floats up without needing comparable raw scores.",
      "score": 0.94   // reranker relevance score
    },
    {
      "note_path": "notes/retrieval/bm25.md",
      "heading_path": ["Lexical search"],
      "text": "BM25 scores are not comparable to cosine distances, which is why fusion works on ranks rather than raw scores.",
      "score": 0.61
    }
  ]
}
```

`search_sources` answers the other question, which notes discuss a concept, and returns notes
with hit counts, best score and snippets.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Configure: point at your vault
cp ariostea.example.toml ariostea.toml   # then edit [vault] path

# 2. Build the index (the first run downloads the models, see First run)
uv run ariostea reindex

# 3. Run the stdio MCP server
uv run ariostea serve

# Other commands
uv run ariostea watch                    # index, then auto-reindex on vault edits
uv run ariostea status                   # print index health
uv run pytest -m "not integration"       # fast test suite
```

Every command reads `ariostea.toml` from the current directory unless `--config` names
another file. Only `[vault].path` is required, and every other value has a built-in default.
[`ariostea.example.toml`](ariostea.example.toml) lists all options and defaults.

After an upgrade that changes the embedding model or the chunking, run `uv run ariostea
reindex` once. When the new model's vectors have a different size, the server answers from
BM25 alone until that reindex. If the reindex stops partway, the next one picks up where it
stopped; until one finishes, the index holds notes from both settings and search quality is
mixed. Fixes to how a note is cut into chunks (code fences, long unbroken strings such as
inline images) apply to each note the next time it changes. For a full rebuild, stop any
server or `watch` using the index, delete the index file together with its `-wal` and `-shm`
files (`rm ~/.ariostea/index.db*`), and run `reindex`.

One index run goes on at a time, whichever process starts it. While one runs, the `reindex`
tool starts nothing and says so, and `ariostea reindex` and each `watch` pass print `waiting
for the index run in progress`, then start when it ends. Searches, `get_note` and `status`
are answered throughout, from the index as it is. A run that stops partway, after a crash or
Ctrl-C, keeps every note it finished, and the next run embeds only the rest.

Three files sit next to the index file: `index.db-wal` and `index.db-shm`, sqlite's
write-ahead log, which lets searches read while a run writes, and `index.db.lock`, which a
run locks while it goes on. The operating system releases the lock when the process ends,
however it ends, and the file stays.

### First run

The first run downloads about 3.2 GB of models: the `BAAI/bge-m3` embedding model (2.1 GB)
and the `jina-reranker-v2-base-multilingual` reranker (1.1 GB). They go to
`~/.cache/fastembed`, or to `FASTEMBED_CACHE_PATH` when it is set, and later runs load them
from there.

Run `uv run ariostea reindex` once from a terminal before connecting an MCP client. The
download and the first index then happen where their progress shows. A client that starts the
server on a fresh install waits for the download too, and may give up before it ends.

`reindex` prints a progress line on stderr every 100 notes or every 10 seconds, a warning for
each file it skipped, and a summary:

```text
38/50 notes · 34 embedded · 1 skipped
50/50 notes · 46 embedded · 1 skipped
warning: skipped Legacy/Vecchia nota latin1.md: not UTF-8 text
Indexed 46 notes, 239 chunks, 1 skipped.
```

`watch` prints the same progress lines and warnings for its first pass, and one line for each
pass that changed something, such as `+2 updated, -1 deleted`.

### What gets indexed

Every `.md` file in the vault, except:

- files and folders whose name starts with a dot, at any depth: `.obsidian/`, `.git/`,
  `.trash/`;
- Excalidraw drawings, `*.excalidraw.md`;
- paths that start with a prefix listed in `vault.ignore`, such as `"Templates/"`.

A file that cannot be read (not UTF-8 text, permission denied) is skipped with a warning that
names it, and the run carries on. If the vault folder is missing, or holds no notes while the
index holds some (an iCloud vault not synced yet, the wrong folder), the command stops with
an error and leaves the index as it was. To empty an index on purpose, stop any server or
`watch` using it and delete the index file together with its `-wal` and `-shm` files
(`rm ~/.ariostea/index.db*`).

### Memory and speed

Measured on a Mac, on CPU: a running server holds about 5 GB of memory, and a search takes
about 2.5 s. Nearly all of both is the reranker. With `rerank.enabled = false` the server
holds 1.7 GB and a search takes about 20 ms, at a large cost in ranking quality (Trial 1
below compares the two).

### Connecting an MCP client

For Claude Code or Claude Desktop, register the stdio server:

```bash
claude mcp add ariostea -- uv run --directory /path/to/ariostea ariostea serve
```

`--directory` makes uv run the server from the repository folder, where it finds the project
and `ariostea.toml`. To keep the config somewhere else, add `--config /path/to/ariostea.toml`
after `serve`.

For HTTP clients, n8n for example, serve over Streamable HTTP instead of stdio:

```bash
uv run ariostea serve --transport http --host 0.0.0.0 --port 8000
# MCP endpoint: http://<host>:8000/mcp
```

## Development

```bash
uv sync                             # install deps
uv run pytest -m "not integration"  # fast unit suite (no model downloads)
uv run pytest                       # full suite, incl. tests that load real models
uv run ruff check .                 # lint
uv run ruff format .                # format
```

## The learning journey

The pipeline was built first and measured after. Each stage (hybrid retrieval, reranking,
contextual retrieval) is a known technique, and the question was whether it holds up on real
material. Answering it needed an evaluation that can detect a change of a few cases. Every
number below comes from `eval/results/runs.jsonl`. The full account, with every figure and the
reproduction commands, is in [docs/retrieval-tuning.md](docs/retrieval-tuning.md).

### The instrument

`eval/wiki/` holds 79 Wikipedia articles in 6 topic clusters (board games, cheese, coffee,
cycling, sailing, string instruments), pinned to revision IDs so the snapshot rebuilds byte
for byte. Articles in a cluster share vocabulary, so every query has near misses to reject.

`eval/wiki/gold.json` holds 167 queries over that corpus. Each points at the text span that
answers it, not just the note, and a retrieved chunk counts as a hit only when it contains the
whole span. The labels stay valid under any chunking policy, which is what made chunking
measurable at all.

| type | stresses | cases |
|---|---|---|
| paraphrase | dense embeddings | 40 |
| exact_term | BM25 full-text search | 41 |
| buried | facts deep inside a long section | 40 |
| cross_lingual | Italian and Spanish queries over English articles | 46 |

The set is LLM-generated. A local model wrote candidates from 505 passages, a different model
judged them, and they passed five gates: span verification, the adversarial judge, a filter
dropping queries every channel already answers at rank 1, an ambiguity gate, and a human spot
review. 338 candidates were rejected, each with its reason in `eval/wiki/gold_rejected.json`.

Every run goes into `eval/results/runs.jsonl` with its configuration, commit and every metric
per channel and query type, compared against a named control. `eval/results/experiment_log.html`
renders the log as a table shaded green for gains and magenta for losses, grey within ±0.03,
the smallest change 167 queries can resolve, about 5 cases.

A decision rule was fixed before any result was in. A change replaces a default only if hybrid
span recall gains at least 0.03, no query type loses more than 0.05, and at least 160 of 167
answers stay reachable.

### Baseline

Span recall at k=5 with the original chunking, 512 words with no overlap:

| channel | span recall | cross_lingual |
|---|---|---|
| dense | 0.335 | 0.391 |
| sparse (BM25) | 0.605 | 0.022 |
| hybrid (fused, reranked) | 0.832 | 0.630 |

Hybrid is what the server serves. Dense at 0.335 was far too low for a multilingual embedding
model on this material.

### Trial 1: chunking

Measuring what the embedding model reads explained the dense number. The size cap counted
words, the model counts subword tokens, and Italian and Spanish run well over one token per
word. 299 chunks, 1 in 5, were longer than the model's 512-token input, and fastembed
truncates silently, so the dense channel embedded only the start of those. 15 answers sat past
the cut, unreachable by dense retrieval at any rank.

The fix was to size chunks in model tokens, with overlap so no answer is cut in half. A sweep
of 9 policies ran dense and sparse first, then hybrid for the 3 best, then a confirmation run,
then the same again with the reranker switched off.

| policy | dense | hybrid | hybrid cross_lingual | fused (no reranker) |
|---|---|---|---|---|
| 512 words (control) | 0.335 | 0.832 | 0.630 | 0.551 |
| 160 tokens, overlap 40 | 0.545 | 0.880 (+0.048) | 0.696 | 0.629 (+0.078) |

<img src="docs/images/chunking-sweep-hybrid.png" alt="Chunking sweep, hybrid span recall at k=5, shaded by change against the 512-word control" width="720">

160 tokens with 40 of overlap was the only policy that passed the rule with the reranker and
without it, and it became the default until Trial 4 moved to 256 tokens with 64. The hybrid gain is probably smaller than +0.048, since
the neighbouring policies gained +0.006 to +0.018, so part of it is luck. Without the reranker
the gain is large and steady, +0.06 to +0.09 for every policy, because the reranker had been
absorbing most of the damage the word cap did. The change is a fix for silent truncation more
than a tuning win, and upgrading re-chunks and re-embeds a vault once.

### Trial 2: note-level blurbs

Contextual indexing asks an LLM for one short blurb per note, its topic and key entities, and
prepends it to every chunk before embedding and indexing. On the earlier eval corpus, 19 short
documents of one chunk each, it had lifted dense and sparse retrieval, while hybrid stayed
flat because the reranker scored the raw chunk. The trial asked two questions: do blurbs help
on the wiki set, and does letting the reranker see them help hybrid.

| configuration | dense | hybrid | hybrid cross_lingual |
|---|---|---|---|
| no blurbs (control) | 0.545 | 0.880 | 0.696 |
| blurbs, reranker scores the raw chunk | 0.503 | 0.844 (−0.036) | 0.609 |
| blurbs, reranker scores blurb and chunk | | 0.814 (−0.066) | 0.609 |

Both fail the rule, so blurbs stay off, and the dense channel shows why. Note recall went up,
0.826 to 0.850, so the blurb helps find the right article. Span recall went down, and on the
29 cases the gold set had dropped as too easy it fell from 0.931 to 0.759. A wiki article here
is dozens of chunks that all carry the same blurb, so their vectors move toward each other and
the right chunk is harder to pick out of its own article. The earlier lift came from one-chunk
notes, where that cannot happen.

### Trial 3: per-chunk context

Anthropic's Contextual Retrieval writes the context per chunk instead. The prompt holds the
whole document and one chunk, and asks for a short text that situates that chunk. Every chunk
gets a different context, which targets the convergence above. All 4,246 chunks got one, from
a local `qwen2.5-7b-instruct` in three and a half hours. The 14B model at a 32k context ran
the Mac out of GPU memory and panicked it on both attempts.

| configuration | dense | fused (no reranker) | hybrid | hybrid cross_lingual |
|---|---|---|---|---|
| no context (control) | 0.545 | 0.629 | 0.880 | 0.696 |
| per-chunk context, reranker scores the raw chunk | 0.581 (+0.036) | 0.665 (+0.036) | 0.850 (−0.030) | 0.630 |
| per-chunk context, reranker sees the context | | | 0.856 (−0.024) | 0.652 |

It fixes what the blurbs broke. Dense goes up instead of down, the easy cases stay at 0.931,
and every stage before the reranker gains. The gain does not survive the reranker. The
cross-encoder already reads chunk and query together, so the context adds little it lacks,
and the changed candidate pool costs a few cases, 3 of them cross-lingual. Hybrid fails the
rule either way, so contextual indexing stays off. The per-chunk option stays in the code for
a setup without a reranker, where it gains +0.036.

### Trial 4: the embedding model

Cross-lingual was the one weak type, and BM25 finds 1 of its 46 cases, so dense retrieval is
the only first-stage channel that works there. Three newer multilingual models were measured
against the one every earlier run used, all at 160 tokens with overlap 40.

The first run turned up a problem older than the trial. The vector store ranks by Euclidean
distance, and fastembed returns unnormalised vectors for the old model, so every dense number
above had ranked raw vectors, not by the cosine similarity the model was trained for. A new
setting, `embedding.normalize`, scales vectors to length 1, and the trial measured it as its
own factor.

Hybrid hits out of 167, cross-lingual out of 46:

| configuration | dense | hybrid | hybrid cross_lingual | one query |
|---|---|---|---|---|
| previous model, raw vectors (control) | 0.545 | 147 | 32 | 4.8 ms |
| previous model, unit vectors | 0.593 | 150 | 35 | 4.9 ms |
| `intfloat/multilingual-e5-large` | 0.778 | 154 | 39 | 14.7 ms |
| `jinaai/jina-embeddings-v3` | 0.832 | 155 | 41 | 116.3 ms |
| `BAAI/bge-m3` | 0.856 | 155 | 41 | 14.3 ms |
| `BAAI/bge-m3`, 256 tokens, overlap 64 | 0.880 | 157 | 41 | 15.2 ms |

Normalisation alone gains 3 cases and fails the rule. bge-m3 and jina-v3 gain 5 over the
normalised control, which is 0.0299 against a rule written as 0.03 but set as "about 5
cases". The reading was settled after the data, as 5 cases, so both pass. bge-m3 won on
cost: the same recall as jina-v3 at an eighth of the query time, under MIT where jina-v3 is
non-commercial. The 160-token chunks had been sized for the old model's 128-token training,
so the trial measured bge-m3 at 256 tokens too, and it found 2 more cases with no type worse.

Switching the default exposed two upgrade bugs on a real vault. An existing index could not
take vectors of a new size, and an index built before the late-September blurb work lacked a
column the store now writes. Both broke the next reindex. The store now rebuilds the vector table and adds missing
columns when it opens.

### The defaults today

`BAAI/bge-m3` embeddings, 256-token chunks with 64 of overlap, the
`jina-reranker-v2-base-multilingual` reranker on, contextual indexing off. bge-m3 returns unit
vectors itself, so `normalize` stays off. Hybrid span recall is 0.940, 157 of 167, against
0.880 before the trial and 0.832 at the baseline. Cross-lingual is 0.891, 41 of 46.

The cost is a larger model: a 2.27 GB download against 1 GB, about three times the indexing
time, and 15 ms to embed a query against 5. That figure is query embedding alone; a whole
search, reranker included, takes about 2.5 s (see Memory and speed). With dense retrieval this strong, fusing in BM25
now lowers the first stage on cross-lingual queries, and the reranker hides it. That, and the
10 cases still missed, are the next things to look at.

## Licence

[MIT](LICENSE).

`eval/wiki/` is different: it holds Wikipedia article text used as a retrieval-evaluation
corpus, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and
modified (converted to Markdown, links rewritten as wikilinks, infoboxes and references
stripped). `eval/wiki/NOTICE` records the licence, the modifications, and a permalink to the
exact revision of every article in the snapshot. Redistributing that text, or adaptations of
it, carries the share-alike obligation. The surrounding source code remains MIT.

The snapshot is rebuilt with:

```bash
uv run python eval/build_wiki_corpus.py   # re-fetches every pinned revision
```

Because every article is pinned to a revision ID in `eval/wiki/clusters.json`, that command
reproduces the committed corpus byte for byte.
