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
> Clean Architecture skeleton. [LEARNING.md](LEARNING.md) covers how it is built and how each
> stage was measured.

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
      "score": 2.31   // reranker score: unbounded, higher is more relevant
    },
    {
      "note_path": "notes/retrieval/bm25.md",
      "heading_path": ["Lexical search"],
      "text": "BM25 scores are not comparable to cosine distances, which is why fusion works on ranks rather than raw scores.",
      "score": -0.48
    }
  ]
}
```

`search_sources` answers the other question, which notes discuss a concept, and returns notes
with hit counts, best score and snippets. It groups the reranked passages of one search by
note, at most 50 passages or `rerank.pool`, whichever is fewer. With the default pool of 30,
a concept whose passages sit in a few notes comes back with fewer notes than asked for: on the
eval set, 101 of 167 queries return fewer than 10 notes at k=10.

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
```

Every command reads `ariostea.toml` from the current directory unless `--config` names
another file. Only `[vault].path` is required, and every other value has a built-in default.
[`ariostea.example.toml`](ariostea.example.toml) lists all options and defaults.

### First run

The first run downloads about 2.8 GB of models: the `BAAI/bge-m3` embedding model (2.27 GB)
and the `mmarco-mMiniLMv2-L12-H384-v1` reranker (0.47 GB). They go to
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

## Running the index

### Upgrades and rebuilds

After an upgrade that changes the embedding model or the chunking, run `uv run ariostea
reindex` once. When the new model's vectors have a different size, the server answers from
BM25 alone until that reindex. If the reindex stops partway, the next one picks up where it
stopped; until one finishes, the index holds notes from both settings and search quality is
mixed. Fixes to how a note is cut into chunks (code fences, long unbroken strings such as
inline images) apply to each note the next time it changes. For a full rebuild, stop any
server or `watch` using the index, delete the index file together with its `-wal` and `-shm`
files (`rm ~/.ariostea/index.db*`), and run `reindex`.

### One run at a time

One index run goes on at a time, whichever process starts it. While one runs, the `reindex`
tool starts nothing and says so, and `ariostea reindex` and each `watch` pass print `waiting
for the index run in progress`, then start when it ends. Searches, `get_note` and `status`
are answered throughout, from the index as it is. A run that stops partway, after a crash or
Ctrl-C, keeps every note it finished, and the next run embeds only the rest.

Three files sit next to the index file: `index.db-wal` and `index.db-shm`, sqlite's
write-ahead log, which lets searches read while a run writes, and `index.db.lock`, which a
run locks while it goes on. The operating system releases the lock when the process ends,
however it ends, and the file stays.

### What gets indexed

Every `.md` file in the vault, except:

- files and folders whose name starts with a dot, at any depth: `.obsidian/`, `.git/`,
  `.trash/`;
- Excalidraw drawings, `*.excalidraw.md`;
- paths that start with a prefix listed in `vault.ignore`, such as `"Templates/"`.

A file that cannot be read (not UTF-8 text, permission denied) is skipped with a warning that
names it, and the run carries on. If the vault folder is missing, or holds no notes while the
index holds some (an iCloud vault not synced yet, the wrong folder), the command stops with
an error and leaves the index as it was. To empty an index on purpose, delete it as for a full
rebuild, above.

### Memory and speed

Measured on a Mac, on CPU, with a 50-note vault: a running server holds about 3 GB of memory,
and a search takes about 0.6 s. On the 167 queries of the eval set, half the searches take
under 0.6 s and 95% within about 0.7 s. Most of the time is the reranker. With
`rerank.enabled = false` the server holds 1.7 GB and a search takes about 20 ms, at a large
cost in ranking quality (Trial 1 in [LEARNING.md](LEARNING.md) compares the two).

The previous reranker ranks slightly better and is much slower: with
`model = "jinaai/jina-reranker-v2-base-multilingual"` and `pool = 100` under `[rerank]`, the
server holds about 5 GB and a search takes about 2.5 s (Trial 5 in [LEARNING.md](LEARNING.md)
compares the two). Switching reranker needs no reindex.

## Development

```bash
uv sync                             # install deps
uv run pytest -m "not integration"  # fast unit suite (no model downloads)
uv run pytest                       # full suite, incl. tests that load real models
uv run ruff check .                 # lint
uv run ruff format .                # format
```

[ARCHITECTURE.md](ARCHITECTURE.md) covers every port and adapter, the indexing and search
flows, the config-fingerprint guard, and the reason for each ranking stage.

## Licence

[MIT](LICENSE).

`eval/wiki/` is different: it holds Wikipedia article text used as a retrieval-evaluation
corpus, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and
modified (converted to Markdown, links rewritten as wikilinks, infoboxes and references
stripped). `eval/wiki/NOTICE` records the licence, the modifications, and a permalink to the
exact revision of every article in the snapshot. Redistributing that text, or adaptations of
it, carries the share-alike obligation. The surrounding source code remains MIT.

[LEARNING.md](LEARNING.md) describes the corpus and how to rebuild it.
