"""AC-20: the awkward vault from the vault-hardening usability test, end to
end. The vault is built and indexed once per module through the real CLI, then
driven through the real stdio MCP server and `ariostea watch`, all on the
default models. The first index takes about a minute."""

from __future__ import annotations

import asyncio
import json
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixtures.awkward_queries import EXCLUDED, FRONTMATTER_ONLY, KNOWN_ANSWERS
from fixtures.awkward_vault import CLIP, CLIP_QUERY, build
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import LATEST_PROTOCOL_VERSION

from ariostea.config.schema import ChunkingCfg, EmbeddingCfg

pytestmark = pytest.mark.integration

ARIOSTEA = [sys.executable, "-m", "ariostea.cli"]
GET_NOTE_PATHS = ["Cucina/Caffè è buono.md", "🚀 Launch plan.md", "日本語/日本語メモ.md"]


@pytest.fixture(scope="module")
def awkward(tmp_path_factory):
    root = tmp_path_factory.mktemp("awkward")
    vault = build(root)
    store = root / "index.db"
    config = root / "ariostea.toml"
    config.write_text(f'[vault]\npath = "{vault}"\n\n[store]\npath = "{store}"\n')
    reindex = subprocess.run(
        [*ARIOSTEA, "reindex", "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    return SimpleNamespace(vault=vault, store=store, config=config, reindex=reindex)


def _over_mcp(config: Path, calls):
    """Start the stdio server on `config`, run `calls(call)` and return its
    result; `call(tool, **args)` returns the tool's JSON payload."""

    async def main():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "ariostea.cli", "serve", "--config", str(config)]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                async def call(tool, **args):
                    result = await session.call_tool(tool, args)
                    assert not result.isError, result
                    return json.loads("".join(c.text for c in result.content if c.type == "text"))

                return await calls(call)

    return asyncio.run(main())


@pytest.fixture(scope="module")
def answers(awkward):
    """Every search and get_note the tests below check, in one server session."""

    async def calls(call):
        out = {"knowledge": {}, "sources": {}, "notes": {}}
        markers = [*EXCLUDED.values(), *FRONTMATTER_ONLY]
        for query in [q for q, _ in KNOWN_ANSWERS] + [CLIP_QUERY] + markers:
            out["knowledge"][query] = (await call("search_knowledge", query=query, k=10))["results"]
        for marker in markers:
            out["sources"][marker] = (await call("search_sources", query=marker, k=10))["sources"]
        for path in [*GET_NOTE_PATHS, "../../etc/passwd"]:
            out["notes"][path] = await call("get_note", path=path)
        return out

    return _over_mcp(awkward.config, calls)


def test_reindex_completes_and_names_bad_files(awkward):
    assert awkward.reindex.returncode == 0, awkward.reindex.stderr
    assert (
        "warning: skipped Legacy/Vecchia nota latin1.md: not UTF-8 text" in awkward.reindex.stderr
    )
    assert re.search(r"Indexed \d+ notes, \d+ chunks, 1 skipped\.", awkward.reindex.stdout)


def test_excluded_files_are_not_searchable(answers):
    for marker in [*EXCLUDED.values(), *FRONTMATTER_ONLY]:
        passages = answers["knowledge"][marker]
        assert not {r["note_path"] for r in passages} & set(EXCLUDED), marker
        assert not [r for r in passages if marker in r["text"]], marker
        sources = answers["sources"][marker]
        assert not {s["note_path"] for s in sources} & set(EXCLUDED), marker
        assert not [s for s in sources if any(marker in snip for snip in s["snippets"])], marker


@pytest.mark.parametrize(("query", "path"), KNOWN_ANSWERS)
def test_known_answer_queries_rank_first(answers, query, path):
    results = answers["knowledge"][query]
    assert results and results[0]["note_path"] == path, [r["note_path"] for r in results[:3]]


def test_long_base64_note_indexes_within_budget(awkward, answers):
    from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings

    db = sqlite3.connect(awkward.store)
    texts = [
        row[0]
        for row in db.execute(
            "SELECT c.text FROM chunks c JOIN notes n ON n.id = c.note_id WHERE n.path = ?", (CLIP,)
        )
    ]
    db.close()
    count = FastEmbedEmbeddings(EmbeddingCfg().local_model).count_tokens
    assert len(texts) > 1
    # The budget counts the model's two special tokens; chunk text excludes them.
    assert max(count(t) for t in texts) <= ChunkingCfg().max_tokens - 2
    assert answers["knowledge"][CLIP_QUERY][0]["note_path"] == CLIP


def test_get_note_returns_files(awkward, answers):
    for path in GET_NOTE_PATHS:
        note = answers["notes"][path]
        assert note["found"], path
        assert note["text"] == (awkward.vault / path).read_text(encoding="utf-8"), path
    assert answers["notes"]["../../etc/passwd"]["found"] is False


def _read_lines(stream, lines: queue.Queue) -> None:
    for line in stream:
        lines.put(line)


def test_stdio_stdout_is_protocol_only_during_reindex(awkward):  # AC-16
    note = awkward.vault / "Hobbies" / "Photography.md"
    note.write_text(note.read_text() + "\nGolden hour light is warm and low.\n")
    server = subprocess.Popen(
        [*ARIOSTEA, "serve", "--config", str(awkward.config)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_read_lines, args=(server.stdout, lines), daemon=True).start()

    def send(message):
        server.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        server.stdin.flush()

    def response(request_id):
        while True:
            line = lines.get(timeout=600)
            seen.append(line)
            message = json.loads(line)  # anything that is not protocol fails here
            if message.get("id") == request_id:
                return message

    seen: list[str] = []
    try:
        send(
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            }
        )
        response(1)
        send({"method": "notifications/initialized"})
        send({"id": 2, "method": "tools/call", "params": {"name": "reindex", "arguments": {}}})
        result = response(2)["result"]
    finally:
        server.terminate()
        server.wait(timeout=30)
    payload = json.loads(result["content"][0]["text"])
    assert payload["notes"] > 0 and payload["skipped"]
    while not lines.empty():
        json.loads(lines.get())


def _first_pass_done(line: str) -> bool:
    match = re.match(r"(\d+)/([\d,]+) notes", line)
    return bool(match) and int(match[1]) == int(match[2].replace(",", ""))


def _chunks_containing(store: Path, word: str) -> int:
    try:
        db = sqlite3.connect(store)
        try:
            return db.execute(
                "SELECT count(*) FROM chunks WHERE text LIKE ?", (f"%{word}%",)
            ).fetchone()[0]
        finally:
            db.close()
    except sqlite3.OperationalError:  # the watcher is writing
        return 0


def test_watch_picks_up_an_edit(awkward):
    watcher = subprocess.Popen(
        [*ARIOSTEA, "watch", "--config", str(awkward.config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_read_lines, args=(watcher.stderr, lines), daemon=True).start()
    try:
        while not _first_pass_done(lines.get(timeout=600)):
            pass
        time.sleep(1)  # the file watcher starts right after the first pass
        note = awkward.vault / "Hobbies" / "Birdwatching.md"
        note.write_text(note.read_text() + "\nThe attic zither carries a ZITHERMARKER_17 label.\n")
        deadline = time.monotonic() + 10
        while (
            not _chunks_containing(awkward.store, "ZITHERMARKER_17") and time.monotonic() < deadline
        ):
            time.sleep(0.25)
        assert _chunks_containing(awkward.store, "ZITHERMARKER_17"), "not indexed within 10 s"
    finally:
        watcher.terminate()  # by PID, never by name
        watcher.wait(timeout=30)

    async def calls(call):
        return (await call("search_knowledge", query="ZITHERMARKER_17 attic zither", k=3))[
            "results"
        ]

    results = _over_mcp(awkward.config, calls)
    assert results and results[0]["note_path"] == "Hobbies/Birdwatching.md"
