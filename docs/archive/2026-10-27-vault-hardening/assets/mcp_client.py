"""Drive the real ariostea stdio MCP server with the mcp client library."""
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
CLONE = HERE / "clone"

QUERIES = [
    # (query, expected note substring, kind)
    ("who invented the moka pot and when", "Caffè è buono", "en->it"),
    ("perché i pomodori hanno il marciume apicale", "Tomato watering", "it->en"),
    ("ZX-4471", "Router troubleshooting", "exact term"),
    ("lighthouse keeper who logged ships in the strait", "Maritime history", "buried heading in long note"),
    ("climbing season for Mount Fuji", "日本語メモ", "en->ja"),
    ("quanto deve essere la pressione della caldaia", "Boiler", "it->en"),
    ("should cream go in carbonara", "Carbonara autentica", "en->it"),
    ("late delivery penalty in the Rossi contract", "Clausole", "en->it deep folder"),
    ("how many snapshots does the nightly backup keep", "Backup script", "code-fence note"),
    ("Ferrara salama da sugo", "Vecchia nota latin1", "latin-1 file"),
    ("sourdough starter feeding ratio", "With BOM", "BOM file"),
    ("printer jams with recycled paper", "Windows CRLF", "CRLF file"),
    ("pesto pine nuts basil", "Shared recipe", "symlinked folder"),
    ("TRASHMARKER_77", None, "trash must NOT appear"),
    ("GITMARKER_55", None, ".git must NOT appear"),
    ("QUOKKA_PLUGIN_README", None, ".obsidian must NOT appear"),
    ("CRLFFRONTMATTER_88", None, "frontmatter leak (CRLF)"),
    ("BOMFRONTMATTER_99", None, "frontmatter leak (BOM)"),
    ("heat pump service interval", "Malformed yaml", "malformed frontmatter note"),
]


def text_of(res):
    out = []
    for c in res.content:
        if getattr(c, "type", "") == "text":
            out.append(c.text)
    return "\n".join(out)


async def main(mode):
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(CLONE), "ariostea", "serve"],
        cwd="/",  # deliberately not the clone, as an MCP client would spawn it
    )
    t0 = time.time()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            print(f"initialize ok in {time.time()-t0:.1f}s: server={init.serverInfo.name} {init.serverInfo.version}")
            tools = await s.list_tools()
            for t in tools.tools:
                print(" tool:", t.name, json.dumps(t.inputSchema.get("properties", {})))
            st = await s.call_tool("status", {})
            print("status:", text_of(st))
            if mode == "status":
                return
            if mode in ("queries", "all"):
                for q, exp, kind in QUERIES:
                    t1 = time.time()
                    res = await s.call_tool("search_knowledge", {"query": q, "k": 5})
                    dt = time.time() - t1
                    data = json.loads(text_of(res))
                    paths = [(x["note_path"], x["heading_path"], round(x["score"], 3)) for x in data["results"]]
                    rank = None
                    for i, (p, _, _) in enumerate(paths):
                        if exp and exp in p:
                            rank = i + 1
                            break
                    marker_hit = None
                    if exp is None:
                        token = q
                        marker_hit = [x["note_path"] for x in data["results"] if token in x["text"]]
                    print(f"\n[{kind}] {q!r} ({dt*1000:.0f} ms) expected={exp} rank={rank} marker_in_results={marker_hit}")
                    for p in paths[:5]:
                        print("    ", p)
                    if exp and rank == 1:
                        top = data["results"][0]["text"][:160].replace("\n", " | ")
                        print("     top text:", top)
            if mode in ("sources", "all"):
                res = await s.call_tool("search_sources", {"query": "watering and soil moisture", "k": 5})
                print("\nsearch_sources:", text_of(res)[:1500])
            if mode in ("getnote", "all"):
                for p in ["Cucina/Caffè è buono.md", "日本語/日本語メモ.md", "🚀 Launch plan.md",
                          "Does/not/exist.md", "../../etc/passwd", "/etc/passwd",
                          "../clone/ariostea.toml", "Inbox/Frontmatter only.md", "Legacy/With BOM.md",
                          "Legacy/Windows CRLF.md", "Tech/Scripts/Code first.md"]:
                    res = await s.call_tool("get_note", {"path": p})
                    print(f"\nget_note({p!r}) isError={res.isError}:", text_of(res)[:400])
            if mode == "reindex":
                t1 = time.time()
                res = await s.call_tool("reindex", {})
                print(f"reindex tool ({time.time()-t1:.1f}s):", text_of(res))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
