import asyncio, json, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
CLONE = Path(__file__).resolve().parent / "clone"
async def main(qs):
    p = StdioServerParameters(command="uv", args=["run", "--directory", str(CLONE), "ariostea", "serve"], cwd="/")
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for q in qs:
                res = await s.call_tool("search_knowledge", {"query": q, "k": 3})
                d = json.loads(res.content[0].text)
                print(repr(q), [(x["note_path"], round(x["score"], 2)) for x in d["results"]])
asyncio.run(main(sys.argv[1:]))
