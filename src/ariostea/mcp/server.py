from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ariostea.mcp.handlers import status_payload


def build_server(admin) -> FastMCP:
    mcp = FastMCP("ariostea")

    @mcp.tool()
    def status() -> dict:
        """Report index health: note/chunk counts, last index time, config fingerprint."""
        return status_payload(admin)

    return mcp
