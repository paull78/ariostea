from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from ariostea.mcp.handlers import status_payload, search_payload, reindex_payload

if TYPE_CHECKING:
    from ariostea.config.container import Container


def build_server(container: "Container") -> FastMCP:
    mcp = FastMCP("ariostea")

    @mcp.tool()
    def status() -> dict:
        """Report index health: note/chunk counts, last index time, config fingerprint."""
        return status_payload(container.admin)

    @mcp.tool()
    def reindex() -> dict:
        """Index (or re-index) the configured vault. Returns note/chunk counts."""
        return reindex_payload(container)

    @mcp.tool()
    def search_knowledge(query: str, k: int = 10) -> dict:
        """Semantic search over the vault. Returns the most relevant passages with their source notes."""
        return search_payload(container, query=query, k=k)

    return mcp
