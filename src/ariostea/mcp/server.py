from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from ariostea.mcp.handlers import (
    get_note_payload,
    reindex_payload,
    search_payload,
    search_sources_payload,
    status_payload,
)

if TYPE_CHECKING:
    from ariostea.config.container import Container


def resolve_transport(name: str) -> str:
    """Map a CLI transport name to the FastMCP transport string."""
    mapping = {"stdio": "stdio", "http": "streamable-http"}
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"unknown transport {name!r}; choose 'stdio' or 'http'") from None


def build_server(container: "Container", host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP("ariostea", host=host, port=port)

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

    @mcp.tool()
    def search_sources(query: str, k: int = 10) -> dict:
        """Find which notes a concept appears in. Returns notes with hit counts, best score, and snippets."""
        return search_sources_payload(container, query=query, k=k)

    @mcp.tool()
    def get_note(path: str) -> dict:
        """Fetch a full note's reconstructed text and title by its vault-relative path."""
        return get_note_payload(container, path=path)

    return mcp
