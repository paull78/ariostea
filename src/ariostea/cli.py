from __future__ import annotations

import typer

from ariostea.config.container import build_container
from ariostea.config.schema import load_config
from ariostea.mcp.handlers import reindex_payload, status_payload
from ariostea.mcp.server import build_server

app = typer.Typer(help="Ariostea — Obsidian RAG MCP server")


@app.command()
def serve(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Run the stdio MCP server."""
    container = build_container(load_config(config))
    server = build_server(container)
    server.run()  # stdio transport


@app.command()
def reindex(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Index the vault once and exit."""
    container = build_container(load_config(config))
    result = reindex_payload(container)
    typer.echo(f"Indexed {result['notes']} notes, {result['chunks']} chunks.")


@app.command()
def status(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Print index status and exit."""
    container = build_container(load_config(config))
    typer.echo(status_payload(container.admin))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
