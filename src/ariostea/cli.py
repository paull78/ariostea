from __future__ import annotations

import os

import typer

from ariostea.config.container import build_container
from ariostea.config.schema import load_config
from ariostea.indexing.watch_vault import WatchVault
from ariostea.mcp.handlers import reindex_payload, status_payload
from ariostea.mcp.server import build_server, resolve_transport

app = typer.Typer(help="Ariostea — Obsidian RAG MCP server")


@app.command()
def serve(
    config: str = typer.Option("ariostea.toml", help="Path to config file"),
    transport: str = typer.Option("stdio", help="Transport: stdio or http"),
    host: str = typer.Option(None, help="HTTP bind host (default from config)"),
    port: int = typer.Option(None, help="HTTP bind port (default from config)"),
) -> None:
    """Run the MCP server (stdio by default, or Streamable HTTP for n8n)."""
    cfg = load_config(config)
    container = build_container(cfg)
    bind_host = host if host is not None else cfg.server.host
    bind_port = port if port is not None else cfg.server.port
    try:
        sdk_transport = resolve_transport(transport)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    server = build_server(container, host=bind_host, port=bind_port)
    if sdk_transport != "stdio":
        typer.echo(f"Serving MCP over HTTP at http://{bind_host}:{bind_port}/mcp")
    server.run(transport=sdk_transport)


@app.command()
def reindex(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Index the vault once and exit."""
    container = build_container(load_config(config))
    result = reindex_payload(container)
    typer.echo(f"Indexed {result['notes']} notes, {result['chunks']} chunks.")


@app.command()
def watch(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Index the vault, then watch for changes and re-index incrementally."""
    container = build_container(load_config(config))
    vault = os.path.expanduser(container.config.vault.path)
    typer.echo(f"Indexing and watching {vault} (Ctrl-C to stop)...")
    WatchVault(container.indexer, vault, ignore=container.config.vault.ignore).run()


@app.command()
def status(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Print index status and exit."""
    container = build_container(load_config(config))
    typer.echo(status_payload(container.admin))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
