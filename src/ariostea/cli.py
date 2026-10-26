from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import typer

from ariostea.config.container import build_container
from ariostea.config.schema import ConfigError, load_config
from ariostea.indexing.watch_vault import WatchVault
from ariostea.mcp.handlers import reindex_payload, status_payload
from ariostea.mcp.server import build_server, resolve_transport
from ariostea.ports.vault import VaultUnavailable

app = typer.Typer(help="Ariostea — Obsidian RAG MCP server")


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(1)


@contextmanager
def _one_line_errors() -> Iterator[None]:
    """Print an error the user can fix as one line on stderr and exit 1.
    Anything else keeps its traceback: it is a bug, not a setup problem."""
    try:
        yield
    except (ConfigError, VaultUnavailable) as exc:
        raise _fail(str(exc)) from None


@app.command()
def serve(
    config: str = typer.Option("ariostea.toml", help="Path to config file"),
    transport: str = typer.Option("stdio", help="Transport: stdio or http"),
    host: str = typer.Option(None, help="HTTP bind host (default from config)"),
    port: int = typer.Option(None, help="HTTP bind port (default from config)"),
) -> None:
    """Run the MCP server (stdio by default, or Streamable HTTP for n8n)."""
    with _one_line_errors():
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
    with _one_line_errors():
        container = build_container(load_config(config))
    result = reindex_payload(container)
    if "error" in result:
        raise _fail(result["error"])
    for skip in result["skipped"]:
        typer.echo(f"warning: skipped {skip['path']}: {skip['reason']}", err=True)
    typer.echo(
        f"Indexed {result['notes']} notes, {result['chunks']} chunks, "
        f"{len(result['skipped'])} skipped."
    )


@app.command()
def watch(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Index the vault, then watch for changes and re-index incrementally."""
    with _one_line_errors():
        container = build_container(load_config(config))
        vault = os.path.expanduser(container.config.vault.path)
        typer.echo(f"Indexing and watching {vault} (Ctrl-C to stop)...")
        WatchVault(container.indexer, vault).run()


@app.command()
def status(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Print index status and exit."""
    with _one_line_errors():
        container = build_container(load_config(config))
    typer.echo(status_payload(container.admin))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
