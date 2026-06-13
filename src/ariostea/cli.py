from __future__ import annotations

import typer

app = typer.Typer(help="Ariostea — Obsidian RAG MCP server")


@app.command()
def serve(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Run the MCP server (full wiring lands in Task 1.8)."""
    typer.echo(f"ariostea serve — config={config} (not yet wired)")


@app.command()
def main_placeholder() -> None:  # keeps the module importable before wiring
    typer.echo("ok")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
