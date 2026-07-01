# HTTP MCP Transport — Design

**Status:** Approved (design phase)
**Date:** 2026-07-01

## 1. Motivation

The Ariostea MCP server currently runs only over **stdio** (`serve` → `FastMCP.run()`), which suits a local MCP client that spawns the process. To drive the same tools from **n8n**, the server must also be reachable over **HTTP**. n8n's MCP Client node speaks the modern **Streamable HTTP** transport, which the installed SDK (`mcp` 1.27.2) supports natively via `FastMCP.run(transport="streamable-http")`. This adds an HTTP entrypoint without touching the tools, use cases, or ports.

## 2. Settled decisions

- **Transport: Streamable HTTP** only (the current MCP standard; SSE is deprecated upstream). SSE is out of scope.
- **Topology: localhost only.** n8n runs on the same machine, so the server binds `127.0.0.1`. No network exposure, therefore **no auth** in this phase.
- **CLI: a flag on the existing `serve` command** — `serve --transport [stdio|http]`, default `stdio` (nothing existing breaks). `--host`/`--port` override the config defaults.
- **Reuse the built-in transport** (approach A): `FastMCP.run(transport="streamable-http")`, not a hand-rolled or externally-mounted ASGI app.

## 3. Verified SDK facts (`mcp` 1.27.2)

- `FastMCP.run(transport="stdio" | "sse" | "streamable-http", mount_path=None)`.
- `FastMCP(name, *, host="127.0.0.1", port=8000, streamable_http_path="/mcp", ...)` — host/port/path are constructor settings.
- Default streamable endpoint path is **`/mcp`**, so the n8n URL is `http://{host}:{port}/mcp`.
- `FastMCP.streamable_http_app()` returns an ASGI app (used only by the integration test).

## 4. Components

### 4.1 Config — `src/ariostea/config/schema.py`
New model, added to `Config`:
```python
class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
```
`Config` gains `server: ServerCfg = ServerCfg()`. Documented in `ariostea.example.toml` under `[server]`. Localhost default keeps the zero-config install unchanged.

### 4.2 Server builder — `src/ariostea/mcp/server.py`
`build_server(container, host="127.0.0.1", port=8000)` constructs `FastMCP("ariostea", host=host, port=port)`. Tool registration is unchanged. Under stdio the host/port are inert; under http they bind the listener. Defaults match FastMCP's own so existing stdio callers that pass nothing behave identically.

### 4.3 Transport mapping helper — `src/ariostea/mcp/server.py`
A pure function isolates the CLI-name → SDK-name mapping so it is unit-testable:
```python
def resolve_transport(name: str) -> str:
    """Map a CLI transport name to the FastMCP transport string."""
    mapping = {"stdio": "stdio", "http": "streamable-http"}
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"unknown transport {name!r}; choose 'stdio' or 'http'")
```

### 4.4 CLI — `src/ariostea/cli.py`
`serve` gains options (config supplies defaults; CLI overrides):
```python
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
    server = build_server(container, host=bind_host, port=bind_port)
    try:
        sdk_transport = resolve_transport(transport)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if sdk_transport == "stdio":
        server.run()
    else:
        typer.echo(f"Serving MCP over HTTP at http://{bind_host}:{bind_port}/mcp")
        server.run(transport=sdk_transport)
```

## 5. Data flow

n8n MCP Client (HTTP Streamable, URL `http://127.0.0.1:8000/mcp`) → FastMCP streamable-http endpoint → the same 5 registered tools → handlers → container use cases. Identical tool behavior to stdio; only the transport differs.

## 6. Error handling

- Unknown `--transport` → `typer.BadParameter` (via `resolve_transport` raising `ValueError`), a clear CLI error before any server starts.
- Port already in use → uvicorn/anyio raises on `run()`; the traceback names the address, which is actionable. No custom wrapping this phase.
- Vault/index/model errors are unchanged — they surface at tool-call time exactly as under stdio.

## 7. Testing

**Fast suite (TDD):**
- `ServerCfg` defaults (`127.0.0.1`, `8000`) and load from a `[server]` toml block.
- `resolve_transport`: `"stdio" → "stdio"`, `"http" → "streamable-http"`, unknown → `ValueError`.
- `build_server(container, host=..., port=...)` applies to `mcp.settings.host` / `mcp.settings.port`.

**Integration (`integration`-marked):**
- Drive `build_server(...).streamable_http_app()` with an in-process ASGI client (httpx `ASGITransport`) through the MCP `initialize` handshake, then `tools/list`, and assert `search_knowledge` is among the tools. Proves the HTTP surface actually serves the tools without opening a real socket. `run()` itself (blocking) is verified manually.

**Manual acceptance:**
- `uv run ariostea serve --transport http` prints the URL and serves; pointing n8n's MCP Client (HTTP Streamable) at `http://127.0.0.1:8000/mcp` lists and calls the tools.

## 8. Out of scope (recorded, gated on real need)

- **Auth / bearer tokens** — only needed once the endpoint leaves localhost.
- **Non-localhost binding / remote / Docker** — would require auth + `0.0.0.0` binding; revisit if n8n moves off-host.
- **SSE transport** — deprecated upstream; add only if an older n8n requires it.
- **`stateless_http` / `json_response` modes** — FastMCP defaults (session-based streaming) work for n8n; expose only if needed.
