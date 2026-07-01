# HTTP MCP Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `ariostea serve` also run over Streamable HTTP (`serve --transport http`) so n8n can reach the MCP tools at `http://127.0.0.1:8000/mcp`, with stdio remaining the default.

**Architecture:** Ride FastMCP's built-in `streamable-http` transport (SDK `mcp` 1.27.2). Add a `[server]` config section (host/port), thread host/port into `build_server`, add a pure CLI-name→SDK-name transport mapping, and wire a `--transport/--host/--port` flag onto the existing `serve` command. No tools, handlers, ports, or use cases change.

**Tech Stack:** Python, FastMCP (`mcp.server.fastmcp`), Typer CLI, pydantic config, pytest.

**Spec:** `docs/superpowers/specs/2026-07-01-ariostea-http-mcp-transport-design.md`

**Refinement vs spec §7 (read before starting):** The spec described the integration test as an in-process ASGI `initialize` handshake. That would require running the ASGI lifespan (an extra test dependency like `asgi-lifespan`) for a localhost-only feature — disproportionate. Verified simpler equivalents that need no socket and no lifespan: `build_server(...).streamable_http_app()` exposes a Starlette route with path `/mcp`, and `await build_server(...).list_tools()` returns the 5 tools. The plan tests both (route mounted + tools served) as a normal fast test. Full end-to-end is covered by the manual n8n acceptance step. This is the only deviation from the spec.

**Conventions (this repo):**
- Flat tests: NO `tests/**/__init__.py`; unique test-file basenames.
- Every source file starts with `from __future__ import annotations`.
- `uv run pytest -m "not integration" -q` is the fast suite; `uv run ruff check .` must stay clean.
- Config models are pydantic `BaseModel` in `src/ariostea/config/schema.py`.

**Verified SDK facts (`mcp` 1.27.2):** `FastMCP(name, *, host="127.0.0.1", port=8000, streamable_http_path="/mcp", ...)`; `FastMCP.run(transport="stdio"|"sse"|"streamable-http")`; `mcp.settings.host/port/streamable_http_path`; `streamable_http_app()` → Starlette app with a `/mcp` route; `await mcp.list_tools()` → registered tools.

---

### Task 1: `[server]` config section

**Files:**
- Modify: `src/ariostea/config/schema.py`
- Modify: `ariostea.example.toml`
- Test: `tests/config/test_schema.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/config/test_schema.py`:

```python
def test_server_defaults_are_localhost_8000(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8000


def test_server_section_parses(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[server]\nhost = "0.0.0.0"\nport = 9001\n')
    cfg = load_config(cfg_file)
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 9001
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_schema.py -k server -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'server'`.

- [ ] **Step 3: Add `ServerCfg` to `schema.py`**

In `src/ariostea/config/schema.py`, add the model after the existing `ContextualCfg` class:

```python
class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
```

And add the field to `Config` (alongside the existing `contextual` field):

```python
    server: ServerCfg = ServerCfg()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_schema.py -k server -v`
Expected: PASS (both tests).

- [ ] **Step 5: Document `[server]` in `ariostea.example.toml`**

Append this block to `ariostea.example.toml`:

```toml
[server]
# HTTP transport bind address for `ariostea serve --transport http`.
# Localhost-only by default; the n8n MCP Client URL is http://<host>:<port>/mcp
host = "127.0.0.1"
port = 8000
```

- [ ] **Step 6: Verify suite + lint**

Run: `uv run pytest -m "not integration" -q && uv run ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/ariostea/config/schema.py ariostea.example.toml tests/config/test_schema.py
git commit -m "feat(config): add [server] host/port section"
```

---

### Task 2: `build_server` host/port + `resolve_transport` + HTTP-app serves tools

**Files:**
- Modify: `src/ariostea/mcp/server.py`
- Test: `tests/mcp/test_server.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/mcp/test_server.py`:

```python
import pytest

from ariostea.mcp.server import build_server, resolve_transport


def test_resolve_transport_maps_cli_names():
    assert resolve_transport("stdio") == "stdio"
    assert resolve_transport("http") == "streamable-http"


def test_resolve_transport_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_transport("grpc")


def test_build_server_applies_host_and_port():
    mcp = build_server(object(), host="0.0.0.0", port=9001)
    assert mcp.settings.host == "0.0.0.0"
    assert mcp.settings.port == 9001


def test_build_server_defaults_to_localhost_8000():
    mcp = build_server(object())
    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.port == 8000


def test_http_app_mounts_mcp_route_and_serves_tools():
    import asyncio

    mcp = build_server(object())
    app = mcp.streamable_http_app()
    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)

    tool_names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "search_knowledge" in tool_names
    assert "get_note" in tool_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/mcp/test_server.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_transport'` (and, once that's added, the host/port test fails until `build_server` accepts the kwargs).

- [ ] **Step 3: Update `src/ariostea/mcp/server.py`**

Add `resolve_transport` above `build_server`:

```python
def resolve_transport(name: str) -> str:
    """Map a CLI transport name to the FastMCP transport string."""
    mapping = {"stdio": "stdio", "http": "streamable-http"}
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"unknown transport {name!r}; choose 'stdio' or 'http'") from None
```

Change the `build_server` signature and the `FastMCP(...)` construction (leave every `@mcp.tool()` block exactly as-is):

```python
def build_server(container: "Container", host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP("ariostea", host=host, port=port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/mcp/test_server.py -v`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Verify suite + lint**

Run: `uv run pytest -m "not integration" -q && uv run ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/mcp/server.py tests/mcp/test_server.py
git commit -m "feat(mcp): build_server host/port + resolve_transport helper"
```

---

### Task 3: `serve --transport http` CLI wiring

**Files:**
- Modify: `src/ariostea/cli.py`
- Test: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py` (note: `SimpleNamespace` and `CliRunner` are already imported at the top of this file):

```python
def test_serve_http_runs_streamable_transport(monkeypatch):
    fake_cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(cli, "load_config", lambda path: fake_cfg)
    monkeypatch.setattr(cli, "build_container", lambda cfg: object())

    recorded = {}

    class FakeServer:
        def run(self, transport="stdio"):
            recorded["transport"] = transport

    def fake_build_server(container, host="127.0.0.1", port=8000):
        recorded["host"] = host
        recorded["port"] = port
        return FakeServer()

    monkeypatch.setattr(cli, "build_server", fake_build_server)

    result = CliRunner().invoke(
        cli.app, ["serve", "--transport", "http", "--port", "9009", "--config", "x.toml"]
    )

    assert result.exit_code == 0
    assert recorded["transport"] == "streamable-http"
    assert recorded["port"] == 9009  # CLI overrides config default
    assert recorded["host"] == "127.0.0.1"  # falls back to config


def test_serve_stdio_is_the_default(monkeypatch):
    fake_cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(cli, "load_config", lambda path: fake_cfg)
    monkeypatch.setattr(cli, "build_container", lambda cfg: object())

    recorded = {}

    class FakeServer:
        def run(self, transport="stdio"):
            recorded["transport"] = transport

    monkeypatch.setattr(cli, "build_server", lambda container, host="127.0.0.1", port=8000: FakeServer())

    result = CliRunner().invoke(cli.app, ["serve", "--config", "x.toml"])

    assert result.exit_code == 0
    assert recorded["transport"] == "stdio"  # run() called with no HTTP transport


def test_serve_rejects_unknown_transport(monkeypatch):
    fake_cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(cli, "load_config", lambda path: fake_cfg)
    monkeypatch.setattr(cli, "build_container", lambda cfg: object())
    monkeypatch.setattr(cli, "build_server", lambda container, host="127.0.0.1", port=8000: object())

    result = CliRunner().invoke(cli.app, ["serve", "--transport", "grpc", "--config", "x.toml"])

    assert result.exit_code != 0  # typer.BadParameter -> non-zero exit
```

Note the stdio test asserts `recorded["transport"] == "stdio"`: the implementation calls `server.run(transport="stdio")` explicitly (not a bare `server.run()`) so the default path is observable. Match that in the implementation below.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k serve -v`
Expected: FAIL (serve has no `--transport` option yet / unexpected extra args).

- [ ] **Step 3: Update `serve` in `src/ariostea/cli.py`**

Add the import at the top (next to the existing `from ariostea.mcp.server import build_server`):

```python
from ariostea.mcp.server import build_server, resolve_transport
```

Replace the existing `serve` command with:

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
    try:
        sdk_transport = resolve_transport(transport)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    server = build_server(container, host=bind_host, port=bind_port)
    if sdk_transport != "stdio":
        typer.echo(f"Serving MCP over HTTP at http://{bind_host}:{bind_port}/mcp")
    server.run(transport=sdk_transport)
```

(`FastMCP.run` accepts `transport="stdio"` explicitly, so a single `server.run(transport=sdk_transport)` covers both paths.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k serve -v`
Expected: PASS (all three serve tests).

- [ ] **Step 5: Verify full suite + lint**

Run: `uv run pytest -m "not integration" -q && uv run ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/cli.py tests/test_cli.py
git commit -m "feat(cli): serve --transport http (Streamable HTTP for n8n)"
```

---

### Task 4: Manual acceptance (n8n)

No code — verify the real end-to-end path. **Requires the user** to drive n8n; if unavailable, report and stop.

- [ ] **Step 1: Start the HTTP server**

Run: `uv run ariostea serve --transport http --config ariostea.toml`
Expected: prints `Serving MCP over HTTP at http://127.0.0.1:8000/mcp` and stays running.

- [ ] **Step 2: Smoke-check the endpoint is listening**

In another shell: `curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/mcp`
Expected: a non-connection-refused HTTP status (e.g. 400/406 for a bare GET is fine — it proves the listener is up; a full MCP client sends the proper headers).

- [ ] **Step 3: Point n8n at it**

In n8n's **MCP Client** node, choose the **HTTP Streamable** transport and set the URL to `http://127.0.0.1:8000/mcp`. Confirm the tool list shows `search_knowledge`, `search_sources`, `get_note`, `status`, `reindex`, and that a `search_knowledge` call returns results.

- [ ] **Step 4: Stop the server** (Ctrl-C).

---

## Self-Review

**1. Spec coverage:**
- Spec §4.1 `[server]` config → Task 1. ✓
- Spec §4.2 `build_server(host, port)` → Task 2. ✓
- Spec §4.3 `resolve_transport` → Task 2. ✓
- Spec §4.4 CLI `serve --transport/--host/--port`, config defaults + CLI override, prints URL → Task 3. ✓
- Spec §6 error handling: unknown transport → `typer.BadParameter` → Task 3 test `test_serve_rejects_unknown_transport`. ✓
- Spec §7 testing: config defaults/parse (Task 1), resolve_transport + host/port applied (Task 2), HTTP app mounts /mcp + serves tools (Task 2, refined per note above), manual n8n (Task 4). ✓
- Spec §5 data flow / §8 out-of-scope: no code (documented). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**3. Type consistency:** `build_server(container, host, port)` defined in Task 2 matches its Task 3 call site and the Task 2 tests. `resolve_transport(name) -> str` defined in Task 2, imported and used in Task 3. `cfg.server.host` / `cfg.server.port` from Task 1's `ServerCfg` match Task 3's usage and the CLI test's `fake_cfg`. FastMCP kwargs (`host`, `port`) and `run(transport=...)` match the verified SDK signature. ✓
