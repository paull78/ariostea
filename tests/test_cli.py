from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ariostea import cli


def test_watch_command_builds_container_and_runs_watcher(monkeypatch, tmp_path):
    fake_container = SimpleNamespace(
        config=SimpleNamespace(vault=SimpleNamespace(path=str(tmp_path), ignore=[".obsidian/"])),
        indexer=object(),
    )
    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: fake_container)

    recorded = {}

    class FakeWatchVault:
        def __init__(self, indexer, root):
            recorded["root"] = root

        def run(self, stop_event=None):
            recorded["ran"] = True

    monkeypatch.setattr(cli, "WatchVault", FakeWatchVault)

    result = CliRunner().invoke(cli.app, ["watch", "--config", "x.toml"])

    assert result.exit_code == 0
    assert recorded["ran"] is True
    assert recorded["root"] == str(tmp_path)


def test_reindex_cli_prints_skipped_warnings_and_count(monkeypatch):
    payload = {
        "notes": 3,
        "chunks": 9,
        "skipped": [
            {"path": "bad.md", "reason": "not UTF-8 text"},
            {"path": "locked.md", "reason": "permission denied"},
        ],
    }
    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: object())
    monkeypatch.setattr(cli, "reindex_payload", lambda container: payload)

    result = CliRunner().invoke(cli.app, ["reindex", "--config", "x.toml"])

    assert result.exit_code == 0
    assert result.stdout == "Indexed 3 notes, 9 chunks, 2 skipped.\n"
    assert "skipped bad.md: not UTF-8 text" in result.stderr
    assert "skipped locked.md: permission denied" in result.stderr


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
    assert recorded["port"] == 9009
    assert recorded["host"] == "127.0.0.1"


def test_serve_stdio_is_the_default(monkeypatch):
    fake_cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(cli, "load_config", lambda path: fake_cfg)
    monkeypatch.setattr(cli, "build_container", lambda cfg: object())

    recorded = {}

    class FakeServer:
        def run(self, transport="stdio"):
            recorded["transport"] = transport

    monkeypatch.setattr(
        cli, "build_server", lambda container, host="127.0.0.1", port=8000: FakeServer()
    )

    result = CliRunner().invoke(cli.app, ["serve", "--config", "x.toml"])

    assert result.exit_code == 0
    assert recorded["transport"] == "stdio"


def test_serve_rejects_unknown_transport(monkeypatch):
    fake_cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(cli, "load_config", lambda path: fake_cfg)
    monkeypatch.setattr(cli, "build_container", lambda cfg: object())
    monkeypatch.setattr(
        cli, "build_server", lambda container, host="127.0.0.1", port=8000: object()
    )

    result = CliRunner().invoke(cli.app, ["serve", "--transport", "grpc", "--config", "x.toml"])

    assert result.exit_code != 0


# --- known errors are one line on stderr and exit 1 (AC-14, AC-15)


@pytest.mark.parametrize("command", ["serve", "reindex", "watch", "status"])
def test_cli_config_error_prints_one_line_and_exits_1(monkeypatch, tmp_path, command):
    built = []
    monkeypatch.setattr(cli, "build_container", lambda cfg: built.append(cfg))
    missing = tmp_path / "nope.toml"

    result = CliRunner().invoke(cli.app, [command, "--config", str(missing)])

    assert result.exit_code == 1
    assert result.stderr == f"config file not found: {missing}\n"
    assert "Traceback" not in result.output
    assert built == []  # no model was loaded


def test_cli_removed_key_prints_one_line_and_exits_1(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/V"\n\n[embedding]\nprovider = "openai_compat"\n')

    result = CliRunner().invoke(cli.app, ["status", "--config", str(cfg_file)])

    assert result.exit_code == 1
    assert result.stderr.count("\n") == 1
    assert "embedding.provider was removed" in result.stderr


def _raising_indexer_container(error):
    class Indexer:
        def index(self):
            raise error

    return SimpleNamespace(
        config=SimpleNamespace(vault=SimpleNamespace(path="/gone")), indexer=Indexer()
    )


def test_cli_vault_unavailable_prints_one_line_and_exits_1(monkeypatch):
    from ariostea.ports.vault import VaultUnavailable

    error = VaultUnavailable("/gone", "no such folder")
    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: _raising_indexer_container(error))

    result = CliRunner().invoke(cli.app, ["reindex", "--config", "x.toml"])

    assert result.exit_code == 1
    assert result.stderr == "vault /gone is unavailable: no such folder\n"
    assert result.stdout == ""


def test_cli_watch_vault_unavailable_on_first_pass_exits_1(monkeypatch):
    from ariostea.ports.vault import VaultUnavailable

    error = VaultUnavailable("/gone", "no such folder")
    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: _raising_indexer_container(error))

    result = CliRunner().invoke(cli.app, ["watch", "--config", "x.toml"])

    assert result.exit_code == 1
    assert result.stderr == "vault /gone is unavailable: no such folder\n"


def test_cli_unknown_errors_keep_their_traceback(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda path: None)

    def boom(cfg):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cli, "build_container", boom)

    result = CliRunner().invoke(cli.app, ["status", "--config", "x.toml"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
