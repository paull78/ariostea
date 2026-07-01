from types import SimpleNamespace

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
        def __init__(self, indexer, root, ignore=()):
            recorded["root"] = root
            recorded["ignore"] = list(ignore)

        def run(self, stop_event=None):
            recorded["ran"] = True

    monkeypatch.setattr(cli, "WatchVault", FakeWatchVault)

    result = CliRunner().invoke(cli.app, ["watch", "--config", "x.toml"])

    assert result.exit_code == 0
    assert recorded["ran"] is True
    assert recorded["root"] == str(tmp_path)
    assert recorded["ignore"] == [".obsidian/"]


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

    monkeypatch.setattr(cli, "build_server", lambda container, host="127.0.0.1", port=8000: FakeServer())

    result = CliRunner().invoke(cli.app, ["serve", "--config", "x.toml"])

    assert result.exit_code == 0
    assert recorded["transport"] == "stdio"


def test_serve_rejects_unknown_transport(monkeypatch):
    fake_cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(cli, "load_config", lambda path: fake_cfg)
    monkeypatch.setattr(cli, "build_container", lambda cfg: object())
    monkeypatch.setattr(cli, "build_server", lambda container, host="127.0.0.1", port=8000: object())

    result = CliRunner().invoke(cli.app, ["serve", "--transport", "grpc", "--config", "x.toml"])

    assert result.exit_code != 0
