from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ariostea import cli


class FakeWatcher:
    """Plays a fixed list of reports and warnings through the CLI's callbacks."""

    def __init__(self, events):
        self.events = events

    def run(self, stop_event=None, on_report=None, on_warning=None, progress=None):
        for kind, value in self.events:
            (on_report if kind == "report" else on_warning)(value)


def _watch_container(tmp_path, watcher):
    return SimpleNamespace(
        config=SimpleNamespace(vault=SimpleNamespace(path=str(tmp_path))), watcher=watcher
    )


def _report(embedded=(), deleted=(), skipped=()):
    from ariostea.domain.models import IndexReport

    return IndexReport(5, 20, tuple(embedded), tuple(deleted), tuple(skipped))


def test_cli_watch_prints_pass_summary(monkeypatch, tmp_path):  # AC-17
    watcher = FakeWatcher(
        [
            ("report", _report(embedded=["a.md", "b.md"])),  # the first pass
            ("report", _report()),  # a pass that changed nothing
            ("report", _report(embedded=["a.md"], deleted=["c.md"])),
            ("warning", "vault /v is unavailable: no such folder"),
        ]
    )
    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: _watch_container(tmp_path, watcher))

    result = CliRunner().invoke(cli.app, ["watch", "--config", "x.toml"])

    assert result.exit_code == 0
    assert result.stdout.splitlines() == [
        f"Indexing and watching {tmp_path} (Ctrl-C to stop)...",
        "+2 updated, -0 deleted",
        "+1 updated, -1 deleted",
    ]
    assert result.stderr == "warning: vault /v is unavailable: no such folder\n"


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
    monkeypatch.setattr(cli, "reindex_payload", lambda container, progress=None: payload)

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
        def index(self, progress=None):
            raise error

    class Watcher:
        def run(self, stop_event=None, on_report=None, on_warning=None, progress=None):
            raise error

    return SimpleNamespace(
        config=SimpleNamespace(vault=SimpleNamespace(path="/gone")),
        indexer=Indexer(),
        watcher=Watcher(),
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


def test_reindex_cli_progress_on_stderr_only(monkeypatch):  # AC-16
    from ariostea.domain.models import IndexReport, Progress

    class Indexer:
        def index(self, progress=None):
            for done in range(1, 251):
                progress(Progress(done=done, total=250, embedded=done, skipped=0))
            return IndexReport(250, 900, (), (), ())

    container = SimpleNamespace(indexer=Indexer())
    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: container)

    result = CliRunner().invoke(cli.app, ["reindex", "--config", "x.toml"])

    assert result.exit_code == 0
    assert result.stdout == "Indexed 250 notes, 900 chunks, 0 skipped.\n"
    assert result.stderr.splitlines() == [
        "100/250 notes · 100 embedded · 0 skipped",
        "200/250 notes · 200 embedded · 0 skipped",
        "250/250 notes · 250 embedded · 0 skipped",
    ]


def test_watch_cli_passes_progress_for_the_first_pass(monkeypatch, tmp_path):
    from ariostea.domain.models import Progress

    class Watcher:
        def run(self, stop_event=None, on_report=None, on_warning=None, progress=None):
            progress(Progress(done=1, total=1, embedded=1, skipped=0))

    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: _watch_container(tmp_path, Watcher()))

    result = CliRunner().invoke(cli.app, ["watch", "--config", "x.toml"])

    assert result.exit_code == 0
    assert result.stderr == "1/1 notes · 1 embedded · 0 skipped\n"
