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
