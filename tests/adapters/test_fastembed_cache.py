from pathlib import Path

from ariostea.adapters.fastembed_cache import fastembed_cache_dir


def test_cache_dir_defaults_under_home(monkeypatch):
    # fastembed's own default is the system temp directory, which macOS purges,
    # leaving dangling model links that break or hang the next load.
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    assert fastembed_cache_dir() == str(Path.home() / ".cache" / "fastembed")


def test_cache_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    assert fastembed_cache_dir() == str(tmp_path)
