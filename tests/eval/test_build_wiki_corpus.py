"""Tests for the corpus build script's failure handling.

The abort policy is the only stateful logic in the build: a counter mutated on
one path, read on another, gating a flag checked in two loop headers. It also
only ever runs against a live API, so a bug in it surfaces as a half-built
corpus rather than a test failure — which is why it is worth the import
gymnastics below. `eval/` is a script directory, not a package on the test
path (`pyproject.toml` sets `pythonpath = ["src"]`), so the module is loaded
by file path.
"""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from ariostea.eval.wiki_corpus import SENTINEL
from ariostea.eval.wiki_fetch import FetchedArticle, WikiFetchError
from ariostea.eval.wiki_manifest import ArticleSpec, Cluster, load_manifest

_PATH = Path(__file__).resolve().parents[2] / "eval" / "build_wiki_corpus.py"
_SPEC = importlib.util.spec_from_file_location("build_wiki_corpus", _PATH)
assert _SPEC and _SPEC.loader
build_script = importlib.util.module_from_spec(_SPEC)
# Registered before exec: @dataclass resolves annotations via
# sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules[_SPEC.name] = build_script
_SPEC.loader.exec_module(build_script)


def _clusters(count: int = 5) -> tuple[Cluster, ...]:
    return (
        Cluster(
            name="coffee",
            articles=tuple(ArticleSpec(title=f"A{i}", lang="en") for i in range(count)),
        ),
    )


def _retryable(*_args, **_kwargs):
    raise WikiFetchError("rate limited", retryable=True)


def _permanent(*_args, **_kwargs):
    raise WikiFetchError("no such page", retryable=False)


def test_backoff_grows_so_a_throttle_has_time_to_clear():
    """A fixed 5s delay gives a tripped Wikimedia throttle 15 seconds total to
    clear across the whole run, which observation says is not enough."""
    assert build_script.backoff_delay(0) == build_script.DELAY_S
    assert build_script.backoff_delay(1) == 5.0
    assert build_script.backoff_delay(2) == 15.0
    assert build_script.backoff_delay(3) == 45.0


def test_run_aborts_after_the_configured_run_of_retryable_failures(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(build_script, "fetch_article", lambda *a, **k: _retryable())
    monkeypatch.setattr(build_script, "fetch_and_write", lambda *a, **k: _retryable())

    result = build_script.build(
        client=None, clusters=_clusters(5), wiki_dir=tmp_path, sleep=calls.append
    )

    assert result.aborted
    # Three attempts, not five: the remaining articles are never requested.
    assert len(result.failures) == build_script.MAX_CONSECUTIVE_RETRYABLE
    assert calls == [5.0, 15.0, 45.0]


def test_a_permanent_failure_resets_the_retryable_run(monkeypatch, tmp_path):
    """A missing article is not evidence that Wikimedia is throttling us, so
    it must not carry the run closer to an abort."""
    outcomes = [_retryable, _retryable, _permanent, _retryable, _retryable]

    def fake(*_args, **_kwargs):
        return outcomes.pop(0)()

    monkeypatch.setattr(build_script, "fetch_and_write", fake)
    result = build_script.build(
        client=None, clusters=_clusters(5), wiki_dir=tmp_path, sleep=lambda _s: None
    )

    assert not result.aborted
    assert len(result.failures) == 5


def test_a_failed_article_keeps_its_pre_run_manifest_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(build_script, "fetch_and_write", lambda *a, **k: _permanent())
    result = build_script.build(
        client=None, clusters=_clusters(2), wiki_dir=tmp_path, sleep=lambda _s: None
    )
    assert [a.revid for a in result.clusters[0].articles] == [None, None]


def test_reconcile_describes_exactly_the_notes_on_disk(tmp_path):
    """The manifest and NOTICE are written in a `finally`, so this runs even
    after an abort or a crash — a note on disk with no attribution row is the
    one outcome that silently breaks the license trail."""
    wiki = tmp_path / "wiki"
    (wiki / "coffee").mkdir(parents=True)
    (wiki / "coffee" / "a0.md").write_text("note", encoding="utf-8")
    notice = wiki / "NOTICE"
    notice.write_text(f"header\n\n{SENTINEL}\n", encoding="utf-8")
    manifest = wiki / "clusters.json"

    clusters = (
        Cluster(
            name="coffee",
            articles=(
                ArticleSpec(title="A0", lang="en", revid=7),
                # Pinned, but its fetch never wrote a file.
                ArticleSpec(title="A1", lang="en", revid=8),
            ),
        ),
    )
    written = build_script.reconcile(clusters, wiki, manifest, notice)

    assert written == 1
    body = notice.read_text(encoding="utf-8")
    assert "coffee/a0.md" in body and "oldid=7" in body
    assert "coffee/a1.md" not in body
    assert load_manifest(manifest) == clusters


def test_an_unknown_cluster_name_is_an_error_not_a_silent_no_op(monkeypatch, tmp_path, capsys):
    """Without this the typo builds nothing, rewrites both files, exits 0."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    manifest = wiki / "clusters.json"
    manifest.write_text(
        json.dumps({"clusters": [{"name": "coffee", "articles": [{"title": "A", "lang": "en"}]}]}),
        encoding="utf-8",
    )
    notice = wiki / "NOTICE"
    notice.write_text(f"header\n\n{SENTINEL}\n", encoding="utf-8")
    before = notice.read_text(encoding="utf-8")

    monkeypatch.setattr(build_script, "WIKI_DIR", wiki)
    monkeypatch.setattr(build_script, "MANIFEST", manifest)
    monkeypatch.setattr(build_script, "NOTICE", notice)
    monkeypatch.setattr("sys.argv", ["build_wiki_corpus.py", "--cluster", "coffe"])

    assert build_script.main() == 2
    assert "unknown cluster" in capsys.readouterr().err
    assert notice.read_text(encoding="utf-8") == before


def test_fetch_and_write_renders_a_note_and_returns_the_pinned_spec(monkeypatch, tmp_path):
    monkeypatch.setattr(
        build_script,
        "fetch_article",
        lambda *a, **k: FetchedArticle(title="Violin", revid=99, wikitext="The '''violin'''."),
    )
    pinned = build_script.fetch_and_write(
        client=None,
        cluster="strings",
        article=ArticleSpec(title="Violin", lang="en"),
        targets={"en": {}},
        wiki_dir=tmp_path,
        dropped=Counter(),
    )
    assert pinned == ArticleSpec(title="Violin", lang="en", revid=99)
    note = (tmp_path / "strings" / "violin.md").read_text(encoding="utf-8")
    assert "revid: 99" in note and "# Violin" in note and "**violin**" in note


def test_the_report_separates_lost_text_from_expected_chrome(capsys):
    build_script.print_report(
        Counter({"cite web": 12, "BLOCKED:langx": 1, "UNKNOWN-MUSIC-ARG:x": 2})
    )
    out = capsys.readouterr().out
    assert "removed as chrome" in out and "cite web" in out
    assert "TEXT LOST TO UNHANDLED MARKUP" in out
    assert out.index("cite web") < out.index("BLOCKED:langx")


@pytest.mark.parametrize("name", ["BLOCKED:x", "UNKNOWN-CONVERT-JOINER:x", "EMPTY-EXPANSION:x"])
def test_every_lost_text_marker_is_recognized_by_the_report(name):
    assert name.startswith(build_script.REPORT_PREFIXES)
