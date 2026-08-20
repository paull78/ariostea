"""Invariants the committed Wikipedia snapshot must satisfy.

Reads the files in `eval/wiki/` — no network, so this runs in the normal
suite. These assertions are what stands between a converter regression and a
corpus that looks fine until Plan 3 anchors a gold span in text that isn't
there.
"""

import re
from pathlib import Path

import pytest

from ariostea.eval.wiki_manifest import load_manifest, note_path

WIKI = Path(__file__).resolve().parents[2] / "eval" / "wiki"
CLUSTERS = load_manifest(WIKI / "clusters.json")
ARTICLES = [(cluster.name, article) for cluster in CLUSTERS for article in cluster.articles]
IDS = [f"{cluster}/{article.slug}" for cluster, article in ARTICLES]
NOTICE = (WIKI / "NOTICE").read_text(encoding="utf-8")

_WIKILINK = re.compile(r"\[\[([^\]|#]+)")
_REVID = re.compile(r"^revid: (\d+)$", re.MULTILINE)
# Anything here means the converter leaked raw markup into the corpus.
LEFTOVERS = ("{{", "{|", "<ref", "[[File:", "<!--", "&nbsp;")
# Long enough to chunk into several passages, which is the entire reason this
# corpus replaced the old few-line fixtures.
MIN_NOTE_CHARS = 1200


def _text(cluster: str, article) -> str:
    return (WIKI / note_path(cluster, article)).read_text(encoding="utf-8")


def test_the_snapshot_matches_the_design_targets():
    assert 5 <= len(CLUSTERS) <= 6
    assert 60 <= len(ARTICLES) <= 80
    assert {article.lang for _, article in ARTICLES} >= {"en", "it", "es"}


@pytest.mark.parametrize("cluster,article", ARTICLES, ids=IDS)
def test_every_article_is_pinned_attributed_and_substantial(cluster, article):
    assert article.revid is not None, "run eval/build_wiki_corpus.py to pin this article"
    text = _text(cluster, article)
    assert text.startswith("---\n")
    assert f"revid: {article.revid}\n" in text
    assert "license: CC BY-SA 4.0\n" in text
    assert f"\n# {article.title}\n" in text
    assert len(text) > MIN_NOTE_CHARS


@pytest.mark.parametrize("cluster,article", ARTICLES, ids=IDS)
def test_no_unconverted_wikitext_survives(cluster, article):
    text = _text(cluster, article)
    for leftover in LEFTOVERS:
        assert leftover not in text, f"{leftover!r} leaked into {note_path(cluster, article)}"


def test_every_wikilink_resolves_to_a_corpus_note():
    """A dangling wikilink means the link rewriter emitted a target that
    isn't a note — which would make the corpus's graph structure a lie."""
    slugs = {article.slug for _, article in ARTICLES}
    for cluster, article in ARTICLES:
        for target in _WIKILINK.findall(_text(cluster, article)):
            assert target.strip() in slugs, (
                f"dangling [[{target}]] in {note_path(cluster, article)}"
            )


@pytest.mark.parametrize("cluster,article", ARTICLES, ids=IDS)
def test_notice_attributes_every_manifest_article(cluster, article):
    assert note_path(cluster, article) in NOTICE
    assert f"oldid={article.revid}" in NOTICE


def test_no_note_on_disk_is_missing_its_attribution_row():
    """The reverse of the check above, and the one that catches a renamed
    article: the manifest entry moves to a new slug, the old note stays
    committed, and every manifest-driven assertion still passes while a
    CC BY-SA file sits in the repository with nothing attributing it."""
    for path in sorted(WIKI.glob("*/*.md")):
        relative = path.relative_to(WIKI).as_posix()
        assert relative in NOTICE, f"{relative} is committed but has no NOTICE row"


@pytest.mark.parametrize("cluster,article", ARTICLES, ids=IDS)
def test_each_notice_row_cites_the_revision_its_note_was_built_from(cluster, article):
    """A partial `--cluster` build regenerates rows from the manifest while
    leaving other notes untouched on disk, so the row and the note can drift
    apart. Attribution that points at a different revision than the text came
    from is worse than none: it looks authoritative and is wrong."""
    row = next(line for line in NOTICE.splitlines() if line.startswith(note_path(cluster, article)))
    revid_in_note = _REVID.search(_text(cluster, article))
    assert revid_in_note is not None
    assert f"oldid={revid_in_note.group(1)}" in row
