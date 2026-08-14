from pathlib import Path

import pytest

from ariostea.eval.wiki_corpus import (
    SENTINEL,
    notice_row,
    permalink,
    render_note,
    write_notice_rows,
)
from ariostea.eval.wiki_manifest import ArticleSpec

VIOLIN = ArticleSpec(title="Violin", lang="en", revid=1234)
VIOLIN_ES = ArticleSpec(title="Violín", lang="es", revid=99)
GO_GAME = ArticleSpec(title="Go (game)", lang="en", revid=42)
ROCK_AND_ROLL = ArticleSpec(title="Rock & Roll", lang="en", revid=7)
UNPINNED = ArticleSpec(title="Violin", lang="en")


def test_permalink_points_at_the_pinned_revision():
    assert permalink(VIOLIN) == "https://en.wikipedia.org/w/index.php?title=Violin&oldid=1234"


def test_permalink_percent_encodes_non_ascii_titles():
    assert permalink(VIOLIN_ES) == "https://es.wikipedia.org/w/index.php?title=Viol%C3%ADn&oldid=99"


def test_permalink_percent_encodes_query_metacharacters_in_titles():
    # '&' and '(' ')' are not URL-safe inside a query value; if they leaked
    # through unescaped the query string would parse as extra parameters
    # (or, for a title containing '=', as a bogus key=value pair) instead of
    # resolving to the intended article.
    assert permalink(GO_GAME) == "https://en.wikipedia.org/w/index.php?title=Go_%28game%29&oldid=42"
    assert (
        permalink(ROCK_AND_ROLL)
        == "https://en.wikipedia.org/w/index.php?title=Rock_%26_Roll&oldid=7"
    )


def test_permalink_rejects_an_unpinned_article():
    # ArticleSpec allows revid=None (Task 8 constructs one before it has
    # fetched anything); a permalink built from that would silently render
    # "oldid=None" -- a URL that looks plausible but resolves to nothing.
    with pytest.raises(ValueError, match="no pinned revid"):
        permalink(UNPINNED)


def test_render_note_writes_parseable_frontmatter_then_the_body():
    note = render_note("string-instruments", VIOLIN, "# Violin\n\nA bowed instrument.\n")
    assert note == (
        "---\n"
        "title: Violin\n"
        "lang: en\n"
        "cluster: string-instruments\n"
        "revid: 1234\n"
        "source: https://en.wikipedia.org/w/index.php?title=Violin&oldid=1234\n"
        "license: CC BY-SA 4.0\n"
        "---\n"
        "\n"
        "# Violin\n"
        "\n"
        "A bowed instrument.\n"
    )


def test_render_note_rejects_an_unpinned_article():
    with pytest.raises(ValueError, match="no pinned revid"):
        render_note("string-instruments", UNPINNED, "# Violin\n\nA bowed instrument.\n")


def test_notice_row_records_path_title_language_and_permalink():
    assert notice_row("string-instruments", VIOLIN) == (
        "string-instruments/violin.md | Violin | en | "
        "https://en.wikipedia.org/w/index.php?title=Violin&oldid=1234"
    )


def test_notice_row_rejects_an_unpinned_article():
    with pytest.raises(ValueError, match="no pinned revid"):
        notice_row("string-instruments", UNPINNED)


def test_write_notice_rows_replaces_everything_after_the_sentinel_and_is_idempotent():
    notice = f"Header text\n\n{SENTINEL}\nstale row\n"
    once = write_notice_rows(notice, ["b | B | en | u2", "a | A | en | u1"])
    assert once == f"Header text\n\n{SENTINEL}\n\na | A | en | u1\nb | B | en | u2\n"
    assert write_notice_rows(once, ["b | B | en | u2", "a | A | en | u1"]) == once


def test_write_notice_rows_raises_without_the_sentinel():
    with pytest.raises(ValueError, match="sentinel"):
        write_notice_rows("Header text, no sentinel here\n", ["a | A | en | u1"])


def test_write_notice_rows_matches_the_committed_notice_file():
    # Pins the real file's sentinel line against this module's SENTINEL
    # constant: if the committed NOTICE ever drifts (a re-wrapped line, a
    # typo fix), write_notice_rows would raise on the very first real build
    # instead of silently no-op'ing on a sentinel it can no longer find.
    notice_path = Path(__file__).resolve().parents[2] / "eval" / "wiki" / "NOTICE"
    assert SENTINEL in notice_path.read_text(encoding="utf-8")
