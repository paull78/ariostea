from pathlib import Path

import pytest

from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.eval.wiki_corpus import (
    SENTINEL,
    notice_row,
    permalink,
    render_note,
    write_notice_rows,
)
from ariostea.eval.wiki_manifest import ArticleSpec, note_path

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


def test_render_note_output_actually_parses_as_obsidian_frontmatter():
    # The whole format -- field order, the blank line before the body -- is
    # derived from how ObsidianMarkdownParser reads it, but nothing exercised
    # that parser directly until now: a change to `_FRONTMATTER` or `_H1` in
    # obsidian.py could silently break this module and nothing here would
    # notice.
    raw = render_note("string-instruments", VIOLIN, "# Violin\n\nA bowed instrument.\n")
    note, body = ObsidianMarkdownParser().parse(note_path("string-instruments", VIOLIN), raw, 0.0)
    assert note.title == "Violin"
    assert note.frontmatter["revid"] == "1234"
    assert note.frontmatter["source"] == permalink(VIOLIN)
    # The parser's frontmatter regex consumes through the closing `---\n` but
    # not the blank line after it -- that separator is part of `body` from
    # the parser's point of view, not of the frontmatter block.
    assert body == "\n# Violin\n\nA bowed instrument.\n"


def test_render_note_strips_leading_whitespace_from_the_body():
    # Pins the reason `.lstrip()` exists: without it, leading blank lines a
    # caller's pipeline left in `body` would open a gap between the
    # frontmatter's closing `---` and the note's H1.
    note = render_note("string-instruments", VIOLIN, "\n\n  # Violin\n\nA bowed instrument.\n")
    assert note.endswith("---\n\n# Violin\n\nA bowed instrument.\n")


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


def test_write_notice_rows_raises_on_a_duplicated_sentinel():
    # A header that quotes the sentinel while explaining the format (exactly
    # what NOTICE's own illustrative example block invites) makes
    # `partition` cut at the *first* occurrence, silently discarding
    # everything from the real sentinel onward -- including hand-written
    # license prose. Two occurrences must raise, not pick one.
    notice = f"HEAD\nmentions {SENTINEL} in prose\nMORE HEADER PROSE\n\n{SENTINEL}\nold row\n"
    with pytest.raises(ValueError, match="sentinel"):
        write_notice_rows(notice, ["a | A | en | u"])


def test_write_notice_rows_with_no_rows_emits_no_stray_blank_line():
    notice = f"Header text\n\n{SENTINEL}\nstale row\n"
    assert write_notice_rows(notice, []) == f"Header text\n\n{SENTINEL}\n"


def test_write_notice_rows_matches_the_committed_notice_file():
    # Pins the real file's sentinel line against this module's SENTINEL
    # constant: if the committed NOTICE ever drifts (a re-wrapped line, a
    # typo fix), write_notice_rows would raise on the very first real build
    # instead of silently no-op'ing on a sentinel it can no longer find.
    notice_path = Path(__file__).resolve().parents[2] / "eval" / "wiki" / "NOTICE"
    assert SENTINEL in notice_path.read_text(encoding="utf-8")
