"""Render fetched Wikipedia articles as corpus notes, and keep the CC BY-SA
attribution table in `eval/wiki/NOTICE` in sync with them.

Attribution is mechanical on purpose: every note carries its revision
permalink in frontmatter, and every note has exactly one NOTICE row pointing
at the same revision, so the license trail cannot silently rot.

Frontmatter is written as fixed `key: value` lines, not real YAML, matching
what `ObsidianMarkdownParser._parse_frontmatter` actually reads: it splits
each line on the *first* `:` (`key.partition(":")`) and does not care about
YAML syntax at all. That means a colon inside `title` (a legitimate part of
a Wikipedia title, e.g. a subtitle) lands safely in the value half of the
split rather than breaking parsing -- there is nothing here to escape. The
one thing that *would* break both this format and the H1 assertion
downstream notes are checked against is a literal newline embedded in a
title; Wikipedia titles cannot contain one (it is not a legal page-title
character), so that case cannot arise from real fetched data and is not
guarded against here.
"""

from __future__ import annotations

from urllib.parse import quote

from ariostea.eval.wiki_manifest import ArticleSpec, note_path

# The line `eval/wiki/NOTICE` reserves for machine-maintained rows. Everything
# below it is rewritten by the build script; everything above is hand-written.
SENTINEL = "--- articles below this line are maintained by the build script ---"


def permalink(article: ArticleSpec) -> str:
    """Canonical permalink for the exact revision `article` is pinned to.

    `title` is percent-encoded via `quote` with its default safe set
    (`/` plus RFC 3986 unreserved characters): every other character --
    `&`, `=`, `#`, `+`, `?`, `(`, `)`, and non-ASCII letters alike -- is
    escaped, so a title like 'Go (game)' or 'Rock & Roll' produces a query
    string that parses back to the one intended `title` parameter rather
    than leaking extra `&`-separated fields or a raw fragment.

    Raises if `article.revid` is None. `ArticleSpec` allows an unpinned
    revid on purpose (Task 8 constructs one before it has fetched anything),
    but a permalink built from that would silently render `oldid=None` -- a
    URL that looks plausible and does not resolve to any real revision. That
    has to fail loudly here, the one place both `render_note` and
    `notice_row` route through, rather than slip into a note's frontmatter
    or a NOTICE row where nobody would notice until a reader clicked it.
    """
    if article.revid is None:
        raise ValueError(f"article {article.title!r} ({article.lang}) has no pinned revid")
    title = quote(article.title.replace(" ", "_"))
    return f"https://{article.lang}.wikipedia.org/w/index.php?title={title}&oldid={article.revid}"


def render_note(cluster: str, article: ArticleSpec, body: str) -> str:
    """Frontmatter (title, lang, cluster, revid, source, license) followed by
    `body`, blank-line separated.

    `body` is expected to already be `# {article.title}\\n\\n...` -- the
    shape `wikitext_to_markdown` produces -- since that H1 is what a reader
    (and Task 9's corpus check) uses to confirm the note's visible title
    matches its frontmatter. This function does not enforce that shape
    itself: it only lays out what it is given, on the same "small pure
    function, do one thing" grounds `wikitext_to_markdown` documents for its
    own stages. `body.lstrip()` trims only leading whitespace the caller's
    pipeline might have left, not trailing -- `wikitext_to_markdown` always
    ends its output with exactly one trailing newline, which this function
    preserves as-is.
    """
    return (
        "---\n"
        f"title: {article.title}\n"
        f"lang: {article.lang}\n"
        f"cluster: {cluster}\n"
        f"revid: {article.revid}\n"
        f"source: {permalink(article)}\n"
        "license: CC BY-SA 4.0\n"
        "---\n"
        "\n"
        f"{body.lstrip()}"
    )


def notice_row(cluster: str, article: ArticleSpec) -> str:
    """One `NOTICE` table row: local path, title, language, revision permalink."""
    return (
        f"{note_path(cluster, article)} | {article.title} | {article.lang} | {permalink(article)}"
    )


def write_notice_rows(notice_text: str, rows: list[str]) -> str:
    """Return `notice_text` with everything after `SENTINEL` replaced by `rows`.

    Rows are sorted before joining, so re-running the build script with the
    same corpus reproduces byte-identical output regardless of the order
    fetching happened to visit articles in (Task 10's whole-build
    reproducibility check depends on this). The sort is plain `sorted()` on
    each row string -- Python compares `str` by code point, not by locale
    collation, so this is stable across machines and Python versions; rows
    happen to start with an all-ASCII path, so in practice this sorts by
    path.

    Raises if `notice_text` has no `SENTINEL` line at all -- a missing
    sentinel means there is no well-defined place to insert rows, and
    silently appending at the end would let a hand-edited `NOTICE` (say, the
    sentinel line accidentally deleted while editing the header prose above
    it) merge machine rows into hand-written text with no seam. An empty
    `rows` list is accepted and produces a table section with no rows below
    the blank line after the sentinel; not exercised by the real build,
    which always has articles to record, but not rejected either.
    """
    head, sep, _ = notice_text.partition(SENTINEL)
    if not sep:
        raise ValueError(f"NOTICE is missing its sentinel line: {SENTINEL!r}")
    return head + SENTINEL + "\n\n" + "\n".join(sorted(rows)) + "\n"
