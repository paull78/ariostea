"""Render fetched Wikipedia articles as corpus notes, and keep the CC BY-SA
attribution table in `eval/wiki/NOTICE` in sync with them.

What this module actually guarantees: a note's frontmatter and its NOTICE
row are both derived from the same `ArticleSpec` (`render_note` and
`notice_row` both call `permalink`, which reads `article.revid` directly),
so for any *one* article the two cannot disagree. That is a per-article
guarantee, not a corpus-wide one -- nothing here has a view of the
filesystem, so it cannot detect an orphaned note left behind after a title
is renamed in the manifest, or a NOTICE row surviving from a build that
covered a different set of articles (a partial `--cluster` run, say). The
corpus-wide invariant -- every committed note has exactly one matching
NOTICE row and vice versa -- is enforced by Task 9's snapshot test, which
has a filesystem view this module deliberately does not.

Frontmatter is written as fixed `key: value` lines, not real YAML, matching
what `_parse_frontmatter` (a module-level function in
`ariostea.adapters.parse.obsidian`, not a method) actually reads: it splits
each line on the *first* `:` (`line.partition(":")`) and does not care about
YAML syntax at all. That means a colon inside `title` (a legitimate part of
a Wikipedia title, e.g. a subtitle) lands safely in the value half of the
split rather than breaking parsing -- there is nothing here to escape. The
one thing that *would* break both this format and the H1 assertion
downstream notes are checked against is a literal newline embedded in a
title; Wikipedia titles cannot contain one (it is not a legal page-title
character), so that case cannot arise from real fetched data and is not
guarded against here.

Known parser interaction, not a bug in this module: `ObsidianMarkdownParser`
also derives `tags` from any `#word` it finds in the body via a simple regex,
with no notion of prose versus markup. An article body containing something
like "reached #1 on the charts" (plausible in this corpus's music articles)
produces a spurious tag `1`. Harmless today -- `Note.tags` is written but
never read anywhere in `src/` -- documented here so the next reader does not
have to rediscover it.
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
        raise ValueError(f"article {article.title!r} ({article.lang}): no pinned revid")
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

    Known limitation, accepted: an empty (or all-whitespace) `body` renders
    fine -- parseable frontmatter followed by nothing -- but then disagrees
    with itself once read back. `ObsidianMarkdownParser` has no H1 to find,
    so it falls back to the note's path stem for `Note.title`, while
    `frontmatter['title']` still holds the real article title. Unreachable
    from Task 8's real caller (`wikitext_to_markdown` always emits an H1) and
    would be caught by Task 9's `\\n# {title}\\n` and length assertions if it
    ever happened, so not guarded against here.
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

    Raises unless `notice_text` contains `SENTINEL` exactly once. Zero
    occurrences means there is no well-defined place to insert rows -- a
    hand-edited `NOTICE` could have had the sentinel line accidentally
    deleted while editing the header prose above it, and silently appending
    at the end would merge machine rows into hand-written text with no seam.
    More than one occurrence is the sharper hazard: `NOTICE`'s own header
    illustrates the row format with an example, which is exactly the kind of
    edit that invites quoting `SENTINEL` a second time in prose. `partition`
    matches the first occurrence found anywhere in the text, not a line by
    itself, so a second occurrence would silently discard every real
    sentinel-onward hand-written line as if it were stale build output --
    this module owns the legal instrument in that text, so that has to raise
    rather than pick one occurrence and guess.

    `rows` may be empty, in which case the sentinel line is left with
    nothing below it and no extra trailing blank line -- not exercised by
    the real build, which always has articles to record, but kept clean
    rather than left to accumulate a stray blank line.
    """
    if notice_text.count(SENTINEL) != 1:
        raise ValueError(
            f"NOTICE must contain the sentinel line {SENTINEL!r} exactly once, "
            f"found {notice_text.count(SENTINEL)}"
        )
    head, _, _ = notice_text.partition(SENTINEL)
    if not rows:
        return head + SENTINEL + "\n"
    return head + SENTINEL + "\n\n" + "\n".join(sorted(rows)) + "\n"
