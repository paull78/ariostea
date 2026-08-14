"""Convert Wikipedia wikitext into the Obsidian-flavored Markdown the eval
corpus is made of.

Pure functions, no network: the fetcher hands raw wikitext in, a note body
comes out. Wikitext (rather than rendered HTML) is the input because its links
are already `[[Article|label]]`, which is exactly the shape Obsidian wants —
rewriting is a lookup, not a URL-resolution problem.

The trade-off is that templates are *removed*, not expanded, so template-borne
values ({{convert}}, {{lang}}) vanish from the prose. For an evaluation corpus
that is acceptable: what matters is real headings, real length, and real
inter-article links. `eval/wiki/NOTICE` records the modification.

Pipeline order, and why
------------------------
`wikitext_to_markdown` runs these functions in this order:

    comments -> refs -> html containers -> templates -> tables -> media
    -> inline html tags -> external links -> lists -> headings -> wikilinks
    -> emphasis -> drop_sections -> normalize_blank_lines

- comments first: a half-edited article can have a stray `{{` or `[[` sitting
  inside an HTML comment. If the comment survives past this step, that brace
  poisons the balanced scanners downstream and leaks the rest of the article
  verbatim (see `_strip_balanced`).
- refs before templates: citations are full of `{{cite web|...}}` templates.
  Removing the whole `<ref>...</ref>` span first means the template scanner
  never has to look inside it.
- html containers before templates/tables: `<gallery>`, `<math>`,
  `<syntaxhighlight>` and friends can hold text that *looks* like `{{...}}`
  or `{|...|}` (LaTeX, source code) but isn't wikitext markup at all.
- media after templates/tables: a `[[File:...]]` caption can itself hold a
  `{{...}}` template (rare, but real); clearing templates and tables first
  means the media scanner only has to worry about nested `[[...]]` links.
- inline html tags last among the strip stages: by the time this runs, every
  structural block is already gone, so a generic catch-all tag pattern can't
  accidentally eat a tag that was still guarding content meant to survive.
- lists before headings: a wikitext numbered item starts with `#`, which
  Markdown reads as a heading marker. Converting lists first consumes that
  `#` into a `1. ` prefix before `convert_headings` — or `drop_sections`,
  downstream — ever sees the line as a bare hash. See `convert_lists`'s
  docstring for the sharper failure mode this avoids.
- lists before emphasis: `'''bold'''` becomes `**bold**`, and a line-leading
  `**` is indistinguishable from a two-deep wikitext bullet (`^(\\*+)`) to
  `convert_lists`. Every article's lede reads `'''Title''' is a ...`, so this
  isn't a corner case — it's the first line of every note. Nothing in either
  function enforces the order structurally; it's pinned by
  `test_lists_before_emphasis_keeps_the_lede_intact` and
  `test_emphasis_before_lists_corrupts_the_lede_into_a_bullet`.
- headings before drop_sections: `drop_sections` recognizes a boilerplate
  section by matching a Markdown `#`-heading line — it has no notion of
  wikitext's `==` heading syntax at all.
- drop_sections before normalize_blank_lines: dropping a section leaves a
  gap where the blank line that used to separate it from its neighbors is
  now doubled up. `normalize_blank_lines` is what collapses that back down,
  so it has to run after, last of all.
- external links after inline html tags, before lists: `strip_external_links`
  is a strip stage (it removes markup it can positively identify — a
  `[url ...]` bracket), so it belongs with the other strip stages and has no
  ordering dependency on the *convert* stages that follow. It runs before
  `convert_lists` only because it's grouped with the rest of the stripping
  work; nothing about list markup can appear inside an external-link bracket
  (`[`/`]` there mean "external link", never "bullet"), so this pairing has
  no sharp edge the way lists-before-headings does.
- wikilinks after headings, before emphasis: a wikilink can appear inside a
  heading line (`== The [[Violin]] Family ==`), and `convert_links` doesn't
  care whether the surrounding line is a heading or prose, so running it
  after `convert_headings` still rewrites the link correctly. It has to run
  before `convert_emphasis` for the same reason lists run before emphasis:
  none of `convert_links`'s own patterns collide with `'''`/`''`, but keeping
  every *convert* stage that could touch a shared span before the emphasis
  pass, which is the last content-shaping stage, keeps that ordering
  invariant simple to state and check.
- wikilinks after media: `convert_links` has no notion of a File-link's
  multi-parameter syntax (`[[File:...|thumb|300px|caption]]`), so an
  unrewritten one gets read as an ordinary link whose "label" is
  `thumb|300px|caption` — and since `File:...` never resolves against
  `targets`, that raw fragment leaks into the prose. See
  `convert_links`'s docstring and the reorder-pin test pair named there.

Invariant: this pipeline never removes text it did not positively identify as
markup. Every stage either matches a construct it can name (a ref, a
template, a table, a tag) or, when a scan can't find where that construct
ends, leaves the ambiguous remainder untouched rather than guessing where
prose resumes.
"""

from __future__ import annotations

import re

from ariostea.eval.normalize import normalize_ws

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# `\b` after the tag name is what keeps `<ref ...>` from matching the start of
# `<references ...>` — "f" and the following character are both word
# characters, so no word boundary exists there unless the tag name really
# ends at that point.
#
# `_REF_PAIR`'s body is tempered with `(?!<ref\b)` so it can't cross a second
# `<ref` opener: without that, an unclosed `<ref>` reads as an opening tag
# whose lazy `.*?` body then runs to the *next* ref's `</ref>`, deleting
# every real paragraph in between. Blocked from crossing, that match fails
# outright and the unclosed `<ref>` is left in place (leaked, not silently
# dropped) — same trade as `_strip_balanced`.
_REF_PAIR = re.compile(r"<ref\b[^>]*>(?:(?!<ref\b).)*?</ref\s*>", re.DOTALL | re.IGNORECASE)
_REFERENCES_SELF = re.compile(r"<references\b[^>]*/\s*>", re.IGNORECASE)
_REFERENCES_PAIR = re.compile(r"<references\b[^>]*>.*?</references\s*>", re.DOTALL | re.IGNORECASE)
# Self-closing refs must be substituted before `_REF_PAIR` runs: otherwise a
# self-closing `<ref name=x />` looks to `_REF_PAIR` like an opening tag with
# odd attributes (the trailing `/` is just more `[^>]*` soup), and its body
# then swallows everything up to the next real `</ref>` — including any
# self-closing ref and prose sitting in between.
_REF_SELF = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)

# Elements whose *content* is a payload (gallery listing, LaTeX, source code,
# map coordinates), not prose — removed whole, unlike the inline tags below.
_CONTAINER_TAGS = (
    "gallery|imagemap|timeline|score|syntaxhighlight|source|math|chem|mapframe|maplink"
)
# Same self-closing-before-paired hazard as `_REF_SELF`/`_REF_PAIR` above: if
# `_CONTAINER_PAIR` ran first, a self-closing `<gallery ... />` would read as
# an opening tag and its body would run to the next container's closer.
_CONTAINER_SELF = re.compile(rf"<(?:{_CONTAINER_TAGS})\b[^>]*/\s*>", re.IGNORECASE)
_CONTAINER_PAIR = re.compile(rf"<({_CONTAINER_TAGS})\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)

# Generic inline tag, content kept. A catch-all beats a hand-maintained
# allowlist: "no HTML tag survives" is a much stronger invariant than "no tag
# from this list of twelve survives". `[^>\n]*` (not `[^>]*`) keeps the match
# on one line: wikitext tags essentially never span a paragraph break, and
# bounding the pattern turns a malformed tag (missing `>`) or a bare `<`/`>`
# used as an inequality into a same-line no-match instead of a scan that
# silently swallows every real paragraph up to the next `>` anywhere later in
# the article.
_INLINE_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>\n]*>")

# Media embeds in the languages this corpus uses. `Media:` is deliberately
# excluded: `[[Media:song.ogg|listen]]` is an inline prose link ("listen to
# this clip"), not an embed — stripping it whole would eat the anchor text.
# Task 3's link rewriter flattens it to its label like any other link.
# `Imagen:` (es-wiki's legacy alias for `Image:`) *is* included below — it's
# a real embed prefix, not a `Media:`-style inline link.
_MEDIA_PREFIX = re.compile(r"\[\[\s*(?:File|Image|Immagine|Archivo|Imagen)\s*:", re.IGNORECASE)

# [[Target]] / [[Target|label]] / [[Target#Section|label]], plus the trailing
# "link trail" letters MediaWiki folds into the rendered label ([[violin]]s).
# Target and label both exclude `\n`, not just `[`/`]`: without that bound, an
# unclosed `[[` (missing `]]`) lets the lazy quantifier scan past a blank line
# hunting for some unrelated later `]]`, swallowing whole paragraphs in
# between as if they were the link's own target/label text. A genuine
# wikilink is always written on one line, so bounding to one line turns the
# unclosed case into a same-line no-match (leaked verbatim, per this module's
# invariant) instead of a scan that eats real prose — the same fix already
# applied to `_INLINE_TAG` above and to `_EXT_LINK` below.
_LINK = re.compile(r"\[\[([^\[\]|\n]+?)(?:\|([^\[\]\n]*))?\]\](\w*)")

# `[[:Category:X]]` / `[[:File:X]]`: a leading colon is wikitext's escape
# that forces an otherwise-special namespace link (category membership,
# media embed) to render as an ordinary, visible inline link instead. It has
# to be recognized and stripped before the invisible-category check below, or
# a genuine inline link like `[[:Category:Strings]]` would vanish from the
# prose along with the real invisible `[[Category:Strings]]` case.
_LEADING_COLON = re.compile(r"^:\s*")

# A category link with *no* leading colon renders as nothing in article
# text — it files the page under the category, it isn't a visible link at
# all. Flattening it to plain text like an ordinary out-of-corpus link would
# inject "Category: ..." straight into the middle of a sentence. Localized
# aliases cover the it/es editions this corpus draws from; `[[Category:...]]`
# is the only namespace this module treats as invisible — everything else
# unresolved (`Help:`, `Portal:`, an interwiki prefix like `fr:`) renders as
# an ordinary, if odd-looking, flattened link in real MediaWiki, and this
# module follows that.
_CATEGORY = re.compile(r"^(?:category|categoria|categoría)\s*:", re.IGNORECASE)

# `[url label]` / `[url]`. The scheme is matched shape-wise
# (`[a-z][a-z0-9+.-]*:`), not from a hand-maintained `http(s)` allowlist —
# the same catch-all-beats-an-allowlist principle `_INLINE_TAG` above
# states explicitly. MediaWiki's real external-link protocols include
# `ftp://`, `mailto:`, `news:`, `irc://`, `magnet:` and more, several of
# which (`mailto:`, `news:`, `magnet:`) have no `//` at all, hence the
# scheme's own `//` being optional; `|//` is the separate protocol-relative
# case (no scheme, just `//host/path`), which the original pattern already
# supported. The label half excludes `\n` for the identical reason `_LINK`'s
# label does: `[^\]]*` with no line bound would let an unterminated
# `[http://...` scan past a blank line hunting for some unrelated later `]`
# (e.g. a stray "]" that's just ordinary prose punctuation in the *next*
# paragraph) and read everything in between as the link's label.
_EXT_LINK = re.compile(
    r"\[(?:[a-z][a-z0-9+.-]*:(?://)?|//)\S+?(?:[ \t]+([^\]\n]*))?\]", re.IGNORECASE
)


def strip_comments(text: str) -> str:
    """Remove `<!-- ... -->` HTML comments. Must run before the balanced
    scanners below — see the module docstring's pipeline-order note."""
    return _COMMENT.sub("", text)


def strip_refs(text: str) -> str:
    """Remove citations and the references-list block, in all four wikitext
    shapes: self-closing `<ref .../>`, paired `<ref>...</ref>`, self-closing
    `<references .../>`, and the `<references>...</references>` container
    (removed whole, including any `<ref>` definitions nested inside it)."""
    text = _REF_SELF.sub("", text)
    text = _REFERENCES_SELF.sub("", text)
    text = _REFERENCES_PAIR.sub("", text)
    text = _REF_PAIR.sub("", text)
    return text


def _strip_balanced(text: str, open_tok: str, close_tok: str) -> str:
    """Remove every `open_tok`...`close_tok` region, tracking nesting depth.

    MediaWiki's own parser treats templates and tables as depth-based (they
    can contain other templates/tables), so a non-nesting regex would stop at
    the first inner closer and leave the rest of the construct behind.
    Depth-tracking is the minimal fix.

    If depth is still open at end-of-input — an unterminated `{{` or `{|`,
    which happens with truncated articles or a comment that ate a brace (see
    the module docstring) — the unclosed region is emitted verbatim instead of
    silently discarded. A dropped template truncates the article with no
    signal and reads downstream as a retrieval failure; a leaked `{{` is loud
    and easy to grep for.

    The leak starts at the *outermost* unmatched opener, so one stray brace
    disables template stripping for the rest of that article by design — a
    later corpus check will see dozens of surviving `{{` in that note, not
    one, which is the correct signal that the article (not the stripper) is
    where the problem is.
    """
    out: list[str] = []
    depth = 0
    open_start = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(open_tok, i):
            if depth == 0:
                open_start = i
            depth += 1
            i += len(open_tok)
        elif depth and text.startswith(close_tok, i):
            depth -= 1
            i += len(close_tok)
        else:
            if not depth:
                out.append(text[i])
            i += 1
    if depth:
        out.append(text[open_start:])
    return "".join(out)


def strip_templates(text: str) -> str:
    """Remove every `{{...}}` template invocation, including nested ones."""
    return _strip_balanced(text, "{{", "}}")


def strip_tables(text: str) -> str:
    """Remove every `{|...|}` wikitable, including nested ones."""
    return _strip_balanced(text, "{|", "|}")


def strip_media_links(text: str) -> str:
    """Remove `[[File:...]]`-style media embeds whole. Captions can contain
    their own `[[links]]`, so the closing `]]` has to be found by depth, not
    by regex. An embed with no closing `]]` (truncated article) is emitted
    verbatim from its opening `[[` to end-of-input rather than discarded —
    same rationale as `_strip_balanced`.

    Known limitation, accepted: the depth scan counts `[[`/`]]` pairs, not
    the individual `[`/`]` characters — a caption with an odd bracket count
    of its own (`[[File:X.jpg|thumb|Violin [sic]]]`, one stray `]`) can close
    the embed one bracket early or late, leaking or eating a character or two
    at the boundary. Real but low-frequency; bracket-balancing the scanner to
    handle it would cost more than the pollution it prevents.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        match = _MEDIA_PREFIX.match(text, i)
        if not match:
            out.append(text[i])
            i += 1
            continue
        start = i
        depth = 1
        j = match.end()
        while j < n and depth:
            if text.startswith("[[", j):
                depth += 1
                j += 2
            elif text.startswith("]]", j):
                depth -= 1
                j += 2
            else:
                j += 1
        if depth:
            out.append(text[start:])
            i = n
        else:
            i = j
    return "".join(out)


def strip_html_containers(text: str) -> str:
    """Remove gallery/math/code/map elements along with their content — their
    bodies are a payload (coordinate lists, LaTeX, source code), not prose.
    Best-effort: an unclosed container (missing closing tag) simply doesn't
    match, so the whole span — opening tag and content both — is left
    untouched. Unlike `_strip_balanced`, there's no depth-aware scan here to
    know where the content would have ended."""
    return _CONTAINER_PAIR.sub("", _CONTAINER_SELF.sub("", text))


def strip_html_tags(text: str) -> str:
    """Strip any remaining HTML tag, keeping its content. Intended to run
    last in the pipeline, once every structural element (refs, containers,
    templates, tables, media) is already gone."""
    return _INLINE_TAG.sub("", text)


# `\1` backreferences the exact run of `=` captured as the opener, so a line
# whose opener and closer run-lengths differ (a real, if rare, editing typo)
# still matches: `(={2,6})` backtracks to the *shorter* run, and the leftover
# `=` characters on the longer side fall into `(.+?)` as literal title text.
# That mirrors MediaWiki's own heading parser (level = min of the two runs),
# without this module having to special-case it. The lower bound of 2 (not 1)
# is deliberate, not just the mismatched-run case above: a level-1 `=Heading=`
# passes through untouched, since every note already gets one H1 from its
# title in `wikitext_to_markdown` — a second one would be a duplicate, not a
# missing conversion.
_HEADING = re.compile(r"^[ \t]*(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.MULTILINE)
_BULLET = re.compile(r"^(\*+)[ \t]*(.*)$", re.MULTILINE)
_NUMBERED = re.compile(r"^(#+)[ \t]*(.*)$", re.MULTILINE)
_DEF_LINE = re.compile(r"^[;:]+[ \t]*", re.MULTILINE)
# Emphasis is deliberately line-bounded (no DOTALL): a stray unmatched quote
# run would otherwise italicise half the article.
_BOLD = re.compile(r"'''(.+?)'''")
_ITALIC = re.compile(r"''(.+?)''")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# Strips every `*`/`_` from the title before the DROP_SECTIONS lookup: under
# Task 3's composition, headings convert before emphasis, so a wikitext
# heading written as `== '''References''' ==` arrives at `drop_sections` as
# `## **References**`, and the wrapping emphasis marks must not hide the
# title from the comparison.
_EMPHASIS_STRIP = str.maketrans("", "", "*_")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")

# Apparatus sections: no prose worth retrieving, and their link lists would
# pollute the wikilink graph. Lower-cased headings, en + it + es.
#
# Known false-positive risk, accepted: a generic subsection genuinely titled
# "Notes" (e.g. construction notes, not footnotes) is indistinguishable from
# a footnotes section by title alone and gets dropped too. Titles are the
# only signal `drop_sections` has; disambiguating would need heuristics
# (e.g. "only if near the end of the article") this module doesn't attempt.
DROP_SECTIONS = frozenset(
    {
        "references",
        "reference",
        "notes",
        "footnotes",
        "citations",
        "sources",
        "bibliography",
        "further reading",
        "external links",
        "see also",
        "gallery",
        "note",
        "bibliografia",
        "voci correlate",
        "altri progetti",
        "collegamenti esterni",
        "referencias",
        "notas",
        "véase también",
        "vease tambien",
        "enlaces externos",
        "bibliografía",
    }
)


def convert_headings(text: str) -> str:
    """Map `==Heading==` depth to `#` count. Must run after `convert_lists` —
    see that function's docstring for why."""
    return _HEADING.sub(lambda m: "#" * len(m.group(1)) + " " + m.group(2), text)


def convert_lists(text: str) -> str:
    """Bullets to `-`, numbered items to `1.`, two spaces per nesting level.

    Must run before `convert_headings`: a wikitext numbered item starts with
    `#`, which Markdown would otherwise read as a heading. Must also run
    before `convert_emphasis` — see that function's docstring.

    `_DEF_LINE` (definition-list markers `;`/`:`) runs *first*, before the
    bullet/numbered passes, not after. `:#`/`:*` are real wikitext — an
    indented numbered or bulleted item, common in bibliography and notes
    sections. If `_DEF_LINE` ran last, stripping the leading `:` would
    re-expose a bare `#`/`*` that `_NUMBERED`/`_BULLET` had already skipped
    over (they don't match a line starting with `:`), leaking an unconverted
    Markdown heading/bullet marker into the output instead of converting it —
    and a leaked `#` is exactly the collision this function exists to
    prevent `drop_sections` from misreading downstream.

    Known limitation, accepted: a literal `#REDIRECT [[Target]]` directive —
    valid wikitext only as an article's very first line — is read the same as
    any other numbered item and becomes `1. REDIRECT [[Target]]` rather than
    being recognized as a redirect. In practice this corpus never sees one:
    `wiki_fetch.fetch_article` requests `redirects=1`, so a redirect page's
    wikitext is never what gets fetched in the first place.
    """
    text = _DEF_LINE.sub("", text)
    text = _BULLET.sub(lambda m: "  " * (len(m.group(1)) - 1) + "- " + m.group(2), text)
    text = _NUMBERED.sub(lambda m: "  " * (len(m.group(1)) - 1) + "1. " + m.group(2), text)
    return text


def convert_emphasis(text: str) -> str:
    """Map `'''bold'''` / `''italic''` to `**bold**` / `*italic*`.

    Must run after `convert_lists`: a bold span at the start of a line
    becomes `**...`, and a line-leading `**` is indistinguishable from a
    two-deep wikitext bullet (`^(\\*+)`) to `convert_lists`. Every article's
    lede reads `'''Title''' is a ...`, so running emphasis first wouldn't be
    a corner case — it would corrupt the opening sentence of every note in
    the corpus. See `test_emphasis_before_lists_corrupts_the_lede_into_a_bullet`.

    Bold is substituted first so a `'''...'''` span never gets read as
    italic-then-stray-quote. The two compose for the `'''''both'''''`
    convention: bold strips the outer three quotes from each end first,
    leaving `''both''` for the italic pass to close — Markdown's own
    `***both***` for the combination falls out for free.

    Known limitation, accepted: MediaWiki's real apostrophe parser is more
    involved than "count runs of 2 or 3 quote characters" — it resolves
    ambiguous runs by looking at the whole line, not just the nearest pair.
    The construct this diverges on most, and the one likeliest to appear in
    this corpus's it/es prose, is an elision immediately before bold text:
    `l''''Italia''' è bella` (4 quotes — `l'` elision + `'''` bold). MediaWiki
    attaches the surplus apostrophe to the word before the bold
    (`l'**Italia**`); this function's two independent regex passes instead
    fold it into the bold span (`l**'Italia**`). No text is lost either way —
    only the rendered boundary of the emphasis differs — so retrieval quality
    is unaffected.
    """
    return _ITALIC.sub(r"*\1*", _BOLD.sub(r"**\1**", text))


def convert_links(text: str, targets: dict[str, str]) -> str:
    """Rewrite links to corpus articles as Obsidian wikilinks; flatten the rest.

    `targets` maps a normalized article title (`normalize_ws(title)`) to the
    slug of the note it became. A link whose target resolves against `targets`
    becomes `[[slug]]` (or `[[slug|label]]` when the rendered label differs
    from the slug), preserving the label so the prose still reads naturally.
    Everything else — an out-of-corpus article, a namespace page, a same-page
    section link — degrades to its rendered label as plain text, the same way
    a Markdown reader with no graph would see it.

    Must run after `strip_media_links` — an unrewritten `[[File:...|thumb|
    300px|caption]]` has no pipe count `_LINK` knows to respect, so it reads
    the whole thing as one link whose "label" is `thumb|300px|caption`; since
    `File:...` never resolves against `targets`, that raw fragment leaks
    straight into the prose. See `test_wikilinks_before_media_links_leaks_the_caption_into_prose`
    / `test_media_links_before_wikilinks_keeps_the_caption_out_of_prose`.
    Must also run after `convert_headings` (so a link inside a heading line
    is still rewritten) and before `convert_emphasis` — see the module
    docstring's pipeline-order note.

    Known limitation, accepted: a bare interlanguage link (`[[fr:Violon]]`)
    is invisible in real MediaWiki output; this function flattens it to
    `fr:Violon` instead. Modern wikitext essentially never contains one
    (Wikidata replaced this mechanism around 2013), so it's left unhandled.
    """

    def replace(match: re.Match[str]) -> str:
        # MediaWiki trims a wikilink target's surrounding whitespace before
        # parsing its namespace, so `[[ Category:X ]]` is valid, real
        # wikitext. `_CATEGORY`/`_LEADING_COLON` are both `^`-anchored;
        # stripping here first is what lets that anchor actually land on the
        # namespace instead of on leading whitespace the regex capture
        # doesn't trim on its own.
        target = match.group(1).strip()
        label, trail = match.group(2), match.group(3)

        if _LEADING_COLON.match(target):
            target = _LEADING_COLON.sub("", target)
        elif _CATEGORY.match(target):
            # Invisible in rendered article text — see `_CATEGORY` above.
            return ""

        # MediaWiki renders `_` as a space in a link's visible text (it's
        # just the URL-safe stand-in for a space in a page title) — apply
        # this before splitting off the section and before it's used for
        # display, not only at the lookup site below.
        target = target.replace("_", " ")
        page = target.split("#", 1)[0].strip()
        # A same-page section link ([[#Construction]]) has an *empty* page —
        # the whole target is the "#Section" part. Falling back to that empty
        # page for the display text would silently delete real reader-visible
        # link text, which this module's invariant forbids; fall back to the
        # untouched target instead, and never look such a page up in
        # `targets` (it can't resolve to a title match).
        fallback = page if page else target.strip()
        # `label` can be a *whitespace-only* string, not just empty or None:
        # a template inside a link's label (`{{lang|it|...}}`, `{{sic}}` —
        # common idioms) is reduced to bare whitespace by `strip_templates`,
        # six stages before this one runs. A bare `label or fallback` treats
        # "  " as truthy and never falls back, so `.strip()` then empties the
        # display outright — silently deleting reader-visible text. Stripping
        # `label` *before* the `or` is what makes the fallback actually fire.
        stripped_label = label.strip() if label else ""
        display = (stripped_label or fallback) + trail
        slug = targets.get(normalize_ws(page)) if page else None
        if slug is None:
            return display
        if not display:
            # Defense in depth: `fallback` is always non-empty whenever
            # `slug` resolves (that only happens when `page` is truthy, and
            # `fallback` is `page` in that case), so this shouldn't be
            # reachable — but a resolved link must never render invisibly,
            # so fall back to the slug itself rather than trust that.
            display = slug
        return f"[[{slug}]]" if display == slug else f"[[{slug}|{display}]]"

    return _LINK.sub(replace, text)


def strip_external_links(text: str) -> str:
    """Flatten `[url label]` / `[url]` to just the label (or nothing).

    An external link carries no corpus-graph meaning — its target isn't an
    article — so unlike `convert_links` there's nothing to rewrite it into.
    Grouped with the strip stages (see the module docstring): it removes
    markup it can positively identify, the same as `strip_refs` or
    `strip_media_links`.

    Known limitation, accepted: `[[https://x.org]]` isn't a real wikilink
    (MediaWiki wikilinks don't take URLs), but this pattern can match
    starting at the *second* `[` and consuming through the *first* `]`,
    leaving the outer `[`/`]` behind as stray literal characters (`[]`).
    Real but rare, and not worth the complexity of a fix.
    """
    return _EXT_LINK.sub(lambda m: m.group(1) or "", text)


def drop_sections(markdown: str, titles: frozenset[str] = DROP_SECTIONS) -> str:
    """Drop each named section together with everything nested under it, up to
    the next heading at the same or a shallower level.

    Must run after `convert_headings`: sections are recognized by matching a
    Markdown `#`-heading line, not wikitext's `==` syntax. The title is
    compared stripped, lower-cased, and with `*`/`_` emphasis markers
    removed, since Task 3's composition converts emphasis after headings —
    `== '''References''' ==` arrives here as `## **References**`.
    """
    kept: list[str] = []
    skip_level = 0
    for line in markdown.splitlines():
        heading = _MD_HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_level and level <= skip_level:
                skip_level = 0
            title = heading.group(2).strip().lower().translate(_EMPHASIS_STRIP)
            if not skip_level and title in titles:
                skip_level = level
                continue
        if not skip_level:
            kept.append(line)
    return "\n".join(kept)


def normalize_blank_lines(text: str) -> str:
    """Trim trailing whitespace per line and collapse runs of 2+ blank lines
    (3+ consecutive newlines) down to a single blank line.

    Also strips leading and trailing whitespace from the whole document, so
    the result has no leading blank lines and — notably — no trailing
    newline. A caller that needs one (e.g. writing a note file) has to add it
    back itself; this function does not assume one.
    """
    return _BLANK_RUN.sub("\n\n", _TRAILING_WS.sub("", text)).strip()


def wikitext_to_markdown(raw: str, title: str, targets: dict[str, str]) -> str:
    """Full pipeline: raw wikitext in, note body (H1 + Markdown) out.

    Stage order is pinned by the module docstring's "Pipeline order, and why"
    section; see it before reordering anything here. Defined last, after
    every stage it composes, so the file itself reads in pipeline order.
    """
    text = strip_comments(raw)
    text = strip_refs(text)
    text = strip_html_containers(text)
    text = strip_templates(text)
    text = strip_tables(text)
    text = strip_media_links(text)
    text = strip_html_tags(text)
    text = strip_external_links(text)
    text = convert_lists(text)
    text = convert_headings(text)
    text = convert_links(text, targets)
    text = convert_emphasis(text)
    text = drop_sections(text)
    return f"# {title}\n\n{normalize_blank_lines(text)}".rstrip() + "\n"
