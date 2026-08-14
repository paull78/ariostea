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
Callers (Task 3's `wikitext_to_markdown`) must run these functions in this
order:

    comments -> refs -> html containers -> templates -> tables -> media
    -> inline html tags

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
- inline html tags last: by the time this runs, every structural block is
  already gone, so a generic catch-all tag pattern can't accidentally eat a
  tag that was still guarding content meant to survive.
"""

from __future__ import annotations

import re

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# `\b` after the tag name is what keeps `<ref ...>` from matching the start of
# `<references ...>` — "f" and the following character are both word
# characters, so no word boundary exists there unless the tag name really
# ends at that point.
_REF_SELF = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
_REF_PAIR = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.DOTALL | re.IGNORECASE)
_REFERENCES_SELF = re.compile(r"<references\b[^>]*/\s*>", re.IGNORECASE)
_REFERENCES_PAIR = re.compile(r"<references\b[^>]*>.*?</references\s*>", re.DOTALL | re.IGNORECASE)

# Elements whose *content* is a payload (gallery listing, LaTeX, source code,
# map coordinates), not prose — removed whole, unlike the inline tags below.
_CONTAINER_TAGS = (
    "gallery|imagemap|timeline|score|syntaxhighlight|source|math|chem|mapframe|maplink"
)
_CONTAINER_SELF = re.compile(rf"<(?:{_CONTAINER_TAGS})\b[^>]*/\s*>", re.IGNORECASE)
_CONTAINER_PAIR = re.compile(rf"<({_CONTAINER_TAGS})\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)

# Generic inline tag, content kept. A catch-all beats a hand-maintained
# allowlist: "no HTML tag survives" is a much stronger invariant than "no tag
# from this list of twelve survives".
_INLINE_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]*>")

# Media embeds in the languages this corpus uses. `Media:`/`Imagen:` (the
# es-wiki legacy alias) are deliberately excluded: `[[Media:song.ogg|listen]]`
# is an inline prose link ("listen to this clip"), not an embed — stripping it
# whole would eat the anchor text. Task 3's link rewriter flattens it to its
# label like any other link.
_MEDIA_PREFIX = re.compile(r"\[\[\s*(?:File|Image|Immagine|Archivo|Imagen)\s*:", re.IGNORECASE)


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
    same rationale as `_strip_balanced`."""
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
    bodies are a payload (coordinate lists, LaTeX, source code), not prose."""
    return _CONTAINER_PAIR.sub("", _CONTAINER_SELF.sub("", text))


def strip_html_tags(text: str) -> str:
    """Strip any remaining HTML tag, keeping its content. Intended to run
    last in the pipeline, once every structural element (refs, containers,
    templates, tables, media) is already gone."""
    return _INLINE_TAG.sub("", text)
