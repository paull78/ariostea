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
"""

from __future__ import annotations

import re

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Self-closing refs first: otherwise the paired pattern swallows text between a
# `<ref name=x />` and the next real `</ref>`.
_REF_SELF = re.compile(r"<ref[^>]*/\s*>", re.IGNORECASE)
_REF_PAIR = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_GALLERY = re.compile(r"<gallery.*?</gallery>", re.DOTALL | re.IGNORECASE)
_HTML_TAG = re.compile(
    r"</?(?:small|big|sub|sup|span|div|br|hr|nowiki|poem|blockquote|center|code)[^>]*>",
    re.IGNORECASE,
)
# Media embeds in the languages this corpus uses.
_MEDIA_PREFIX = re.compile(r"\[\[\s*(?:File|Image|Media|Immagine|Archivo)\s*:", re.IGNORECASE)


def strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def strip_refs(text: str) -> str:
    return _REF_PAIR.sub("", _REF_SELF.sub("", text))


def _strip_balanced(text: str, open_tok: str, close_tok: str) -> str:
    """Drop every `open_tok`…`close_tok` region, honouring nesting."""
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith(open_tok, i):
            depth += 1
            i += len(open_tok)
        elif depth and text.startswith(close_tok, i):
            depth -= 1
            i += len(close_tok)
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def strip_templates(text: str) -> str:
    return _strip_balanced(text, "{{", "}}")


def strip_tables(text: str) -> str:
    return _strip_balanced(text, "{|", "|}")


def strip_media_links(text: str) -> str:
    """Remove `[[File:…]]` embeds whole. Captions can contain their own
    `[[links]]`, so the closing `]]` has to be found by depth, not by regex."""
    out: list[str] = []
    i = 0
    while i < len(text):
        match = _MEDIA_PREFIX.match(text, i)
        if not match:
            out.append(text[i])
            i += 1
            continue
        depth = 1
        j = match.end()
        while j < len(text) and depth:
            if text.startswith("[[", j):
                depth += 1
                j += 2
            elif text.startswith("]]", j):
                depth -= 1
                j += 2
            else:
                j += 1
        i = j
    return "".join(out)


def strip_html_tags(text: str) -> str:
    return _HTML_TAG.sub("", _GALLERY.sub("", text))
