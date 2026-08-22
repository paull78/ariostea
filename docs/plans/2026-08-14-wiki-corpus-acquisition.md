# Wikipedia Corpus Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pinned, attribution-complete Wikipedia snapshot the span-anchored eval runs on — six densely-linked topic clusters (~75 articles) converted to Obsidian-flavored Markdown under `eval/wiki/`, reproducible from a committed manifest of revision IDs.

**Architecture:** Every piece of judgment lives in pure, unit-tested modules under `src/ariostea/eval/` — wikitext→Markdown conversion, the cluster manifest, note/NOTICE rendering, and a thin httpx fetch function. `eval/build_wiki_corpus.py` is a dumb orchestrator over those parts, mirroring how `eval/run_eval.py` sits on top of `eval/channels.py`. Conversion works from **raw wikitext** (MediaWiki `action=query&prop=revisions`), not rendered HTML: wikitext already carries `[[Article|label]]` links, so wikilink rewriting is close to free and no HTML-parsing dependency is needed.

**Tech Stack:** Python 3.12, `httpx` (already a dependency), `re` + small hand-written balanced-token scanners, `pytest` with `httpx.MockTransport` for the network boundary.

This is Plan 2 of 3 for the eval-corpus expansion (spec: `docs/design/2026-07-09-eval-corpus-expansion.md`, section "1. Corpus acquisition and conversion"). Plan 1 (span-anchored gold schema + harness upgrade) is merged. Plan 3 generates the gold set against the corpus this plan produces.

---

## File Structure

- Create `src/ariostea/eval/wikitext.py` — pure wikitext → Obsidian Markdown conversion (chrome stripping, structure conversion, link rewriting, composition).
- Create `src/ariostea/eval/wiki_manifest.py` — `ArticleSpec` / `Cluster` dataclasses, slug rules, JSON load/save, note paths, per-language link-target maps.
- Create `src/ariostea/eval/wiki_corpus.py` — note rendering (frontmatter + body), revision permalinks, NOTICE attribution rows.
- Create `src/ariostea/eval/wiki_fetch.py` — `fetch_article`, the single network call, injectable client.
- Create `eval/build_wiki_corpus.py` — the offline build script (CLI).
- Create `eval/wiki/clusters.json` — hand-authored manifest; the build script writes pinned `revid`s back into it.
- Create `eval/wiki/<cluster>/<slug>.md` — the snapshot (~75 files, written by the script).
- Modify `eval/wiki/NOTICE` — attribution rows appended below the existing sentinel line.
- Modify `README.md` — one paragraph noting `eval/wiki/` is third-party CC BY-SA data, distinct from the MIT code.
- Tests: `tests/eval/test_wikitext.py`, `tests/eval/test_wiki_manifest.py`, `tests/eval/test_wiki_corpus.py`, `tests/eval/test_wiki_fetch.py`, `tests/eval/test_wiki_corpus_snapshot.py`.

Nothing under `src/ariostea/` outside `eval/` is touched — this plan builds evaluation capability only.

**Branch:** work on `feat/eval-wiki-corpus`, cut from `master`.

```bash
git checkout master && git pull && git checkout -b feat/eval-wiki-corpus
```

---

## Task 1: Strip wikitext chrome

Wikipedia wikitext is mostly prose wrapped in machinery: templates (`{{...}}`), tables (`{|...|}`), footnotes (`<ref>...</ref>`), media embeds (`[[File:...]]`), HTML comments and inline tags. All of it has to go before anything else can be parsed, and templates/tables/media **nest**, so regexes alone are not enough — this task adds a small balanced-token scanner.

**Files:**
- Create: `src/ariostea/eval/wikitext.py`
- Test: `tests/eval/test_wikitext.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_wikitext.py
from ariostea.eval.wikitext import (
    strip_comments,
    strip_html_tags,
    strip_media_links,
    strip_refs,
    strip_tables,
    strip_templates,
)


def test_strip_comments_and_refs():
    raw = "A<!-- hidden -->B<ref name=x>Smith 2001</ref>C<ref name=y />D"
    assert strip_refs(strip_comments(raw)) == "ABCD"


def test_strip_refs_removes_the_references_placeholder():
    assert strip_refs("body\n<references/>\n") == "body\n\n"


def test_strip_templates_handles_nesting():
    assert strip_templates("a {{convert|4|{{frac|1|2}} ft}} b") == "a  b"


def test_strip_tables_removes_the_whole_table():
    assert strip_tables("x\n{| class=wikitable\n|-\n| cell\n|}\ny") == "x\n\ny"


def test_strip_media_links_removes_captions_including_nested_links():
    assert strip_media_links("[[File:V.jpg|thumb|A [[violin]] on a table]]Text") == "Text"
    assert strip_media_links("[[Immagine:V.jpg|thumb|foto]]Testo") == "Testo"


def test_strip_media_links_leaves_ordinary_links_alone():
    assert strip_media_links("a [[violin]] b") == "a [[violin]] b"


def test_strip_html_tags_removes_galleries_and_inline_tags():
    assert strip_html_tags("<small>a</small><gallery>\nFile:x.jpg\n</gallery>b") == "ab"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wikitext.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.wikitext'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/eval/wikitext.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wikitext.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wikitext.py tests/eval/test_wikitext.py
git commit -m "feat(eval): strip wikitext templates, refs, tables and media embeds"
```

---

## Task 2: Convert wikitext structure to Markdown

Headings, lists, and emphasis. Order matters and is easy to get wrong: wikitext numbered lists start with `#`, which is a *heading* character in Markdown, so lists must be converted **before** headings. Boilerplate sections (References, See also, and their it/es equivalents) are dropped after heading conversion, when nesting levels are visible.

**Files:**
- Modify: `src/ariostea/eval/wikitext.py`
- Test: `tests/eval/test_wikitext.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/eval/test_wikitext.py
from ariostea.eval.wikitext import (
    convert_formatting,
    convert_headings,
    convert_lists,
    drop_sections,
    normalize_blank_lines,
)


def test_convert_headings_maps_equals_depth_to_hashes():
    assert convert_headings("== Construction ==\n=== Body ===") == "## Construction\n### Body"


def test_convert_lists_handles_bullets_numbers_and_definitions():
    raw = "* one\n** two\n# first\n## second\n: indented"
    assert convert_lists(raw) == "- one\n  - two\n1. first\n  1. second\nindented"


def test_convert_formatting_maps_quotes_to_asterisks():
    raw = "The '''violin''' is a ''chordophone''."
    assert convert_formatting(raw) == "The **violin** is a *chordophone*."


def test_drop_sections_removes_the_section_and_its_subsections():
    md = "## Construction\nbody\n\n## References\n- r1\n### Notes\nx\n\n## Playing\nmusic"
    assert drop_sections(md) == "## Construction\nbody\n\n## Playing\nmusic"


def test_drop_sections_covers_italian_and_spanish_boilerplate():
    assert drop_sections("## Storia\nc\n## Voci correlate\nx") == "## Storia\nc"
    assert drop_sections("## Historia\nc\n## Enlaces externos\nx") == "## Historia\nc"


def test_normalize_blank_lines_collapses_runs_and_trailing_space():
    assert normalize_blank_lines("a  \n\n\n\nb\n") == "a\n\nb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wikitext.py -v`
Expected: FAIL with `ImportError: cannot import name 'convert_headings'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/ariostea/eval/wikitext.py
_HEADING = re.compile(r"^[ \t]*(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.MULTILINE)
_BULLET = re.compile(r"^(\*+)[ \t]*(.*)$", re.MULTILINE)
_NUMBERED = re.compile(r"^(#+)[ \t]*(.*)$", re.MULTILINE)
_DEF_LINE = re.compile(r"^[;:]+[ \t]*", re.MULTILINE)
# Emphasis is deliberately line-bounded (no DOTALL): a stray unmatched quote
# run would otherwise italicise half the article.
_BOLD = re.compile(r"'''(.+?)'''")
_ITALIC = re.compile(r"''(.+?)''")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")

# Apparatus sections: no prose worth retrieving, and their link lists would
# pollute the wikilink graph. Lower-cased headings, en + it + es.
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
    return _HEADING.sub(lambda m: "#" * len(m.group(1)) + " " + m.group(2), text)


def convert_lists(text: str) -> str:
    """Bullets to `-`, numbered items to `1.`, two spaces per nesting level.

    Must run before `convert_headings`: a wikitext numbered item starts with
    `#`, which Markdown would otherwise read as a heading.
    """
    text = _BULLET.sub(lambda m: "  " * (len(m.group(1)) - 1) + "- " + m.group(2), text)
    text = _NUMBERED.sub(lambda m: "  " * (len(m.group(1)) - 1) + "1. " + m.group(2), text)
    return _DEF_LINE.sub("", text)


def convert_formatting(text: str) -> str:
    return _ITALIC.sub(r"*\1*", _BOLD.sub(r"**\1**", text))


def drop_sections(markdown: str, titles: frozenset[str] = DROP_SECTIONS) -> str:
    """Drop each named section together with everything nested under it, up to
    the next heading at the same or a shallower level."""
    kept: list[str] = []
    skip_level = 0
    for line in markdown.splitlines():
        heading = _MD_HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_level and level <= skip_level:
                skip_level = 0
            if not skip_level and heading.group(2).strip().lower() in titles:
                skip_level = level
                continue
        if not skip_level:
            kept.append(line)
    return "\n".join(kept)


def normalize_blank_lines(text: str) -> str:
    return _BLANK_RUN.sub("\n\n", _TRAILING_WS.sub("", text)).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wikitext.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wikitext.py tests/eval/test_wikitext.py
git commit -m "feat(eval): convert wikitext headings, lists, emphasis and drop apparatus sections"
```

---

## Task 3: Rewrite links and compose the converter

The point of the whole exercise: a link to an article **inside this corpus** becomes an Obsidian `[[wikilink]]` (graph-ready, per the design doc); a link to anything else degrades to its plain label. Targets are matched on whitespace/case-normalized titles, reusing `ariostea.eval.normalize.normalize_ws` from Plan 1.

Two wikitext details worth knowing: `[[Violin#Tuning|tuning]]` links to a section (the `#…` part is not part of the title), and `[[violin]]s` is a "link trail" — the trailing letters render as part of the link text.

**Files:**
- Modify: `src/ariostea/eval/wikitext.py`
- Test: `tests/eval/test_wikitext.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/eval/test_wikitext.py
from ariostea.eval.wikitext import convert_links, strip_external_links, wikitext_to_markdown

TARGETS = {"violin": "violin", "double bass": "double-bass", "string instrument": "string-instrument"}


def test_convert_links_rewrites_in_corpus_links_with_the_original_label():
    raw = "The [[Violin]] and the [[Double bass|contrabass]]."
    assert convert_links(raw, TARGETS) == "The [[violin|Violin]] and the [[double-bass|contrabass]]."


def test_convert_links_drops_out_of_corpus_links_to_plain_text():
    assert convert_links("A [[Spruce]] top.", TARGETS) == "A Spruce top."


def test_convert_links_keeps_the_link_trail_in_the_label():
    assert convert_links("two [[violin]]s", TARGETS) == "two [[violin|violins]]"


def test_convert_links_ignores_the_section_part_of_a_target():
    assert convert_links("see [[Violin#Tuning|tuning]]", TARGETS) == "see [[violin|tuning]]"


def test_convert_links_emits_the_bare_form_when_label_equals_slug():
    assert convert_links("a [[violin]] b", TARGETS) == "a [[violin]] b"


def test_strip_external_links_keeps_the_label_only():
    assert strip_external_links("see [https://x.org the site] and [https://y.org]") == "see the site and "


def test_wikitext_to_markdown_end_to_end():
    raw = (
        "{{Infobox instrument|name=Violin}}\n"
        "The '''violin''' is a [[String instrument|string instrument]].<ref>Smith</ref>\n"
        "\n"
        "== Construction ==\n"
        "[[File:Violin.jpg|thumb|A [[violin]]]]\n"
        "It has a [[Spruce]] top and is tuned in perfect fifths.\n"
        "\n"
        "* four strings\n"
        "** tuned G, D, A, E\n"
        "\n"
        "== References ==\n"
        "<references/>\n"
    )

    assert wikitext_to_markdown(raw, title="Violin", targets=TARGETS) == (
        "# Violin\n"
        "\n"
        "The **violin** is a [[string-instrument|string instrument]].\n"
        "\n"
        "## Construction\n"
        "\n"
        "It has a Spruce top and is tuned in perfect fifths.\n"
        "\n"
        "- four strings\n"
        "  - tuned G, D, A, E\n"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wikitext.py -v`
Expected: FAIL with `ImportError: cannot import name 'convert_links'`

- [ ] **Step 3: Write minimal implementation**

Add the import at the top of `src/ariostea/eval/wikitext.py`, below `import re`:

```python
from ariostea.eval.normalize import normalize_ws
```

Then append:

```python
# [[Target]] / [[Target|label]] / [[Target#Section|label]], plus the trailing
# "link trail" letters MediaWiki folds into the rendered label ([[violin]]s).
_LINK = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\[\]]*))?\]\](\w*)")
_EXT_LINK = re.compile(r"\[(?:https?:)?//\S+?(?:[ \t]+([^\]]*))?\]")


def convert_links(text: str, targets: dict[str, str]) -> str:
    """Rewrite links to corpus articles as Obsidian wikilinks; flatten the rest.

    `targets` maps a normalized article title to the slug of the note it became
    (see `wiki_manifest.link_targets`). The rendered label is preserved as a
    wikilink alias so the prose still reads naturally.
    """

    def replace(match: re.Match[str]) -> str:
        target, label, trail = match.group(1), match.group(2), match.group(3)
        page = target.split("#", 1)[0]
        display = (label or target).strip() + trail
        slug = targets.get(normalize_ws(page.replace("_", " ")))
        if slug is None:
            return display
        return f"[[{slug}]]" if display == slug else f"[[{slug}|{display}]]"

    return _LINK.sub(replace, text)


def strip_external_links(text: str) -> str:
    return _EXT_LINK.sub(lambda m: m.group(1) or "", text)


def wikitext_to_markdown(raw: str, title: str, targets: dict[str, str]) -> str:
    """Full pipeline: raw wikitext in, note body (H1 + Markdown) out."""
    text = strip_comments(raw)
    text = strip_refs(text)
    text = strip_templates(text)
    text = strip_tables(text)
    text = strip_media_links(text)
    text = strip_html_tags(text)
    text = strip_external_links(text)
    text = convert_lists(text)
    text = convert_headings(text)
    text = convert_links(text, targets)
    text = convert_formatting(text)
    text = drop_sections(text)
    return f"# {title}\n\n{normalize_blank_lines(text)}".rstrip() + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wikitext.py -v`
Expected: PASS (20 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wikitext.py tests/eval/test_wikitext.py
git commit -m "feat(eval): rewrite in-corpus links as wikilinks and compose the converter"
```

---

## Task 4: Cluster manifest

The manifest is the reproducibility contract: which articles, in which language, pinned to which revision. Slugs are derived, not authored, so a note path can never drift from its title. Non-English articles carry a language suffix because `Violin` (en) and `Violín` (es) would otherwise collide in the same cluster.

**Files:**
- Create: `src/ariostea/eval/wiki_manifest.py`
- Test: `tests/eval/test_wiki_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_wiki_manifest.py
import json

from ariostea.eval.wiki_manifest import (
    ArticleSpec,
    Cluster,
    link_targets,
    load_manifest,
    note_path,
    save_manifest,
    slugify,
)


def test_slugify_is_ascii_lowercase_and_hyphenated():
    assert slugify("Double bass") == "double-bass"
    assert slugify("Go (game)") == "go-game"
    assert slugify("Violín") == "violin"


def test_slug_is_language_qualified_outside_english():
    assert ArticleSpec(title="Violin", lang="en").slug == "violin"
    assert ArticleSpec(title="Violín", lang="es").slug == "violin-es"


def test_note_path_puts_the_slug_under_its_cluster():
    assert note_path("string-instruments", ArticleSpec(title="Cello", lang="en")) == (
        "string-instruments/cello.md"
    )


def test_link_targets_maps_normalized_titles_per_language():
    clusters = (
        Cluster(
            name="string-instruments",
            articles=(
                ArticleSpec(title="Double bass", lang="en"),
                ArticleSpec(title="Violino", lang="it"),
            ),
        ),
    )
    assert link_targets(clusters, "en") == {"double bass": "double-bass"}
    assert link_targets(clusters, "it") == {"violino": "violino-it"}


def test_manifest_roundtrips_through_json_keeping_pinned_revisions(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "name": "coffee",
                        "articles": [
                            {"title": "Espresso", "lang": "en", "revid": 991},
                            {"title": "Latte", "lang": "en"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    clusters = load_manifest(path)
    assert clusters == (
        Cluster(
            name="coffee",
            articles=(
                ArticleSpec(title="Espresso", lang="en", revid=991),
                ArticleSpec(title="Latte", lang="en", revid=None),
            ),
        ),
    )

    out = tmp_path / "out.json"
    save_manifest(out, clusters)
    assert load_manifest(out) == clusters
    # An unpinned article is written without a revid key, not with a null.
    assert "revid" not in json.loads(out.read_text(encoding="utf-8"))["clusters"][0]["articles"][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wiki_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.wiki_manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/eval/wiki_manifest.py
"""The eval corpus manifest: which Wikipedia articles form which cluster, and
the exact revision each one is pinned to.

The manifest is the reproducibility contract. Note paths are *derived* from the
article title, never authored, so a file can never drift from the article it
claims to be.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ariostea.eval.normalize import normalize_ws

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """ASCII, lower-case, hyphen-separated: 'Go (game)' -> 'go-game'."""
    decomposed = unicodedata.normalize("NFKD", title.lower())
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_SLUG.sub("-", ascii_only).strip("-")


@dataclass(frozen=True)
class ArticleSpec:
    title: str
    lang: str
    revid: int | None = None

    @property
    def slug(self) -> str:
        """Note stem. Non-English articles are language-suffixed so parallel
        titles ('Violin' / 'Violín') cannot collide inside a cluster."""
        base = slugify(self.title)
        return base if self.lang == "en" else f"{base}-{self.lang}"


@dataclass(frozen=True)
class Cluster:
    name: str
    articles: tuple[ArticleSpec, ...]


def note_path(cluster: str, article: ArticleSpec) -> str:
    return f"{cluster}/{article.slug}.md"


def link_targets(clusters: tuple[Cluster, ...], lang: str) -> dict[str, str]:
    """Normalized article title -> note slug, for one language edition. Links
    in an it.wikipedia article point at it titles, so the maps stay separate."""
    return {
        normalize_ws(article.title): article.slug
        for cluster in clusters
        for article in cluster.articles
        if article.lang == lang
    }


def load_manifest(path: str | Path) -> tuple[Cluster, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        Cluster(
            name=cluster["name"],
            articles=tuple(
                ArticleSpec(
                    title=article["title"],
                    lang=article["lang"],
                    revid=article.get("revid"),
                )
                for article in cluster["articles"]
            ),
        )
        for cluster in data["clusters"]
    )


def save_manifest(path: str | Path, clusters: tuple[Cluster, ...]) -> None:
    data = {
        "clusters": [
            {
                "name": cluster.name,
                "articles": [
                    {"title": a.title, "lang": a.lang}
                    | ({"revid": a.revid} if a.revid is not None else {})
                    for a in cluster.articles
                ],
            }
            for cluster in clusters
        ]
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wiki_manifest.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wiki_manifest.py tests/eval/test_wiki_manifest.py
git commit -m "feat(eval): cluster manifest with derived slugs and pinned revisions"
```

---

## Task 5: Note rendering and NOTICE attribution rows

Each note gets frontmatter that `ObsidianMarkdownParser` can read (simple `key: value` lines) and that carries the license trail: title, language, cluster, revision id, revision permalink. Each note also gets a row in `eval/wiki/NOTICE`, below the sentinel line Plan 1 left there. Row rewriting is idempotent — the build script may run many times.

**Files:**
- Create: `src/ariostea/eval/wiki_corpus.py`
- Test: `tests/eval/test_wiki_corpus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_wiki_corpus.py
from ariostea.eval.wiki_corpus import SENTINEL, notice_row, permalink, render_note, write_notice_rows
from ariostea.eval.wiki_manifest import ArticleSpec

VIOLIN = ArticleSpec(title="Violin", lang="en", revid=1234)
VIOLIN_ES = ArticleSpec(title="Violín", lang="es", revid=99)


def test_permalink_points_at_the_pinned_revision():
    assert permalink(VIOLIN) == "https://en.wikipedia.org/w/index.php?title=Violin&oldid=1234"


def test_permalink_percent_encodes_non_ascii_titles():
    assert permalink(VIOLIN_ES) == "https://es.wikipedia.org/w/index.php?title=Viol%C3%ADn&oldid=99"


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


def test_notice_row_records_path_title_language_and_permalink():
    assert notice_row("string-instruments", VIOLIN) == (
        "string-instruments/violin.md | Violin | en | "
        "https://en.wikipedia.org/w/index.php?title=Violin&oldid=1234"
    )


def test_write_notice_rows_replaces_everything_after_the_sentinel_and_is_idempotent():
    notice = f"Header text\n\n{SENTINEL}\nstale row\n"
    once = write_notice_rows(notice, ["b | B | en | u2", "a | A | en | u1"])
    assert once == f"Header text\n\n{SENTINEL}\n\na | A | en | u1\nb | B | en | u2\n"
    assert write_notice_rows(once, ["b | B | en | u2", "a | A | en | u1"]) == once
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wiki_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.wiki_corpus'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/eval/wiki_corpus.py
"""Render fetched Wikipedia articles as corpus notes, and keep the CC BY-SA
attribution table in `eval/wiki/NOTICE` in sync with them.

Attribution is mechanical on purpose: every note carries its revision
permalink in frontmatter, and every note has exactly one NOTICE row pointing
at the same revision, so the license trail cannot silently rot.
"""

from __future__ import annotations

from urllib.parse import quote

from ariostea.eval.wiki_manifest import ArticleSpec, note_path

# The line `eval/wiki/NOTICE` reserves for machine-maintained rows. Everything
# below it is rewritten by the build script; everything above is hand-written.
SENTINEL = "--- articles below this line are maintained by the build script ---"


def permalink(article: ArticleSpec) -> str:
    title = quote(article.title.replace(" ", "_"))
    return f"https://{article.lang}.wikipedia.org/w/index.php?title={title}&oldid={article.revid}"


def render_note(cluster: str, article: ArticleSpec, body: str) -> str:
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
    return (
        f"{note_path(cluster, article)} | {article.title} | {article.lang} | {permalink(article)}"
    )


def write_notice_rows(notice_text: str, rows: list[str]) -> str:
    """Return NOTICE with its attribution table replaced by `rows`, sorted."""
    head, sep, _ = notice_text.partition(SENTINEL)
    if not sep:
        raise ValueError(f"NOTICE is missing its sentinel line: {SENTINEL!r}")
    return head + SENTINEL + "\n\n" + "\n".join(sorted(rows)) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wiki_corpus.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wiki_corpus.py tests/eval/test_wiki_corpus.py
git commit -m "feat(eval): render corpus notes and maintain the NOTICE attribution table"
```

---

## Task 6: Wikipedia fetch

One function, one HTTP call, an injectable `httpx.Client` so the tests never touch the network — the same shape as `OpenAICompatChat`. It fetches by title (recording whichever revision is current) or by pinned `revid` (reproducing the committed snapshot exactly).

**Files:**
- Create: `src/ariostea/eval/wiki_fetch.py`
- Test: `tests/eval/test_wiki_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_wiki_fetch.py
import httpx
import pytest

from ariostea.eval.wiki_fetch import FetchedArticle, WikiFetchError, fetch_article


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _page(**over):
    page = {"title": "Violin", "revisions": [{"revid": 42, "slots": {"main": {"content": "raw"}}}]}
    page.update(over)
    return {"query": {"pages": [page]}}


def test_fetch_article_requests_wikitext_by_title_and_returns_the_revision():
    seen = {}

    def handler(request):
        seen["host"] = request.url.host
        seen["params"] = dict(request.url.params)
        seen["agent"] = request.headers.get("user-agent")
        return httpx.Response(200, json=_page())

    article = fetch_article(_client(handler), lang="en", title="Violin")

    assert article == FetchedArticle(title="Violin", revid=42, wikitext="raw")
    assert seen["host"] == "en.wikipedia.org"
    assert seen["params"]["titles"] == "Violin"
    assert seen["params"]["rvslots"] == "main"
    assert seen["params"]["redirects"] == "1"
    assert "ariostea" in seen["agent"]


def test_fetch_article_pins_by_revid_when_given_one():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_page())

    fetch_article(_client(handler), lang="en", title="Violin", revid=42)

    assert seen["params"]["revids"] == "42"
    assert "titles" not in seen["params"]


def test_fetch_article_raises_on_a_missing_page():
    def handler(request):
        return httpx.Response(200, json={"query": {"pages": [{"title": "Nope", "missing": True}]}})

    with pytest.raises(WikiFetchError, match="no revision"):
        fetch_article(_client(handler), lang="en", title="Nope")


def test_fetch_article_raises_on_an_http_error():
    def handler(request):
        return httpx.Response(503, text="down")

    with pytest.raises(WikiFetchError, match="503"):
        fetch_article(_client(handler), lang="en", title="Violin")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wiki_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.wiki_fetch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/eval/wiki_fetch.py
"""Fetch raw wikitext from the MediaWiki API.

The only networked part of the corpus build, kept to one function with an
injectable client so everything around it stays testable offline. Wikimedia
asks API clients to identify themselves, hence the explicit User-Agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

USER_AGENT = "ariostea-eval-corpus/0.1 (https://github.com/paull78/ariostea; corpus build script)"


class WikiFetchError(RuntimeError):
    """An article could not be fetched, or came back in an unusable shape."""


@dataclass(frozen=True)
class FetchedArticle:
    title: str
    revid: int
    wikitext: str


def fetch_article(
    client: httpx.Client, lang: str, title: str, revid: int | None = None
) -> FetchedArticle:
    """Return one article's wikitext. With `revid`, fetch that exact revision;
    without it, fetch the current one and report which revision that was."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content|ids",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    params["revids" if revid is not None else "titles"] = str(revid) if revid else title

    try:
        response = client.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
    except httpx.HTTPError as exc:
        raise WikiFetchError(f"{lang}:{title}: request failed: {exc}") from exc
    if response.status_code >= 400:
        raise WikiFetchError(f"{lang}:{title}: HTTP {response.status_code} {response.text[:200]}")

    try:
        page = response.json()["query"]["pages"][0]
        revision = page["revisions"][0]
        return FetchedArticle(
            title=page["title"],
            revid=int(revision["revid"]),
            wikitext=revision["slots"]["main"]["content"],
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise WikiFetchError(f"{lang}:{title}: no revision in response: {response.text[:200]}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wiki_fetch.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wiki_fetch.py tests/eval/test_wiki_fetch.py
git commit -m "feat(eval): fetch pinned Wikipedia revisions as wikitext"
```

---

## Task 7: Author the cluster manifest

Six clusters, 75 articles. Each cluster is internally dense: its articles link to each other and share vocabulary, which is what turns them into hard negatives ("four strings tuned in fifths" is true of the violin *and* the cello). Italian and Spanish parallels sit in `string-instruments`, and Italian in `coffee`, to feed the `cross_lingual` query type.

**Files:**
- Create: `eval/wiki/clusters.json`
- Test: `tests/eval/test_wiki_manifest.py`

- [ ] **Step 1: Write the manifest**

```json
{
  "clusters": [
    {
      "name": "string-instruments",
      "articles": [
        {"title": "Violin", "lang": "en"},
        {"title": "Viola", "lang": "en"},
        {"title": "Cello", "lang": "en"},
        {"title": "Double bass", "lang": "en"},
        {"title": "Classical guitar", "lang": "en"},
        {"title": "Mandolin", "lang": "en"},
        {"title": "Banjo", "lang": "en"},
        {"title": "Harp", "lang": "en"},
        {"title": "Lute", "lang": "en"},
        {"title": "Ukulele", "lang": "en"},
        {"title": "Luthier", "lang": "en"},
        {"title": "Fingerboard", "lang": "en"},
        {"title": "Violino", "lang": "it"},
        {"title": "Violoncello", "lang": "it"},
        {"title": "Mandolino", "lang": "it"},
        {"title": "Violín", "lang": "es"},
        {"title": "Violonchelo", "lang": "es"},
        {"title": "Guitarra clásica", "lang": "es"}
      ]
    },
    {
      "name": "coffee",
      "articles": [
        {"title": "Espresso", "lang": "en"},
        {"title": "Moka pot", "lang": "en"},
        {"title": "French press", "lang": "en"},
        {"title": "Coffee bean", "lang": "en"},
        {"title": "Coffea arabica", "lang": "en"},
        {"title": "Robusta coffee", "lang": "en"},
        {"title": "Coffee roasting", "lang": "en"},
        {"title": "Latte", "lang": "en"},
        {"title": "Cappuccino", "lang": "en"},
        {"title": "Cold brew coffee", "lang": "en"},
        {"title": "Coffee preparation", "lang": "en"},
        {"title": "Caffè espresso", "lang": "it"},
        {"title": "Cappuccino", "lang": "it"}
      ]
    },
    {
      "name": "cheese",
      "articles": [
        {"title": "Cheese", "lang": "en"},
        {"title": "Cheddar cheese", "lang": "en"},
        {"title": "Mozzarella", "lang": "en"},
        {"title": "Parmigiano Reggiano", "lang": "en"},
        {"title": "Gouda cheese", "lang": "en"},
        {"title": "Brie", "lang": "en"},
        {"title": "Blue cheese", "lang": "en"},
        {"title": "Rennet", "lang": "en"},
        {"title": "Curd", "lang": "en"},
        {"title": "Whey", "lang": "en"},
        {"title": "Cheese ripening", "lang": "en"}
      ]
    },
    {
      "name": "cycling",
      "articles": [
        {"title": "Bicycle", "lang": "en"},
        {"title": "Road bicycle", "lang": "en"},
        {"title": "Mountain bike", "lang": "en"},
        {"title": "Track bicycle", "lang": "en"},
        {"title": "Derailleur gears", "lang": "en"},
        {"title": "Bicycle frame", "lang": "en"},
        {"title": "Bicycle brake", "lang": "en"},
        {"title": "Bicycle wheel", "lang": "en"},
        {"title": "Bicycle pedal", "lang": "en"},
        {"title": "Bicycle chain", "lang": "en"},
        {"title": "Bicycle handlebar", "lang": "en"}
      ]
    },
    {
      "name": "sailing",
      "articles": [
        {"title": "Sailing", "lang": "en"},
        {"title": "Sailboat", "lang": "en"},
        {"title": "Mainsail", "lang": "en"},
        {"title": "Jib", "lang": "en"},
        {"title": "Spinnaker", "lang": "en"},
        {"title": "Keel", "lang": "en"},
        {"title": "Rigging", "lang": "en"},
        {"title": "Mast (sailing)", "lang": "en"},
        {"title": "Boom (sailing)", "lang": "en"},
        {"title": "Rudder", "lang": "en"},
        {"title": "Tacking (sailing)", "lang": "en"}
      ]
    },
    {
      "name": "board-games",
      "articles": [
        {"title": "Chess", "lang": "en"},
        {"title": "Chess opening", "lang": "en"},
        {"title": "Chess endgame", "lang": "en"},
        {"title": "Chess piece", "lang": "en"},
        {"title": "Chessboard", "lang": "en"},
        {"title": "Checkmate", "lang": "en"},
        {"title": "Draughts", "lang": "en"},
        {"title": "Go (game)", "lang": "en"},
        {"title": "Shogi", "lang": "en"},
        {"title": "Xiangqi", "lang": "en"},
        {"title": "Backgammon", "lang": "en"}
      ]
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# append to tests/eval/test_wiki_manifest.py
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "eval" / "wiki" / "clusters.json"


def test_committed_manifest_matches_the_design_targets():
    clusters = load_manifest(MANIFEST)
    articles = [a for c in clusters for a in c.articles]

    assert 5 <= len(clusters) <= 6
    assert 60 <= len(articles) <= 80
    # Slugs are note filenames: a collision would silently overwrite an article.
    slugs = [f"{c.name}/{a.slug}" for c in clusters for a in c.articles]
    assert len(set(slugs)) == len(slugs)
    # The cross_lingual query type needs non-English parallels in at least one cluster.
    assert {a.lang for a in articles} >= {"en", "it", "es"}
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/eval/test_wiki_manifest.py -v`
Expected: PASS — the manifest was written in Step 1, so this test documents and locks the shape rather than driving it. If it fails, the counts or slugs in the JSON are wrong; fix the JSON.

- [ ] **Step 4: Commit**

```bash
git add eval/wiki/clusters.json tests/eval/test_wiki_manifest.py
git commit -m "feat(eval): cluster manifest for the Wikipedia eval corpus"
```

---

## Task 8: Build script and a single-cluster trial run

The script is deliberately thin — load manifest, fetch, convert, write, pin, rewrite NOTICE. This task also does the first **real network run**, on one cluster only, so conversion output gets eyeballed before 75 files land in git.

**Files:**
- Create: `eval/build_wiki_corpus.py`

- [ ] **Step 1: Write the script**

```python
# eval/build_wiki_corpus.py
"""Fetch the pinned Wikipedia snapshot for the eval corpus and convert it to
Obsidian-flavored Markdown under eval/wiki/<cluster>/.

Usage:  uv run python eval/build_wiki_corpus.py [--cluster NAME] [--refresh]

Reads eval/wiki/clusters.json. Articles that already carry a `revid` are
fetched at that exact revision, so a re-run reproduces the committed snapshot;
--refresh re-pins every article to the current latest revision. The manifest is
written back with whatever revisions were pinned, and the attribution table in
eval/wiki/NOTICE is regenerated from the files on disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

from ariostea.eval.wiki_corpus import notice_row, render_note, write_notice_rows
from ariostea.eval.wiki_fetch import WikiFetchError, fetch_article
from ariostea.eval.wiki_manifest import (
    ArticleSpec,
    Cluster,
    link_targets,
    load_manifest,
    note_path,
    save_manifest,
)
from ariostea.eval.wikitext import wikitext_to_markdown

WIKI_DIR = Path(__file__).resolve().parent / "wiki"
MANIFEST = WIKI_DIR / "clusters.json"
NOTICE = WIKI_DIR / "NOTICE"
# Wikimedia asks for serial, unhurried API access from batch clients.
DELAY_S = 0.2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", help="build only this cluster")
    parser.add_argument(
        "--refresh", action="store_true", help="re-pin every article to the latest revision"
    )
    args = parser.parse_args()

    clusters = load_manifest(MANIFEST)
    langs = {a.lang for c in clusters for a in c.articles}
    targets = {lang: link_targets(clusters, lang) for lang in langs}

    built: list[Cluster] = []
    failures: list[str] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for cluster in clusters:
            if args.cluster and cluster.name != args.cluster:
                built.append(cluster)
                continue
            print(f"\n=== {cluster.name} ===")
            articles: list[ArticleSpec] = []
            for article in cluster.articles:
                try:
                    fetched = fetch_article(
                        client,
                        lang=article.lang,
                        title=article.title,
                        revid=None if args.refresh else article.revid,
                    )
                except WikiFetchError as exc:
                    failures.append(f"{cluster.name}/{article.title}: {exc}")
                    articles.append(article)
                    continue
                if fetched.title != article.title:
                    print(f"  note: {article.title!r} redirects to {fetched.title!r}")
                pinned = ArticleSpec(title=article.title, lang=article.lang, revid=fetched.revid)
                body = wikitext_to_markdown(
                    fetched.wikitext, title=article.title, targets=targets[article.lang]
                )
                path = WIKI_DIR / note_path(cluster.name, pinned)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_note(cluster.name, pinned, body), encoding="utf-8")
                articles.append(pinned)
                print(f"  {note_path(cluster.name, pinned)}  rev {fetched.revid}  {len(body):,} chars")
                time.sleep(DELAY_S)
            built.append(Cluster(name=cluster.name, articles=tuple(articles)))

    save_manifest(MANIFEST, tuple(built))
    rows = [
        notice_row(c.name, a)
        for c in built
        for a in c.articles
        if a.revid is not None and (WIKI_DIR / note_path(c.name, a)).exists()
    ]
    NOTICE.write_text(write_notice_rows(NOTICE.read_text(encoding="utf-8"), rows), encoding="utf-8")
    print(f"\n{len(rows)} articles in the snapshot; NOTICE updated.")

    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Trial-run one cluster against live Wikipedia**

Run: `uv run python eval/build_wiki_corpus.py --cluster string-instruments`
Expected: 18 lines like `string-instruments/violin.md  rev 1298…  38,412 chars`, then `18 articles in the snapshot; NOTICE updated.` and exit code 0.

If an article fails with "no revision", the title is wrong or has been renamed — look it up on Wikipedia, fix the title in `eval/wiki/clusters.json`, and re-run.

- [ ] **Step 3: Eyeball the conversion**

Run:
```bash
head -40 eval/wiki/string-instruments/violin.md
grep -c '' eval/wiki/string-instruments/*.md | head
grep -o '\[\[[^]]*\]\]' eval/wiki/string-instruments/violin.md | sort -u | head
grep -n '{{\|{|\|<ref\|\[\[File:' eval/wiki/string-instruments/*.md | head
```
Expected: frontmatter with a real `revid` and permalink; an `# Violin` H1; `##` section headings; wikilinks such as `[[viola|viola]]` and `[[cello|cello]]` pointing only at corpus slugs; **no output** from the last command (no leftover markup).

If leftovers appear, fix the relevant function in `wikitext.py`, add a regression test to `tests/eval/test_wikitext.py` covering the exact construct that leaked, and re-run this step.

- [ ] **Step 4: Commit the script only (not the trial corpus yet)**

```bash
git add eval/build_wiki_corpus.py
git commit -m "feat(eval): build script for the pinned Wikipedia corpus"
```

---

## Task 9: Fetch the full snapshot and lock it with invariants

**Files:**
- Create: `eval/wiki/<cluster>/*.md` (~75 files, written by the script)
- Modify: `eval/wiki/clusters.json` (revision ids), `eval/wiki/NOTICE` (attribution rows)
- Test: `tests/eval/test_wiki_corpus_snapshot.py`

- [ ] **Step 1: Build every cluster**

Run: `uv run python eval/build_wiki_corpus.py`
Expected: all six clusters fetched, `75 articles in the snapshot; NOTICE updated.`, exit code 0. Takes roughly a minute (75 requests at 0.2 s apart plus transfer).

Fix any failing titles in the manifest and re-run until there are no `FAILED:` lines.

- [ ] **Step 2: Write the snapshot invariant test**

```python
# tests/eval/test_wiki_corpus_snapshot.py
"""Invariants the committed Wikipedia snapshot must satisfy. Reads files from
eval/wiki/ — no network, so this runs in the normal suite."""

import re
from pathlib import Path

import pytest

from ariostea.eval.wiki_manifest import load_manifest, note_path

WIKI = Path(__file__).resolve().parents[2] / "eval" / "wiki"
CLUSTERS = load_manifest(WIKI / "clusters.json")
ARTICLES = [(cluster.name, article) for cluster in CLUSTERS for article in cluster.articles]
IDS = [f"{cluster}/{article.slug}" for cluster, article in ARTICLES]
_WIKILINK = re.compile(r"\[\[([^\]|#]+)")
# Anything that means the converter leaked raw wikitext into the corpus.
LEFTOVERS = ("{{", "{|", "<ref", "[[File:", "<!--")


def _text(cluster: str, article) -> str:
    return (WIKI / note_path(cluster, article)).read_text(encoding="utf-8")


@pytest.mark.parametrize("cluster,article", ARTICLES, ids=IDS)
def test_every_article_is_pinned_attributed_and_substantial(cluster, article):
    assert article.revid is not None, "run eval/build_wiki_corpus.py to pin this article"
    text = _text(cluster, article)
    assert text.startswith("---\n")
    assert f"revid: {article.revid}\n" in text
    assert "license: CC BY-SA 4.0\n" in text
    assert f"\n# {article.title}\n" in text
    # Long enough to chunk into several passages — the whole point of the corpus.
    assert len(text) > 1200


@pytest.mark.parametrize("cluster,article", ARTICLES, ids=IDS)
def test_no_unconverted_wikitext_survives(cluster, article):
    text = _text(cluster, article)
    for leftover in LEFTOVERS:
        assert leftover not in text, f"{leftover!r} leaked into {note_path(cluster, article)}"


def test_every_wikilink_resolves_to_a_corpus_note():
    slugs = {article.slug for _, article in ARTICLES}
    for cluster, article in ARTICLES:
        for target in _WIKILINK.findall(_text(cluster, article)):
            assert target.strip() in slugs, f"dangling [[{target}]] in {note_path(cluster, article)}"


def test_notice_attributes_every_committed_note():
    notice = (WIKI / "NOTICE").read_text(encoding="utf-8")
    for cluster, article in ARTICLES:
        assert note_path(cluster, article) in notice
        assert f"oldid={article.revid}" in notice
```

- [ ] **Step 3: Run the invariant test**

Run: `uv run pytest tests/eval/test_wiki_corpus_snapshot.py -q`
Expected: PASS.

Two failures are plausible on real data and each has a specific fix:
- **`len(text) > 1200`** — that article is a stub. Replace it in `eval/wiki/clusters.json` with a longer sibling, delete the stub file, and re-run the build for that cluster.
- **A leftover marker** — the converter missed a construct. Add a regression test in `tests/eval/test_wikitext.py`, fix `wikitext.py`, then `uv run python eval/build_wiki_corpus.py` to regenerate (pinned revids make this deterministic).

- [ ] **Step 4: Confirm the snapshot is actually staged**

Run: `git status --short eval/wiki | head -20` and `git status --short eval/wiki | wc -l`
Expected: ~78 entries (75 notes plus `clusters.json`, `NOTICE`). If notes are missing, check `.gitignore` isn't excluding them.

- [ ] **Step 5: Commit the snapshot**

```bash
git add eval/wiki tests/eval/test_wiki_corpus_snapshot.py
git commit -m "feat(eval): pinned Wikipedia corpus snapshot with attribution"
```

---

## Task 10: README note and full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the third-party corpus in the README**

Add this to `README.md`, at the end of the section that discusses licensing (immediately before or after the existing MIT license line):

```markdown
### Evaluation corpus license

The Ariostea source code is MIT-licensed. `eval/wiki/` is different: it holds
Wikipedia article text used as a retrieval-evaluation corpus, licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and modified
(converted to Markdown, links rewritten as wikilinks, infoboxes and references
stripped). `eval/wiki/NOTICE` records the license, the modifications, and a
permalink to the exact revision of every article in the snapshot. Redistributing
that text — or adaptations of it — carries the share-alike obligation.
```

- [ ] **Step 2: Run lint, format check, and the fast suite**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not integration" -q
```
Expected: all pass. If `ruff format --check` reports changes, run `uv run ruff format .`, re-run, and amend.

- [ ] **Step 3: Confirm no production code changed**

Run: `git diff --name-only master -- src/ariostea | grep -v '^src/ariostea/eval/'`
Expected: empty output — this plan touches only `src/ariostea/eval/`, `eval/`, tests, and the README.

- [ ] **Step 4: Confirm the build is reproducible**

Run: `uv run python eval/build_wiki_corpus.py && git status --short eval/wiki`
Expected: the script re-fetches every article at its pinned revision and `git status` reports **no changes** — same revisions in, same bytes out. Any diff here means something non-deterministic leaked into the conversion; fix it before merging.

- [ ] **Step 5: Commit and open the PR**

```bash
git add README.md
git commit -m "docs: note the CC BY-SA license of the eval corpus"
git push -u origin feat/eval-wiki-corpus
gh pr create --title "eval: pinned Wikipedia corpus (Plan 2/3)" --body "..."
```

Note: `gh` must be acting as the `paull78` account for this repository — check with `gh auth status` and switch with `gh auth switch --user paull78` if needed.

---

## Self-Review

- **Spec coverage** (design doc section 1): offline script fetching via the Wikipedia API ✓ (Task 6, 8); pinned revision IDs for a frozen, reproducible corpus ✓ (Tasks 4, 8, 10 Step 4); headings and section structure preserved ✓ (Task 2); in-corpus links rewritten as `[[wikilinks]]`, out-of-corpus links flattened to plain text ✓ (Task 3); infoboxes, reference lists and navigation chrome stripped ✓ (Tasks 1, 2); frontmatter records source URL, revision ID and license ✓ (Task 5); output at `eval/wiki/<cluster>/<article>.md`, separate from `eval/corpus/` ✓ (Tasks 4, 9); `NOTICE` maintained mechanically with a row per article ✓ (Tasks 5, 8); README note on third-party CC BY-SA data ✓ (Task 10); 5–6 clusters and 60–80 articles with it/es parallels ✓ (Task 7 — six clusters, 75 articles, it + es in `string-instruments`, it in `coffee`). Testing section of the spec: wikilink rewriting limited to in-corpus articles ✓ (Task 3 + Task 9 dangling-link test); heading/structure preserved ✓ (Task 2 + snapshot H1/heading assertions); frontmatter and license present ✓ (Task 9).
- **Out of scope, by design:** gold generation and the validation gate (Plan 3), the difficulty guard (Plan 3, since it needs gold to score), and multi-hop/graph gold (deferred by the design doc).
- **Type consistency:** `ArticleSpec(title, lang, revid)` with a derived `.slug` is defined in Task 4 and used identically in Tasks 5, 8, 9. `note_path(cluster: str, article: ArticleSpec)` keeps the same argument order everywhere. `link_targets(clusters, lang) -> dict[str, str]` produces exactly the `targets` mapping `convert_links`/`wikitext_to_markdown` consume (Task 3). `FetchedArticle(title, revid, wikitext)` is produced in Task 6 and destructured in Task 8. `SENTINEL` is defined once in `wiki_corpus.py` and matches the line Plan 1 wrote into `eval/wiki/NOTICE`.
- **Placeholder scan:** every code step carries complete code; the only deliberately abbreviated string is the `gh pr create --body` text in the final step.
- **Known risk, accepted:** stripping (rather than expanding) templates removes template-borne values from prose, so a few sentences will read oddly. It costs no retrieval signal, it is recorded in `NOTICE` as a modification, and avoiding it would mean an HTML pipeline with two new dependencies.
