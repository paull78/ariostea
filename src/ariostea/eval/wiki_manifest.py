"""The eval corpus manifest: which Wikipedia articles form which cluster, and
the exact revision each one is pinned to.

The manifest is the reproducibility contract. Note paths are *derived* from
the article title, never authored, so a file can never drift from the
article it claims to be. Collisions -- two articles landing on the same
slug, or the same (title, language) pair appearing twice -- need no data
outside the manifest itself to detect, so `load_manifest` checks for them
eagerly rather than leaving them to surface later: a manifest typo caught at
build time costs a re-run of this module; the same typo caught by Task 9's
corpus-wide uniqueness check costs a full fetch-and-convert build first, and
silently wrong output (one note overwritten by another, or a wikilink
resolving to the wrong article) if nothing catches it at all.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ariostea.eval.normalize import normalize_ws

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """ASCII, lower-case, hyphen-separated: 'Go (game)' -> 'go-game'.

    Accents fold to their base letter via NFKD decomposition plus
    combining-mark removal ('Violín' -> 'violin'), not via a hand-maintained
    transliteration table. A title with no Latin-script fallback at all (a
    CJK title, say) folds to nothing -- `slugify` just reports that; it is
    `ArticleSpec.slug` below that treats an empty result as an error rather
    than handing back a blank note stem.
    """
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
        titles ('Violin' / 'Violín') cannot collide inside a cluster.

        Raises if the title slugifies to an empty string. Left unchecked,
        that becomes a note literally named '.md' (en) or '-zh.md'
        (non-en) -- a silent, hard-to-notice failure at write time, not a
        loud one at the point where the bad title is actually known.
        """
        base = slugify(self.title)
        if not base:
            raise ValueError(f"article {self.title!r} ({self.lang}) slugifies to an empty string")
        return base if self.lang == "en" else f"{base}-{self.lang}"


@dataclass(frozen=True)
class Cluster:
    name: str
    articles: tuple[ArticleSpec, ...]


def note_path(cluster: str, article: ArticleSpec) -> str:
    return f"{cluster}/{article.slug}.md"


def link_targets(clusters: tuple[Cluster, ...], lang: str) -> dict[str, str]:
    """Normalized article title -> note slug, for one language edition. Links
    in an it.wikipedia article point at it titles, so the maps stay separate.

    Raises on a duplicate normalized title within the language rather than
    silently keeping whichever cluster happens to come last (a plain dict
    comprehension would): a link elsewhere in the corpus that names that
    title would then resolve to whichever slug won, with no signal that the
    other article's inbound links are now pointing at the wrong note.
    `load_manifest` already rejects this shape at load time (see
    `_check_for_collisions`), so this only fires for manifests assembled by
    hand rather than loaded from JSON.
    """
    targets: dict[str, str] = {}
    for cluster in clusters:
        for article in cluster.articles:
            if article.lang != lang:
                continue
            key = normalize_ws(article.title)
            if key in targets:
                # Raise even when the two slugs happen to agree: a repeated
                # title still means the same article was listed twice, which
                # is a manifest defect regardless of whether it happens to be
                # harmless here.
                raise ValueError(
                    f"duplicate {lang} title {article.title!r}: maps to both "
                    f"{targets[key]!r} and {article.slug!r}"
                )
            targets[key] = article.slug
    return targets


def _check_for_collisions(clusters: tuple[Cluster, ...]) -> None:
    """Raise with every collision found, not just the first, so a manifest
    author fixing one typo doesn't have to re-run this to find the next.

    Two independent things can go wrong in a hand-authored manifest, and
    they need two independent checks: two *different* titles can slugify to
    the same stem ('Go (game)' and 'Go game' both -> 'go-game'), which a
    title-only check would miss; and the *same* title can be listed twice
    under the same language (whether that's a copy-paste mistake within one
    cluster or the same article claimed by two clusters), which a slug-only
    check would miss whenever -- as it usually will, since the slug is a
    deterministic function of the title -- the two slugs happen to agree
    rather than differ. Either shape breaks `link_targets`, which assumes
    exactly one slug per (title, language) across the whole manifest.
    """
    slug_owners: dict[str, str] = {}
    title_owners: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    for cluster in clusters:
        for article in cluster.articles:
            slug = article.slug  # may itself raise -- let it.
            owner = f"{cluster.name}/{article.title!r} ({article.lang})"
            if slug in slug_owners:
                errors.append(f"slug {slug!r} is claimed by both {slug_owners[slug]} and {owner}")
            else:
                slug_owners[slug] = owner

            title_key = (article.lang, normalize_ws(article.title))
            if title_key in title_owners:
                errors.append(
                    f"{article.lang} title {article.title!r} is claimed by both "
                    f"{title_owners[title_key]} and {owner}"
                )
            else:
                title_owners[title_key] = owner
    if errors:
        raise ValueError("manifest has collisions:\n" + "\n".join(errors))


def _require(data: Any, key: str, where: str) -> Any:
    """`data[key]`, but a message naming the exact cluster/article at fault
    instead of a bare `KeyError('lang')` -- this file is hand-authored (Task
    7), so a missing key is an expected kind of typo, not a freak occurrence.

    Also catches `TypeError`: `data` is only a `dict` if the JSON is shaped
    the way this module expects (e.g. an `articles` entry that's a bare
    string instead of an `{"title": ..., "lang": ...}` object is a `str`,
    and `data[key]` on a `str` raises `TypeError`, not `KeyError`). Either
    way the manifest is malformed in the same sense, so both get the same
    clear error instead of one of them crashing with a raw traceback.
    """
    try:
        return data[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{where}: missing required key {key!r}") from exc


def _article_from_json(cluster_name: str, index: int, data: dict[str, Any]) -> ArticleSpec:
    where = f"cluster {cluster_name!r} article {index}"
    title = _require(data, "title", where)
    lang = _require(data, "lang", where)
    if not lang:
        # Left unguarded, an empty lang still produces a *non-empty* slug
        # (base + "-" + "" -> "base-"), so `ArticleSpec.slug`'s empty-slug
        # check can't catch it -- it would silently become a note named
        # e.g. "foo-.md" instead of failing loudly here.
        raise ValueError(f"{where} ({title!r}): lang is empty")
    revid = data.get("revid")
    # `bool` is a subclass of `int` in Python, so `isinstance(True, int)` is
    # true; excluded explicitly so a stray JSON `true`/`false` (a plausible
    # authoring slip next to a bare numeric revid) is rejected rather than
    # silently pinned as revision 1 or 0.
    if revid is not None and (isinstance(revid, bool) or not isinstance(revid, int)):
        raise ValueError(f"{where} ({title!r}): revid must be an int, got {revid!r}")
    return ArticleSpec(title=title, lang=lang, revid=revid)


def load_manifest(path: str | Path) -> tuple[Cluster, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_clusters = _require(data, "clusters", str(path))

    clusters = []
    for i, cluster_data in enumerate(raw_clusters):
        where = f"cluster {i}"
        name = _require(cluster_data, "name", where)
        if not name:
            # An empty cluster name isn't caught by any per-article check --
            # `note_path` would happily emit "/slug.md", a path with no
            # cluster directory at all.
            raise ValueError(f"{where}: name is empty")
        raw_articles = _require(cluster_data, "articles", f"cluster {name!r}")
        articles = tuple(
            _article_from_json(name, j, article) for j, article in enumerate(raw_articles)
        )
        clusters.append(Cluster(name=name, articles=articles))

    result = tuple(clusters)
    _check_for_collisions(result)
    return result


def save_manifest(path: str | Path, clusters: tuple[Cluster, ...]) -> None:
    """Write the manifest back out, e.g. after Task 8's build script pins a
    `revid` it fetched. Key order is fixed (title, lang, then revid only when
    pinned) and `json.dumps` is otherwise deterministic for a given input, so
    resaving an unchanged manifest reproduces the same bytes -- Task 10's
    build-reproducibility check depends on that: a rebuild with nothing
    actually different must not show a manifest diff.
    """
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
