"""The eval corpus manifest: which Wikipedia articles form which cluster, and
the exact revision each one is pinned to.

The manifest is the reproducibility contract. Note paths are *derived* from
the article title, never authored, so a file can never drift from the
article it claims to be.

`ArticleSpec` and `Cluster` validate their own fields in `__post_init__`, not
on later attribute access: Task 8's build script constructs `ArticleSpec`
directly from a fetched revision (`ArticleSpec(title=..., lang=...,
revid=fetched.revid)`), never touching `load_manifest` at all, so a check
that only ran during JSON parsing would never see that path. Validating at
construction means both entry points get the same guarantee for free, and it
lets `.slug` be a *total* function -- no property has to raise from inside a
comprehension, an f-string, or a pytest `parametrize` list, all places this
module's slug ends up.

A slug collision needs no data outside the manifest to detect, so
`load_manifest` checks for it eagerly rather than leaving it to surface
later: a manifest typo caught here costs a re-run of this module; the same
typo caught by Task 9's corpus-wide uniqueness check costs a full
fetch-and-convert build first. The reason global (not just per-cluster)
uniqueness matters isn't file overwrites -- `games/go-game.md` and
`misc/go-game.md` are different files and coexist fine. It's that
`link_targets` maps a title to a *bare* slug with no cluster prefix, Obsidian
resolves `[[wikilink]]`s vault-globally, and Task 9 compares every inbound
link against one flat slug set -- so two clusters sharing a slug makes every
`[[go-game]]` in the corpus ambiguous about which article it means.
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
_LANG = re.compile(r"[a-z]{2,3}")
_BAD_NAME_CHARS = re.compile(r"[/\\]")


def slugify(title: str) -> str:
    """ASCII, lower-case, hyphen-separated: 'Go (game)' -> 'go-game'.

    Accents fold to their base letter via NFKD decomposition plus
    combining-mark removal ('Violín' -> 'violin'), not via a hand-maintained
    transliteration table. A title with no Latin-script fallback at all (a
    CJK title, say) folds to nothing -- `slugify` just reports that;
    `ArticleSpec.__post_init__` is what refuses to construct an instance
    whose title folds to an empty stem.

    Known limitation, accepted (same spirit as `wikitext.py`'s documented
    trade-offs): NFKD only decomposes characters that *have* a canonical
    decomposition into a base letter plus combining marks. A Latin letter
    that instead has a *compatibility* ligature or digraph expansion --
    German 'ß', Danish/Norwegian 'Æ'/'Ø', Polish 'Ł', Icelandic 'Þ' -- has no
    such decomposition at all, so it is dropped outright rather than
    transliterated: 'Straße' -> 'stra-e', 'Ærø' -> 'r', 'Łódź' -> 'odz'.
    None of this corpus's titles hit that case, so the result is a mangled
    but non-empty slug, not the empty-title failure `ArticleSpec` guards
    against; a full transliteration table would fix it but isn't worth the
    weight for six clusters of en/it/es titles.
    """
    decomposed = unicodedata.normalize("NFKD", title.lower())
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_SLUG.sub("-", ascii_only).strip("-")


@dataclass(frozen=True)
class ArticleSpec:
    title: str
    lang: str
    revid: int | None = None

    def __post_init__(self) -> None:
        """Establish the invariant `.slug` relies on, once, here -- not
        wherever `.slug` happens to get called. See the module docstring for
        why construction (not access) is the right place.
        """
        if not isinstance(self.title, str):
            raise ValueError(f"article title must be a string, got {self.title!r}")
        if not slugify(self.title):
            # Covers both an empty/whitespace-only title and one made
            # entirely of characters `slugify` can't represent (see its
            # docstring) -- either way there is no usable note stem.
            raise ValueError(f"article {self.title!r} slugifies to an empty string")
        if not isinstance(self.lang, str) or not _LANG.fullmatch(self.lang):
            raise ValueError(
                f"article {self.title!r}: lang must be a lowercase 2-3 letter code, "
                f"got {self.lang!r}"
            )
        if self.revid is not None and (
            isinstance(self.revid, bool) or not isinstance(self.revid, int) or self.revid < 1
        ):
            # `bool` is a subclass of `int` in Python (`isinstance(True, int)`
            # is true), so it's excluded explicitly: a stray JSON `true`/
            # `false` next to a numeric revid is a plausible authoring slip,
            # and should be rejected rather than silently pinned as revision
            # 1 or 0. MediaWiki revision ids are always >= 1; 0 is the
            # dangerous case to let through since it's falsy -- `if revid:`
            # anywhere downstream would misread a pinned revid 0 as
            # "unpinned".
            raise ValueError(
                f"article {self.title!r}: revid must be a positive int, got {self.revid!r}"
            )

    @property
    def slug(self) -> str:
        """Note stem, derived from `title` and, outside English,
        language-suffixed so parallel titles ('Violin' / 'Violín') cannot
        collide inside a cluster. Always succeeds: `__post_init__` has
        already ruled out the only way this could fail.
        """
        base = slugify(self.title)
        return base if self.lang == "en" else f"{base}-{self.lang}"


@dataclass(frozen=True)
class Cluster:
    name: str
    articles: tuple[ArticleSpec, ...]

    def __post_init__(self) -> None:
        """`name` becomes a filesystem path segment in `note_path`. Not
        attacker-controlled (the manifest is committed and hand-authored),
        so this is typo-catching and defense in depth rather than a security
        boundary -- but a stray '/', '\\\\', or '..' in a hand-typed cluster
        name would otherwise nest or escape the note tree silently.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"cluster name must be a non-empty string, got {self.name!r}")
        if _BAD_NAME_CHARS.search(self.name) or ".." in self.name:
            raise ValueError(
                f"cluster name {self.name!r} must not contain a path separator or '..'"
            )


def note_path(cluster: str, article: ArticleSpec) -> str:
    """Where an article's note lives, relative to the corpus root:
    `<cluster>/<slug>.md`. Both `cluster` and `article.slug` are guaranteed
    safe path segments by the point this runs -- `Cluster.__post_init__`
    rejects a `cluster` name containing a path separator or '..', and
    `ArticleSpec.__post_init__` rejects a title that can't produce a
    non-empty slug -- so this function itself does no validation of its own.
    """
    return f"{cluster}/{article.slug}.md"


def link_targets(clusters: tuple[Cluster, ...], lang: str) -> dict[str, str]:
    """Normalized article title -> note slug, for one language edition. Links
    in an it.wikipedia article point at it titles, so the maps stay separate.

    The key is NFC-normalized as well as whitespace/case-normalized:
    MediaWiki serves wikitext as NFC, but a manifest title typed by hand can
    end up NFD (decomposed accents are an easy, invisible paste artifact on
    some editors/OSes). Without normalizing here, a manifest title stored as
    NFD would never match the NFC title `convert_links` (wikitext.py) looks
    up, and the link would silently flatten to plain text instead of erroring.

    Raises on a duplicate normalized title within the language rather than
    silently keeping whichever cluster happens to come last (a plain dict
    comprehension would). `load_manifest` already rejects this shape via its
    slug-uniqueness check (same title, same language, implies the same
    slug), so this only fires for `Cluster`s assembled by hand rather than
    loaded from JSON -- kept as a standalone guard since this function makes
    no assumption about how its input was built.
    """
    targets: dict[str, str] = {}
    for cluster in clusters:
        for article in cluster.articles:
            if article.lang != lang:
                continue
            key = normalize_ws(unicodedata.normalize("NFC", article.title))
            if key in targets:
                raise ValueError(
                    f"duplicate {lang} title {article.title!r}: maps to both "
                    f"{targets[key]!r} and {article.slug!r}"
                )
            targets[key] = article.slug
    return targets


def _find_slug_collisions(clusters: tuple[Cluster, ...]) -> list[str]:
    """Every slug reused anywhere in the manifest, keyed case-insensitively
    (`slug.lower()`): `ArticleSpec` already forces `lang` to be lowercase and
    `slugify` already lowercases the title, so two slugs differing only in
    case shouldn't be reachable in practice -- but macOS APFS treats
    'Foo' and 'foo' as the same file, so keying case-sensitively would miss
    a collision that's real on the filesystem this build actually writes to,
    for the cost of a `.lower()` call.

    A *title* collision (the same (lang, title) pair listed twice) is
    deliberately not checked separately here: `normalize_ws` and `slugify`
    both fold whitespace/case, and the difference between them (accent
    folding, punctuation-to-hyphen) never separates two titles that already
    compare equal under `normalize_ws` -- so any title collision within one
    language is already, unconditionally, a slug collision too. A dedicated
    title check would be dead code that never fires without this one also
    firing (confirmed by fuzzing during review).
    """
    owners: dict[str, str] = {}
    errors: list[str] = []
    for cluster in clusters:
        for article in cluster.articles:
            key = article.slug.lower()
            owner = f"{cluster.name}/{article.title!r} ({article.lang})"
            if key in owners:
                errors.append(f"slug {article.slug!r} is claimed by both {owners[key]} and {owner}")
            else:
                owners[key] = owner
    return errors


def _require(data: Any, key: str, where: str) -> Any:
    """`data[key]`, raising a message that names the exact cluster/article at
    fault instead of a bare `KeyError('lang')` -- this file is hand-authored
    (Task 7), so a missing key is an expected kind of typo, not a freak
    occurrence.

    Raises `TypeError`, not `ValueError`, when `data` itself isn't a dict (an
    `articles` entry that's a bare string, say). That distinction is what
    lets `load_manifest` tell two failure modes apart: a missing key inside
    an otherwise well-shaped entry is a content problem worth accumulating
    and continuing past; an entry with the wrong shape entirely can't be
    inspected for *any* field, so there's nothing to accumulate -- it aborts
    the load immediately instead.
    """
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected an object, got {type(data).__name__}")
    try:
        return data[key]
    except KeyError as exc:
        raise ValueError(f"{where}: missing required key {key!r}") from exc


def load_manifest(path: str | Path) -> tuple[Cluster, ...]:
    """Parse and fully validate the manifest, raising once with every
    semantic problem found (bad lang, bad cluster name, bad revid, an
    unslugifiable title, a slug collision) rather than stopping at the
    first -- this file is hand-edited, so a second typo is likely once a
    first one exists, and the accumulation is what lets an author fix both
    in one pass instead of re-running this after each fix.

    Structural failures are the exception and still raise immediately: bad
    JSON (from `json.loads`), a missing top-level `clusters` key, or a
    cluster/article entry that isn't even a JSON object. There's no
    half-parsed result to accumulate errors against in those cases -- unlike
    a well-shaped object with a bad or missing field, a bare string in an
    `articles` list has no field to name in an error message at all.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_clusters = _require(data, "clusters", str(path))

    errors: list[str] = []
    if not raw_clusters:
        errors.append(f"{path}: 'clusters' is empty")

    clusters: list[Cluster] = []
    for i, cluster_data in enumerate(raw_clusters):
        cwhere = f"cluster {i}"
        try:
            name = _require(cluster_data, "name", cwhere)
            raw_articles = _require(cluster_data, "articles", cwhere)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if not raw_articles:
            errors.append(f"cluster {name!r}: articles is empty")

        articles: list[ArticleSpec] = []
        for j, article_data in enumerate(raw_articles):
            awhere = f"cluster {name!r} article {j}"
            try:
                title = _require(article_data, "title", awhere)
                lang = _require(article_data, "lang", awhere)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            revid = article_data.get("revid")
            try:
                articles.append(ArticleSpec(title=title, lang=lang, revid=revid))
            except ValueError as exc:
                errors.append(f"{awhere}: {exc}")

        try:
            clusters.append(Cluster(name=name, articles=tuple(articles)))
        except ValueError as exc:
            errors.append(f"{cwhere}: {exc}")

    errors.extend(_find_slug_collisions(tuple(clusters)))

    if errors:
        raise ValueError(f"{path}: manifest is invalid:\n" + "\n".join(f"- {e}" for e in errors))

    return tuple(clusters)


def save_manifest(path: str | Path, clusters: tuple[Cluster, ...]) -> None:
    """Write the manifest back out, e.g. after Task 8's build script pins a
    `revid` it fetched. Key order is fixed (title, lang, then revid only when
    pinned) and `json.dumps` is otherwise deterministic for a given input, so
    resaving an unchanged manifest reproduces the same bytes -- Task 10's
    build-reproducibility check depends on that: a rebuild with nothing
    actually different must not show a manifest diff. The input to that
    guarantee doesn't have to already be canonical JSON: a hand-authored
    manifest with arbitrary key order or indentation canonicalizes on its
    first save and is then stable, which is the case that actually matters
    here (`json.dumps` being deterministic for a fixed input was never in
    question).
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
