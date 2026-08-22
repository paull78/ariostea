import json
from pathlib import Path

import pytest

from ariostea.eval.normalize import normalize_ws
from ariostea.eval.wiki_manifest import (
    ArticleSpec,
    Cluster,
    link_targets,
    load_manifest,
    note_path,
    save_manifest,
    slugify,
)

MANIFEST = Path(__file__).resolve().parents[2] / "eval" / "wiki" / "clusters.json"


def test_slugify_is_ascii_lowercase_and_hyphenated():
    assert slugify("Double bass") == "double-bass"
    assert slugify("Go (game)") == "go-game"
    assert slugify("Violín") == "violin"


def test_slugify_folds_accents_from_the_corpus_titles():
    # Real titles this corpus will actually contain (see the plan's article
    # list): accented Latin letters fold to their base letter via NFKD.
    assert slugify("Coffea arabica") == "coffea-arabica"
    assert slugify("Parmigiano Reggiano") == "parmigiano-reggiano"
    assert slugify("Caffè espresso") == "caffe-espresso"
    assert slugify("Guitarra clásica") == "guitarra-clasica"
    assert slugify("Mast (sailing)") == "mast-sailing"


def test_slugify_of_a_title_with_no_latin_fallback_is_empty():
    # Nothing to fold to: CJK ideographs have no NFKD decomposition to a base
    # Latin letter, so the whole title is non-slug characters. `slugify` itself
    # just reports what it finds; `ArticleSpec.__post_init__` is what refuses
    # to construct an instance whose title folds to an empty stem.
    assert slugify("围棋") == ""


def test_slugify_drops_letters_with_no_canonical_decomposition():
    # Documented, accepted limitation: NFKD only decomposes accent+base-letter
    # pairs. A ligature/digraph letter (German 'ß', Danish 'Æ') has no such
    # decomposition and is dropped outright rather than transliterated.
    assert slugify("Straße") == "stra-e"
    assert slugify("Ærø") == "r"


def test_slug_is_language_qualified_outside_english():
    assert ArticleSpec(title="Violin", lang="en").slug == "violin"
    assert ArticleSpec(title="Violín", lang="es").slug == "violin-es"


def test_cappuccino_in_two_languages_does_not_collide():
    # The coffee cluster carries "Cappuccino" in both en and it; the language
    # suffix is exactly what's supposed to keep these apart.
    en = ArticleSpec(title="Cappuccino", lang="en")
    it = ArticleSpec(title="Cappuccino", lang="it")
    assert en.slug == "cappuccino"
    assert it.slug == "cappuccino-it"
    assert en.slug != it.slug


def test_note_path_puts_the_slug_under_its_cluster():
    assert note_path("string-instruments", ArticleSpec(title="Cello", lang="en")) == (
        "string-instruments/cello.md"
    )


def test_note_path_keeps_the_language_suffix_for_a_non_english_article():
    assert note_path("coffee", ArticleSpec(title="Caffè espresso", lang="it")) == (
        "coffee/caffe-espresso-it.md"
    )


# --- ArticleSpec construction validates, not `.slug` access -----------------


def test_article_spec_rejects_a_title_with_no_ascii_fallback():
    # An empty slug would silently become a note named ".md" (en) or "-zh.md"
    # (non-en) -- refuse to even construct the object, rather than deferring
    # the failure to whenever `.slug` next happens to be read.
    with pytest.raises(ValueError, match="围棋"):
        ArticleSpec(title="围棋", lang="zh")


def test_article_spec_rejects_a_non_string_title():
    # A null title in JSON (`{"title": null}`) would otherwise reach
    # `slugify` and crash five frames deep with `AttributeError: 'NoneType'
    # object has no attribute 'lower'`. Reject the type up front instead.
    with pytest.raises(ValueError, match="title"):
        ArticleSpec(title=None, lang="en")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "lang",
    ["EN", "klingon", "", "../x", "e", "e n"],
)
def test_article_spec_rejects_a_malformed_lang(lang):
    with pytest.raises(ValueError, match="lang"):
        ArticleSpec(title="Espresso", lang=lang)


def test_article_spec_rejects_a_non_string_lang():
    with pytest.raises(ValueError, match="lang"):
        ArticleSpec(title="Espresso", lang=7)  # type: ignore[arg-type]


def test_article_spec_accepts_a_three_letter_lang_code():
    # Not every language this manifest could someday need has a 2-letter
    # ISO code; the format allows 2-3 letters for exactly that reason.
    assert ArticleSpec(title="Espresso", lang="ita").lang == "ita"


def test_article_spec_rejects_a_bool_revid():
    # `bool` is a subclass of `int` in Python; a stray JSON `true`/`false`
    # next to a numeric revid must not be silently accepted as revision 1/0.
    with pytest.raises(ValueError, match="revid"):
        ArticleSpec(title="Espresso", lang="en", revid=True)


@pytest.mark.parametrize("revid", [0, -5])
def test_article_spec_rejects_a_non_positive_revid(revid):
    # MediaWiki revision ids are always >= 1. 0 is the dangerous case: it's
    # falsy, so `if article.revid:` downstream would misread a pinned
    # revision 0 as "unpinned".
    with pytest.raises(ValueError, match="revid"):
        ArticleSpec(title="Espresso", lang="en", revid=revid)


def test_article_spec_rejects_a_non_int_revid():
    with pytest.raises(ValueError, match="revid"):
        ArticleSpec(title="Espresso", lang="en", revid="991")  # type: ignore[arg-type]


# --- Cluster construction validates its name --------------------------------


def test_cluster_rejects_an_empty_name():
    with pytest.raises(ValueError, match="name"):
        Cluster(name="", articles=())


def test_cluster_rejects_a_non_string_name():
    with pytest.raises(ValueError, match="name"):
        Cluster(name=5, articles=())  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["../escape", "coffee/sub", "coffee\\sub", "a..b"])
def test_cluster_rejects_a_name_with_a_path_separator_or_dotdot(name):
    # `name` becomes a filesystem path segment in `note_path`; Task 8 does
    # `WIKI_DIR / note_path(...)` then creates the parent directories, so an
    # unguarded name could nest or escape the note tree.
    with pytest.raises(ValueError, match="name"):
        Cluster(name=name, articles=())


# --- link_targets ------------------------------------------------------------


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


def test_link_targets_for_a_language_with_no_articles_is_empty():
    clusters = (Cluster(name="strings", articles=(ArticleSpec(title="Cello", lang="en"),)),)
    assert link_targets(clusters, "fr") == {}


def test_link_targets_matches_nfd_and_nfc_forms_of_the_same_title():
    # A manifest title typed by hand can end up NFD (decomposed accents); the
    # wikitext this maps against is NFC (what MediaWiki serves). Built from
    # explicit code points, not typed as a literal accented character: two
    # source-code string literals that look identical on screen are not a
    # reliable way to pin down which Unicode form is actually in the file's
    # bytes -- exactly the class of mistake this test exists to catch.
    nfc_title = "Viol\u00edn"  # i-acute as one precomposed code point (U+00ED)
    nfd_title = "Violi\u0301n"  # "i" + a separate combining acute (U+0301)
    assert nfc_title != nfd_title

    clusters = (Cluster(name="strings", articles=(ArticleSpec(title=nfd_title, lang="es"),)),)
    targets = link_targets(clusters, "es")
    assert normalize_ws(nfc_title) in targets


def test_link_targets_raises_on_duplicate_title_within_a_language():
    # A plain dict comprehension would silently keep whichever cluster comes
    # last, so some article's inbound links would resolve to the wrong note
    # with no signal that happened. Two clusters both naming an English
    # "Bass" article is exactly the shape of mistake a hand-authored manifest
    # can make.
    clusters = (
        Cluster(name="strings", articles=(ArticleSpec(title="Bass", lang="en"),)),
        Cluster(name="fish", articles=(ArticleSpec(title="Bass", lang="en"),)),
    )
    with pytest.raises(ValueError, match="Bass"):
        link_targets(clusters, "en")


# --- load_manifest / save_manifest round trip --------------------------------


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


def test_manifest_roundtrips_non_ascii_titles(tmp_path):
    clusters = (
        Cluster(
            name="strings",
            articles=(ArticleSpec(title="Violín", lang="es", revid=42),),
        ),
    )
    path = tmp_path / "clusters.json"
    save_manifest(path, clusters)
    assert load_manifest(path) == clusters
    # ensure_ascii=False: the title is stored as real characters, not a
    # \uXXXX escape -- easy to review/diff in a PR, which \u escapes are not.
    assert "Violín" in path.read_text(encoding="utf-8")


def test_hand_authored_manifest_canonicalizes_and_then_stays_byte_stable(tmp_path):
    # Task 10's build-reproducibility check assumes a rebuild with nothing
    # actually different produces no manifest diff. `json.dumps` being
    # deterministic for a fixed input was never in doubt; what matters is
    # that a *hand-authored* file -- arbitrary key order, arbitrary indent,
    # no relation to save_manifest's own output shape -- canonicalizes on
    # its first save, and the result of *that* is what stays stable.
    raw = tmp_path / "raw.json"
    raw.write_text(
        "{\n"
        '        "clusters": [\n'
        "                {\n"
        '                        "articles": [\n'
        '                                {"revid": 991, "lang": "en", "title": "Espresso"},\n'
        '                                {"lang": "en", "title": "Latte"}\n'
        "                        ],\n"
        '                        "name": "coffee"\n'
        "                }\n"
        "        ]\n"
        "}\n",
        encoding="utf-8",
    )
    clusters = load_manifest(raw)

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_manifest(first, clusters)
    save_manifest(second, load_manifest(first))
    assert first.read_bytes() == second.read_bytes()


# --- load_manifest: structural failures raise immediately -------------------


def test_load_manifest_raises_immediately_on_missing_clusters_key(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps({"not_clusters": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="clusters"):
        load_manifest(path)


def test_load_manifest_raises_immediately_on_non_dict_cluster_entry(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps({"clusters": ["not-a-cluster"]}), encoding="utf-8")
    with pytest.raises(TypeError, match="expected an object"):
        load_manifest(path)


def test_load_manifest_raises_immediately_on_non_dict_article_entry(tmp_path):
    # A hand-authored manifest could have a stray string in the articles
    # list; there is no field to accumulate an error against, so this aborts
    # the whole load rather than being folded into the semantic error list.
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"clusters": [{"name": "coffee", "articles": ["Espresso"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="expected an object"):
        load_manifest(path)


# --- load_manifest: semantic failures accumulate -----------------------------


def test_load_manifest_raises_on_missing_required_key(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"clusters": [{"name": "coffee", "articles": [{"title": "Espresso"}]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lang"):
        load_manifest(path)


def test_load_manifest_raises_on_non_int_revid(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "name": "coffee",
                        "articles": [{"title": "Espresso", "lang": "en", "revid": "991"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revid"):
        load_manifest(path)


def test_load_manifest_raises_on_empty_lang(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {"clusters": [{"name": "coffee", "articles": [{"title": "Espresso", "lang": ""}]}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lang"):
        load_manifest(path)


def test_load_manifest_raises_on_empty_cluster_name(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"clusters": [{"name": "", "articles": [{"title": "Espresso", "lang": "en"}]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="name"):
        load_manifest(path)


def test_load_manifest_raises_on_empty_articles_list(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"clusters": [{"name": "coffee", "articles": []}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="articles"):
        load_manifest(path)


def test_load_manifest_raises_on_empty_clusters_list(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps({"clusters": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="clusters"):
        load_manifest(path)


def test_load_manifest_raises_on_duplicate_slug_across_clusters(tmp_path):
    # Two different titles that happen to slugify identically. Silent, this
    # makes every inbound `[[go-game]]` link in the corpus ambiguous about
    # which article it means, discovered only much later (if at all) by
    # Task 9's corpus-wide uniqueness check.
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "clusters": [
                    {"name": "games", "articles": [{"title": "Go (game)", "lang": "en"}]},
                    {"name": "misc", "articles": [{"title": "Go game", "lang": "en"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="go-game"):
        load_manifest(path)


def test_load_manifest_raises_on_repeated_title_within_the_same_cluster(tmp_path):
    # A copy-paste mistake inside a single cluster's article list. Renamed
    # from a duplicate-*title* framing: the same title always produces the
    # same slug, so this is really exercising slug-collision detection, not
    # a separate title check (there is no separate title check -- see
    # `_find_slug_collisions`'s docstring for why one would be dead code).
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "name": "coffee",
                        "articles": [
                            {"title": "Espresso", "lang": "en"},
                            {"title": "Espresso", "lang": "en"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="espresso"):
        load_manifest(path)


def test_load_manifest_raises_on_the_same_title_claimed_by_two_clusters(tmp_path):
    # Same rename rationale as the within-cluster case above: this exercises
    # the slug check, since "Bass" in both clusters produces the same slug.
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "clusters": [
                    {"name": "strings", "articles": [{"title": "Bass", "lang": "en"}]},
                    {"name": "fish", "articles": [{"title": "Bass", "lang": "en"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bass"):
        load_manifest(path)


def test_load_manifest_reports_multiple_independent_problems_together(tmp_path):
    # The defining behavior of accumulation: an author fixing a slug
    # collision shouldn't have to re-run this to discover the bad lang code
    # sitting right next to it.
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "name": "coffee",
                        "articles": [
                            {"title": "Espresso", "lang": "en"},
                            {"title": "Espresso", "lang": "en"},
                            {"title": "Latte", "lang": "XX"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        load_manifest(path)
    message = str(exc_info.value)
    assert "espresso" in message
    assert "XX" in message


# --- the committed manifest ---------------------------------------------------


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
