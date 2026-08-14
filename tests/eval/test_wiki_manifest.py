import json

import pytest

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
    # just reports what it finds; `ArticleSpec.slug` is what refuses to hand
    # back an empty note stem (see below).
    assert slugify("围棋") == ""


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


def test_article_slug_raises_on_a_title_with_no_ascii_fallback():
    # An empty slug would silently become a note named ".md" (en) or "-zh.md"
    # (non-en) -- a landmine no downstream check is positioned to catch before
    # the file is actually written. Refuse at the source instead.
    article = ArticleSpec(title="围棋", lang="zh")
    with pytest.raises(ValueError, match="围棋"):
        _ = article.slug


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


def test_resaving_an_unchanged_manifest_is_byte_stable(tmp_path):
    # Task 10's build-reproducibility check asserts a clean rebuild produces
    # no git diff; if save_manifest churned formatting on every run, an
    # unrelated build would always show a spurious manifest diff.
    clusters = (
        Cluster(
            name="coffee",
            articles=(
                ArticleSpec(title="Espresso", lang="en", revid=991),
                ArticleSpec(title="Latte", lang="en"),
            ),
        ),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_manifest(first, clusters)
    save_manifest(second, load_manifest(first))
    assert first.read_bytes() == second.read_bytes()


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


def test_load_manifest_raises_on_duplicate_slug_across_clusters(tmp_path):
    # Two different titles that happen to slugify identically. Silent, this
    # means one article's note overwrites -- or is shadowed by -- another's
    # when Task 8's build script writes files, discovered only much later
    # (if at all) by Task 9's corpus-wide uniqueness check.
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


def test_load_manifest_raises_on_duplicate_title_in_one_language(tmp_path):
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
    with pytest.raises(ValueError, match="Bass"):
        load_manifest(path)
