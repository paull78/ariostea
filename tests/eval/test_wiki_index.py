from pathlib import Path

import pytest

from ariostea.eval.wiki_index import CHUNK_POOL, MULTILINGUAL_MODEL, wiki_config

CORPUS = Path(__file__).resolve().parents[2] / "eval" / "wiki"


def test_wiki_config_points_at_the_corpus_and_the_given_database(tmp_path):
    db = str(tmp_path / "eval.db")
    config = wiki_config(CORPUS, db)
    assert config.vault.path == str(CORPUS)
    assert config.store.path == db


def test_wiki_config_ignores_nothing_so_every_note_is_indexed():
    # The default vault ignore list skips `.obsidian/`; the eval corpus has no
    # such directory and every file in it is a note under test.
    assert wiki_config(CORPUS, "x.db").vault.ignore == []


def test_wiki_config_uses_the_multilingual_embedding_model():
    # An English-only model would score the it/es notes near zero and make the
    # cross_lingual track measure the model choice rather than the pipeline.
    assert wiki_config(CORPUS, "x.db").embedding.local_model == MULTILINGUAL_MODEL


def test_wiki_config_disables_contextualization_by_default():
    # The baseline index must not silently depend on a running chat endpoint.
    assert wiki_config(CORPUS, "x.db").contextual.enabled is False


def test_wiki_config_passes_a_contextual_setting_through():
    from ariostea.config.schema import ContextualCfg

    config = wiki_config(CORPUS, "x.db", ContextualCfg(enabled=True, model="m"))
    assert config.contextual.enabled is True and config.contextual.model == "m"


@pytest.mark.integration
def test_index_and_channels_retrieve_from_the_real_corpus(tmp_path):
    from ariostea.eval.wiki_index import index_wiki_corpus, wiki_channels

    db = str(tmp_path / "eval.db")
    container = index_wiki_corpus(CORPUS, db)
    assert container.admin.stats().notes == 79

    channels = wiki_channels(db, container)
    assert set(channels) == {"DENSE", "SPARSE", "HYBRID"}
    for name, search_fn in channels.items():
        hits = search_fn("how is a violin tuned", CHUNK_POOL)
        assert hits, name
        note, text = hits[0]
        assert note.endswith(".md") and text
