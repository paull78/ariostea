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


def test_wiki_config_carries_a_chunking_policy(tmp_path):
    from ariostea.config.schema import ChunkingCfg
    from ariostea.eval.wiki_index import wiki_config

    assert wiki_config(tmp_path, "db").chunking == ChunkingCfg()
    policy = ChunkingCfg(max_tokens=128, overlap=32, unit="model_tokens")
    assert wiki_config(tmp_path, "db", chunking=policy).chunking == policy


def test_fused_config_turns_off_only_the_reranker(tmp_path):
    # The FUSED channel is the production search with the reranker switched
    # off: what users get when reranking is disabled or its model fails to
    # load. Everything else, chunking included, must match the indexed config.
    from ariostea.config.schema import ChunkingCfg
    from ariostea.eval.wiki_index import fused_config, wiki_config

    indexed = wiki_config(tmp_path, "db", chunking=ChunkingCfg(max_tokens=160, overlap=40))
    fused = fused_config(indexed)
    assert fused.rerank.enabled is False
    assert indexed.rerank.enabled is True  # the original is untouched
    assert fused.model_dump(exclude={"rerank"}) == indexed.model_dump(exclude={"rerank"})


def test_context_rerank_config_changes_only_use_context(tmp_path):
    # Arm 3 is arm 2's index searched with the blurb-aware reranker: same
    # reranker model, same pool, same contextual config, nothing reindexed.
    from ariostea.config.schema import ContextualCfg
    from ariostea.eval.wiki_index import context_rerank_config, wiki_config

    indexed = wiki_config(tmp_path, "db", contextual=ContextualCfg(enabled=True, model="m"))
    arm3 = context_rerank_config(indexed)

    assert arm3.rerank.use_context is True
    assert indexed.rerank.use_context is False  # the original is untouched
    assert arm3.rerank.model_dump(exclude={"use_context"}) == indexed.rerank.model_dump(
        exclude={"use_context"}
    )
    assert arm3.model_dump(exclude={"rerank"}) == indexed.model_dump(exclude={"rerank"})
