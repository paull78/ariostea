from ariostea.domain.models import Chunk, Query, RetrievedChunk, SearchResult
from ariostea.search.search_sources import SearchSources


def _rc(path, ordinal, score, text):
    chunk = Chunk(note_path=path, ordinal=ordinal, heading_path=("H",), text=text, token_count=1)
    return RetrievedChunk(chunk=chunk, score=score)


class FakeSearcher:
    """Stands in for SearchKnowledge — records the Query it was given."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_query = None

    def search(self, query):
        self.last_query = query
        return SearchResult(chunks=tuple(self._chunks))


class FakeReader:
    def note_titles(self, paths):
        return {p: p.upper() for p in paths}

    def read_note(self, path):
        return None


def test_groups_chunks_by_note_with_counts_scores_snippets():
    chunks = [
        _rc("a.md", 0, 0.9, "alpha one"),
        _rc("a.md", 1, 0.4, "alpha two"),
        _rc("b.md", 0, 0.7, "beta one"),
    ]
    uc = SearchSources(searcher=FakeSearcher(chunks), reader=FakeReader())
    hits = uc.search(Query(text="x", k=10))

    by_path = {h.note_path: h for h in hits}
    assert by_path["a.md"].hit_count == 2
    assert by_path["a.md"].best_score == 0.9  # max fused score in the note
    assert by_path["a.md"].title == "A.MD"  # from reader.note_titles
    assert "alpha one" in by_path["a.md"].snippets
    assert by_path["b.md"].hit_count == 1


def test_sources_sorted_by_best_score_and_truncated_to_k():
    chunks = [
        _rc("a.md", 0, 0.3, "a"),
        _rc("b.md", 0, 0.9, "b"),
        _rc("c.md", 0, 0.6, "c"),
    ]
    uc = SearchSources(searcher=FakeSearcher(chunks), reader=FakeReader())
    hits = uc.search(Query(text="x", k=2))
    assert [h.note_path for h in hits] == ["b.md", "c.md"]  # best first, top 2


def test_retrieves_wide_pool_not_just_query_k():
    uc = SearchSources(searcher=(fs := FakeSearcher([])), reader=FakeReader(), pool=50)
    uc.search(Query(text="x", k=3))
    assert fs.last_query.k == 50  # broad chunk pool, independent of returned-note count
    assert fs.last_query.text == "x"
