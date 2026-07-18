"""Unit tests for unified search result models."""

from pathlib import Path

import pytest

from src.search.unified import SearchSource, UnifiedSearchResult, UnifiedSearchResults


class TestSearchSource:
    """Test SearchSource enum."""

    def test_bm25_value(self):
        assert SearchSource.BM25 == "bm25"

    def test_grep_value(self):
        assert SearchSource.GREP == "grep"

    def test_is_str_enum(self):
        assert isinstance(SearchSource.BM25, str)
        assert isinstance(SearchSource.GREP, str)

    def test_from_value(self):
        assert SearchSource("bm25") is SearchSource.BM25
        assert SearchSource("grep") is SearchSource.GREP

    def test_enum_members(self):
        members = list(SearchSource)
        assert len(members) == 2
        assert SearchSource.BM25 in members
        assert SearchSource.GREP in members


class TestUnifiedSearchResult:
    """Test UnifiedSearchResult dataclass."""

    def test_minimal_creation(self):
        r = UnifiedSearchResult(doc_id="doc1")
        assert r.doc_id == "doc1"
        assert r.source_path is None
        assert r.title == ""
        assert r.snippet == ""
        assert r.highlights == []
        assert r.raw_score == 0.0
        assert r.normalized_score == 0.0
        assert r.rrf_score == 0.0
        assert r.rank == 0
        assert r.search_source == SearchSource.BM25
        assert r.index_name == ""
        assert r.grep_matches == 0
        assert r.grep_line_matches == []
        assert r.retrieval_time == 0.0

    def test_full_creation(self):
        r = UnifiedSearchResult(
            doc_id="doc1",
            source_path=Path("/data/file.md"),
            title="Test File",
            snippet="A snippet of text",
            highlights=["test"],
            raw_score=12.5,
            normalized_score=0.85,
            rrf_score=0.031,
            rank=1,
            search_source=SearchSource.GREP,
            index_name="policies",
            grep_matches=5,
            grep_line_matches=[{"file": "a.md", "line_no": 3, "content": "match"}],
            retrieval_time=0.045,
        )
        assert r.doc_id == "doc1"
        assert r.source_path == Path("/data/file.md")
        assert r.title == "Test File"
        assert r.snippet == "A snippet of text"
        assert r.highlights == ["test"]
        assert r.raw_score == 12.5
        assert r.normalized_score == 0.85
        assert r.rrf_score == 0.031
        assert r.rank == 1
        assert r.search_source is SearchSource.GREP
        assert r.index_name == "policies"
        assert r.grep_matches == 5
        assert len(r.grep_line_matches) == 1
        assert r.retrieval_time == 0.045

    def test_source_path_str_auto_converted(self):
        """__post_init__ converts string source_path to Path."""
        r = UnifiedSearchResult(doc_id="d1", source_path="/some/path.md")
        assert isinstance(r.source_path, Path)
        assert r.source_path == Path("/some/path.md")

    def test_source_path_none_stays_none(self):
        r = UnifiedSearchResult(doc_id="d1", source_path=None)
        assert r.source_path is None

    def test_source_path_path_preserved(self):
        p = Path("relative/path.md")
        r = UnifiedSearchResult(doc_id="d1", source_path=p)
        assert r.source_path is p

    def test_default_lists_are_independent(self):
        """Two instances should not share mutable defaults."""
        r1 = UnifiedSearchResult(doc_id="d1")
        r2 = UnifiedSearchResult(doc_id="d2")
        r1.highlights.append("x")
        assert r2.highlights == []


class TestUnifiedSearchResults:
    """Test UnifiedSearchResults container."""

    def _make_result(self, doc_id: str, score: float = 1.0) -> UnifiedSearchResult:
        return UnifiedSearchResult(
            doc_id=doc_id,
            source_path=Path(f"data/{doc_id}.md"),
            raw_score=score,
            rrf_score=score,
        )

    def test_creation(self):
        results = [self._make_result("d1"), self._make_result("d2")]
        container = UnifiedSearchResults(
            results=results,
            total=2,
            query="test",
            sources_used=["bm25"],
        )
        assert len(container.results) == 2
        assert container.total == 2
        assert container.query == "test"
        assert container.sources_used == ["bm25"]
        assert container.execution_time == 0.0
        assert container.bm25_count == 0
        assert container.grep_count == 0

    def test_iteration(self):
        results = [self._make_result("d1"), self._make_result("d2")]
        container = UnifiedSearchResults(
            results=results,
            total=2,
            query="q",
            sources_used=["bm25"],
        )
        ids = [r.doc_id for r in container.results]
        assert ids == ["d1", "d2"]

    def test_slicing(self):
        results = [self._make_result(f"d{i}") for i in range(5)]
        container = UnifiedSearchResults(
            results=results,
            total=5,
            query="q",
            sources_used=["bm25"],
        )
        assert len(container.results[:2]) == 2
        assert container.results[0].doc_id == "d0"

    def test_empty_results(self):
        container = UnifiedSearchResults(
            results=[],
            total=0,
            query="nothing",
            sources_used=[],
        )
        assert len(container.results) == 0
        assert container.total == 0

    def test_counts(self):
        container = UnifiedSearchResults(
            results=[self._make_result("d1")],
            total=1,
            query="q",
            sources_used=["bm25", "grep"],
            bm25_count=1,
            grep_count=0,
        )
        assert container.bm25_count == 1
        assert container.grep_count == 0

    def test_results_sorted_by_score(self):
        """Results should be sortable by rrf_score."""
        r1 = self._make_result("low", score=0.01)
        r2 = self._make_result("high", score=0.99)
        r3 = self._make_result("mid", score=0.5)
        results = [r1, r2, r3]
        results.sort(key=lambda r: r.rrf_score, reverse=True)
        assert results[0].doc_id == "high"
        assert results[1].doc_id == "mid"
        assert results[2].doc_id == "low"

    def test_dedup_by_source_path(self):
        """Demonstrate dedup by source_path: same path → keep higher score."""
        r1 = UnifiedSearchResult(
            doc_id="d1",
            source_path=Path("same.md"),
            rrf_score=0.8,
            search_source=SearchSource.BM25,
        )
        r2 = UnifiedSearchResult(
            doc_id="d2",
            source_path=Path("same.md"),
            rrf_score=0.3,
            search_source=SearchSource.GREP,
        )
        by_path = {}
        for r in [r1, r2]:
            key = str(r.source_path)
            if key not in by_path or r.rrf_score > by_path[key].rrf_score:
                by_path[key] = r
        assert len(by_path) == 1
        assert by_path["same.md"].rrf_score == 0.8


class TestScoreProperty:
    """Test the .score compatibility property on UnifiedSearchResult."""

    def test_score_returns_raw_score_when_present(self):
        """score property returns raw_score (BM25 scale, same as SearchPreview)."""
        r = UnifiedSearchResult(doc_id="d1", raw_score=11.15)
        assert r.score == 11.15

    def test_score_falls_back_to_rrf_when_raw_zero(self):
        """When raw_score is 0, score falls back to rrf_score."""
        r = UnifiedSearchResult(doc_id="d1", raw_score=0.0, rrf_score=0.031)
        assert r.score == 0.031

    def test_score_defaults_to_zero(self):
        """Both scores zero → score is 0."""
        r = UnifiedSearchResult(doc_id="d1")
        assert r.score == 0.0


class TestUnifiedSearchResultsCompat:
    """Test PaginatedResults-compatible fields on UnifiedSearchResults."""

    def test_has_offset_limit_has_more_defaults(self):
        """UnifiedSearchResults has PaginatedResults-compatible defaults."""
        r = UnifiedSearchResults(
            results=[],
            total=0,
            query="test",
            sources_used=[],
        )
        assert r.offset == 0
        assert r.limit == 10
        assert r.has_more is False

    def test_can_set_pagination_fields(self):
        """Pagination fields can be set."""
        r = UnifiedSearchResults(
            results=[],
            total=100,
            query="test",
            sources_used=["idx1"],
            offset=20,
            limit=10,
            has_more=True,
        )
        assert r.offset == 20
        assert r.limit == 10
        assert r.has_more is True
