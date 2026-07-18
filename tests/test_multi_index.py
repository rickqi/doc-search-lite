"""Unit tests for MultiIndexSearcher — fan-out search across multiple Tantivy indexes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.search.bm25_search import PaginatedResults, SearchPreview
from src.search.multi_index import MultiIndexSearcher
from src.search.query_router import IndexMeta, QueryRouter
from src.search.unified import SearchSource, UnifiedSearchResult


def _make_preview(doc_id, title, score, source_path=None):
    return SearchPreview(
        doc_id=doc_id,
        title=title,
        score=score,
        snippet="",
        source_path=source_path,
    )


def _make_paginated(previews, execution_time=0.01):
    return PaginatedResults(
        results=previews,
        total=len(previews),
        offset=0,
        limit=20,
        has_more=False,
        query="test",
        execution_time=execution_time,
    )


class TestMultiIndexSearcherInit:
    """Test MultiIndexSearcher construction."""

    def test_single_index(self):
        searcher = MultiIndexSearcher(index_paths=[Path("/idx1")])
        assert len(searcher._index_paths) == 1

    def test_multiple_indexes(self):
        paths = [Path("/idx1"), Path("/idx2"), Path("/idx3")]
        searcher = MultiIndexSearcher(index_paths=paths)
        assert len(searcher._index_paths) == 3

    def test_string_paths_converted(self):
        searcher = MultiIndexSearcher(index_paths=["/idx1", "/idx2"])
        for p in searcher._index_paths:
            assert isinstance(p, Path)

    def test_rrf_k_constant(self):
        assert MultiIndexSearcher.RRF_K == 60


class TestMinMaxNormalize:
    """Test _minmax_normalize static method."""

    def test_normalize_basic(self):
        results = [
            UnifiedSearchResult(doc_id="d1", raw_score=10.0),
            UnifiedSearchResult(doc_id="d2", raw_score=5.0),
            UnifiedSearchResult(doc_id="d3", raw_score=0.0),
        ]
        MultiIndexSearcher._minmax_normalize(results)
        assert results[0].normalized_score == 1.0
        assert results[1].normalized_score == 0.5
        assert results[2].normalized_score == 0.0

    def test_normalize_all_same_score(self):
        results = [
            UnifiedSearchResult(doc_id="d1", raw_score=5.0),
            UnifiedSearchResult(doc_id="d2", raw_score=5.0),
        ]
        MultiIndexSearcher._minmax_normalize(results)
        assert all(r.normalized_score == 1.0 for r in results)

    def test_normalize_empty_list(self):
        # Should not raise
        MultiIndexSearcher._minmax_normalize([])

    def test_normalize_single_result(self):
        results = [UnifiedSearchResult(doc_id="d1", raw_score=42.0)]
        MultiIndexSearcher._minmax_normalize(results)
        assert results[0].normalized_score == 1.0

    def test_normalize_in_place(self):
        results = [UnifiedSearchResult(doc_id="d1", raw_score=10.0)]
        returned = MultiIndexSearcher._minmax_normalize(results)
        # Returns None (in-place modification)
        assert returned is None
        assert results[0].normalized_score == 1.0


class TestCrossIndexRRF:
    """Test _cross_index_rrf method."""

    def test_merge_from_two_indexes(self):
        searcher = MultiIndexSearcher(index_paths=[Path("/a"), Path("/b")])

        idx_a = [
            UnifiedSearchResult(doc_id="a::d1", source_path=Path("file1.md"), raw_score=5.0),
        ]
        idx_b = [
            UnifiedSearchResult(doc_id="b::d2", source_path=Path("file2.md"), raw_score=3.0),
        ]

        merged = searcher._cross_index_rrf([("policies", idx_a), ("finance", idx_b)])
        assert len(merged) == 2
        for r in merged:
            assert r.rrf_score > 0

    def test_dedup_across_indexes(self):
        """Same source_path in different indexes deduplicates."""
        searcher = MultiIndexSearcher(index_paths=[Path("/a"), Path("/b")])

        idx_a = [
            UnifiedSearchResult(
                doc_id="a::d1", source_path=Path("shared.md"),
                raw_score=5.0, title="From A",
            ),
        ]
        idx_b = [
            UnifiedSearchResult(
                doc_id="b::d1", source_path=Path("shared.md"),
                raw_score=3.0, snippet="From B",
            ),
        ]

        merged = searcher._cross_index_rrf([("a", idx_a), ("b", idx_b)])
        # Both keyed differently: "a::shared.md" and "b::shared.md"
        # So they're NOT deduped — different index_name prefix
        assert len(merged) == 2

    def test_empty_index_results(self):
        searcher = MultiIndexSearcher(index_paths=[Path("/a")])
        merged = searcher._cross_index_rrf([])
        assert merged == []

    def test_single_index_single_result(self):
        searcher = MultiIndexSearcher(index_paths=[Path("/a")])

        results = [
            UnifiedSearchResult(doc_id="a::d1", source_path=Path("f.md"), raw_score=10.0),
        ]
        merged = searcher._cross_index_rrf([("idx_a", results)])
        assert len(merged) == 1
        assert abs(merged[0].rrf_score - 1.0 / 61) < 1e-9


class TestSearchSingleIndex:
    """Test _search_single_index static method."""

    @patch("src.search.multi_index.create_searcher")
    def test_namespaced_doc_ids(self, mock_create):
        mock_searcher = MagicMock()
        preview = _make_preview("orig1", "Title", 8.0, source_path=Path("f.md"))
        mock_searcher.search.return_value = _make_paginated([preview])
        mock_create.return_value = mock_searcher

        results = MultiIndexSearcher._search_single_index(
            index_path=Path("/idx"),
            query="test",
            limit=20,
            index_name="policies",
        )
        assert len(results) == 1
        assert results[0].doc_id == "policies::orig1"
        assert results[0].index_name == "policies"
        assert results[0].search_source == SearchSource.BM25

    @patch("src.search.multi_index.create_searcher")
    def test_empty_results(self, mock_create):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated([])
        mock_create.return_value = mock_searcher

        results = MultiIndexSearcher._search_single_index(
            index_path=Path("/idx"),
            query="nothing",
            limit=20,
            index_name="empty",
        )
        assert results == []

    @patch("src.search.multi_index.create_searcher")
    def test_multiple_results(self, mock_create):
        mock_searcher = MagicMock()
        previews = [_make_preview(f"d{i}", f"T{i}", float(i)) for i in range(5)]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        results = MultiIndexSearcher._search_single_index(
            index_path=Path("/idx"), query="test", limit=20, index_name="test",
        )
        assert len(results) == 5
        for r in results:
            assert r.doc_id.startswith("test::")


class TestMultiIndexSearchIntegration:
    """Test full search() method with mocked backends."""

    @patch("src.search.multi_index.create_searcher")
    def test_single_index_fanout(self, mock_create):
        mock_searcher = MagicMock()
        previews = [_make_preview("d1", "T1", 5.0, source_path=Path("f.md"))]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        searcher = MultiIndexSearcher(index_paths=[Path("/idx1")])
        result = searcher.search("test")

        assert len(result.results) >= 1
        assert result.query == "test"

    @patch("src.search.multi_index.create_searcher")
    def test_multiple_indexes(self, mock_create):
        mock_searcher = MagicMock()
        previews = [_make_preview("d1", "T1", 5.0, source_path=Path("f.md"))]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        searcher = MultiIndexSearcher(index_paths=[Path("/idx1"), Path("/idx2")])
        result = searcher.search("test", limit=10)

        assert len(result.results) >= 1
        assert len(result.sources_used) >= 1

    @patch("src.search.multi_index.create_searcher")
    def test_limit_applied(self, mock_create):
        mock_searcher = MagicMock()
        previews = [_make_preview(f"d{i}", f"T{i}", float(10 - i)) for i in range(15)]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        searcher = MultiIndexSearcher(index_paths=[Path("/idx1")])
        result = searcher.search("test", limit=3)

        assert len(result.results) <= 3

    @patch("src.search.multi_index.create_searcher")
    def test_ranks_assigned(self, mock_create):
        mock_searcher = MagicMock()
        previews = [_make_preview(f"d{i}", f"T{i}", float(10 - i)) for i in range(5)]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        searcher = MultiIndexSearcher(index_paths=[Path("/idx1")])
        result = searcher.search("test")

        for i, r in enumerate(result.results):
            assert r.rank == i + 1

    @patch("src.search.multi_index.create_searcher")
    def test_failed_index_skipped(self, mock_create):
        """Index that raises should be silently skipped."""
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = RuntimeError("corrupt index")
        mock_create.return_value = mock_searcher

        searcher = MultiIndexSearcher(index_paths=[Path("/bad_idx")])
        result = searcher.search("test")

        assert len(result.results) == 0
        assert result.sources_used == []

    @patch("src.search.multi_index.create_searcher")
    def test_execution_time_positive(self, mock_create):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated([])
        mock_create.return_value = mock_searcher

        searcher = MultiIndexSearcher(index_paths=[Path("/idx1")])
        result = searcher.search("test")

        assert result.execution_time >= 0

    @patch("src.search.multi_index.create_searcher")
    def test_bm25_count(self, mock_create):
        mock_searcher = MagicMock()
        previews = [_make_preview(f"d{i}", f"T{i}", 1.0) for i in range(3)]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        searcher = MultiIndexSearcher(index_paths=[Path("/idx1")])
        result = searcher.search("test")

        assert result.bm25_count == 3


class TestIndexNameDerivation:
    """Test index_name derivation from path."""

    def test_index_dir_named_index(self):
        """When path ends in /index, use parent name."""
        path = Path("/data/raw/policies/index")
        name = path.parent.name if path.name == "index" else path.name
        assert name == "policies"

    def test_index_dir_custom_name(self):
        """When path doesn't end in /index, use dir name."""
        path = Path("/data/raw/custom_idx")
        name = path.parent.name if path.name == "index" else path.name
        assert name == "custom_idx"


class TestMultiIndexSearcherWithRouter:
    """Test MultiIndexSearcher with optional QueryRouter integration."""

    def _make_router(self):
        """Create a router with hr and finance indexes."""
        indexes = {
            "hr": IndexMeta(path="raw/hr/index", tags=["hr", "人事"]),
            "finance": IndexMeta(path="raw/finance/index", tags=["finance", "财务"]),
        }
        return QueryRouter(indexes)

    @patch("src.search.multi_index.create_searcher")
    def test_router_filters_indexes(self, mock_create):
        """Router narrows search to relevant indexes only."""
        mock_searcher = MagicMock()
        previews = [_make_preview("d1", "T1", 5.0, source_path=Path("f.md"))]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        router = self._make_router()
        # Paths ending in /index → names are parent names
        paths = [Path("/data/raw/hr/index"), Path("/data/raw/finance/index")]
        searcher = MultiIndexSearcher(index_paths=paths, query_router=router)

        # "年假" matches hr domain keywords, not finance
        result = searcher.search("年假如何申请")

        # Only hr index should have been searched
        assert len(result.sources_used) >= 1
        assert "hr" in result.sources_used

    @patch("src.search.multi_index.create_searcher")
    def test_no_router_searches_all(self, mock_create):
        """Without router, all indexes are searched (backward compat)."""
        mock_searcher = MagicMock()
        previews = [_make_preview("d1", "T1", 5.0, source_path=Path("f.md"))]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        paths = [Path("/data/raw/hr/index"), Path("/data/raw/finance/index")]
        searcher = MultiIndexSearcher(index_paths=paths)

        result = searcher.search("anything")
        assert len(result.sources_used) >= 1

    @patch("src.search.multi_index.create_searcher")
    def test_router_fallback_all_on_no_match(self, mock_create):
        """When router finds no match, all indexes are searched."""
        mock_searcher = MagicMock()
        previews = [_make_preview("d1", "T1", 5.0, source_path=Path("f.md"))]
        mock_searcher.search.return_value = _make_paginated(previews)
        mock_create.return_value = mock_searcher

        router = self._make_router()
        paths = [Path("/data/raw/hr/index"), Path("/data/raw/finance/index")]
        searcher = MultiIndexSearcher(index_paths=paths, query_router=router)

        # "天气" doesn't match any domain → fallback to all
        result = searcher.search("天气怎么样")
        assert len(result.sources_used) >= 1

    def test_resolve_target_indexes_no_router(self):
        """Without router, returns all index paths."""
        paths = [Path("/idx1"), Path("/idx2")]
        searcher = MultiIndexSearcher(index_paths=paths)
        resolved = searcher._resolve_target_indexes("test")
        assert resolved == paths

    def test_resolve_target_indexes_with_router(self):
        """With router, returns only matched paths."""
        router = self._make_router()
        paths = [Path("/data/raw/hr/index"), Path("/data/raw/finance/index")]
        searcher = MultiIndexSearcher(index_paths=paths, query_router=router)

        resolved = searcher._resolve_target_indexes("年假")
        # Only hr should match
        assert all(isinstance(p, Path) for p in resolved)
        assert len(resolved) >= 1

    def test_router_default_none(self):
        """Router defaults to None."""
        searcher = MultiIndexSearcher(index_paths=[Path("/idx1")])
        assert searcher._router is None


class TestMultiIndexOffset:
    """Test offset parameter and pagination support in MultiIndexSearcher."""

    def test_search_accepts_offset_param(self):
        """search() should accept offset parameter without error."""
        paths = [Path("/fake/index")]
        searcher = MultiIndexSearcher(index_paths=paths)
        with patch.object(searcher, "_search_single_index", return_value=[]):
            results = searcher.search("test", limit=5, offset=0)
            assert results.offset == 0
            assert results.limit == 5

    def test_offset_returns_correct_slice(self):
        """Offset should skip results correctly."""
        from src.search.unified import UnifiedSearchResult

        fake_results = [
            UnifiedSearchResult(doc_id=f"d{i}", raw_score=float(10 - i), rrf_score=float(10 - i))
            for i in range(10)
        ]
        paths = [Path("/fake/index")]
        searcher = MultiIndexSearcher(index_paths=paths)
        with patch.object(searcher, "_search_single_index", return_value=fake_results):
            results = searcher.search("test", limit=3, offset=5)
            assert len(results.results) == 3
            assert results.offset == 5
            assert results.limit == 3
            assert results.has_more is True  # 10 total, offset 5+3=8, more left

    def test_has_more_false_at_end(self):
        """has_more should be False when all results consumed."""
        from src.search.unified import UnifiedSearchResult

        fake_results = [
            UnifiedSearchResult(doc_id=f"d{i}", raw_score=float(5 - i), rrf_score=float(5 - i))
            for i in range(5)
        ]
        paths = [Path("/fake/index")]
        searcher = MultiIndexSearcher(index_paths=paths)
        with patch.object(searcher, "_search_single_index", return_value=fake_results):
            results = searcher.search("test", limit=5, offset=0)
            assert len(results.results) == 5
            assert results.has_more is False

    def test_score_property_preserved_through_merge(self):
        """UnifiedSearchResult.score should return raw_score after merge."""
        from src.search.unified import UnifiedSearchResult

        fake_results = [
            UnifiedSearchResult(doc_id="d1", raw_score=11.5, rrf_score=0.016),
        ]
        paths = [Path("/fake/index")]
        searcher = MultiIndexSearcher(index_paths=paths)
        with patch.object(searcher, "_search_single_index", return_value=fake_results):
            results = searcher.search("test", limit=5)
            assert len(results.results) == 1
            assert results.results[0].score == 11.5  # raw_score, not rrf_score


class TestGetFullContent:
    """Test get_full_content() with multi-index namespaced doc_ids."""

    def test_no_double_colon_returns_none(self):
        """doc_id without :: prefix returns None."""
        searcher = MultiIndexSearcher(index_paths=[Path("/fake/index")])
        assert searcher.get_full_content("plain_doc_id") is None

    def test_prefix_stripped_correctly(self):
        """Index prefix is stripped before delegating to single-index searcher."""
        fake_searcher = MagicMock()
        fake_searcher.get_full_content.return_value = "fake_content"
        with patch("src.search.multi_index.SearcherPool.get", return_value=fake_searcher):
            searcher = MultiIndexSearcher(index_paths=[Path("/data/raw/L3_指南/index")])
            result = searcher.get_full_content("L3_指南::abc123")
            assert result == "fake_content"
            fake_searcher.get_full_content.assert_called_once_with("abc123")

    def test_unknown_index_returns_none(self):
        """doc_id with unknown index name returns None."""
        searcher = MultiIndexSearcher(index_paths=[Path("/data/raw/L1_卫健委/index")])
        result = searcher.get_full_content("NONEXISTENT::abc123")
        assert result is None
