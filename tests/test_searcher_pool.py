"""Unit tests for SearcherPool — BM25Searcher cache with TTL and LRU eviction."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.search.multi_index import SearcherPool


def _fake_searcher(label: str = "default") -> MagicMock:
    """Create a mock BM25Searcher with a label for identity checking."""
    s = MagicMock()
    s._label = label
    return s


class TestSearcherPoolCaching:
    """Test that same path returns same cached instance."""

    @patch("src.search.multi_index.create_searcher")
    def test_same_path_returns_same_instance(self, mock_create):
        mock_create.return_value = _fake_searcher("s1")
        path = Path("/data/index")

        result1 = SearcherPool.get(path)
        result2 = SearcherPool.get(path)

        assert result1 is result2
        assert mock_create.call_count == 1

    @patch("src.search.multi_index.create_searcher")
    def test_different_paths_return_different_instances(self, mock_create):
        call_count = 0
        def side_effect(index_path, **kwargs):
            nonlocal call_count
            call_count += 1
            return _fake_searcher(f"s{call_count}")

        mock_create.side_effect = side_effect

        r1 = SearcherPool.get(Path("/idx_a"))
        r2 = SearcherPool.get(Path("/idx_b"))

        assert r1 is not r2
        assert r1._label == "s1"
        assert r2._label == "s2"
        assert mock_create.call_count == 2

    @patch("src.search.multi_index.create_searcher")
    def test_resolve_normalizes_path(self, mock_create):
        """Relative and absolute paths resolving to same location share cache entry."""
        mock_create.return_value = _fake_searcher("shared")
        abs_path = Path("/data/index")

        r1 = SearcherPool.get(abs_path)
        # Same resolved path should hit cache
        r2 = SearcherPool.get(abs_path)

        assert r1 is r2
        assert mock_create.call_count == 1


class TestSearcherPoolTTL:
    """Test TTL-based expiry of cached entries."""

    @patch("src.search.multi_index.create_searcher")
    def test_ttl_expiry_creates_new_instance(self, mock_create):
        mock_create.return_value = _fake_searcher("fresh")
        path = Path("/idx_ttl")

        # First call — creates entry
        r1 = SearcherPool.get(path)
        assert mock_create.call_count == 1

        # Manually age the timestamp to simulate TTL expiry
        key = str(path.resolve())
        SearcherPool._timestamps[key] = time.time() - SearcherPool._ttl - 1

        # Second call — TTL expired, should create new
        mock_create.return_value = _fake_searcher("refreshed")
        r2 = SearcherPool.get(path)

        assert r2 is not r1
        assert r2._label == "refreshed"
        assert mock_create.call_count == 2

    @patch("src.search.multi_index.create_searcher")
    def test_within_ttl_returns_cached(self, mock_create):
        mock_create.return_value = _fake_searcher("cached")
        path = Path("/idx_within")

        r1 = SearcherPool.get(path)
        # Timestamp is fresh, so should return cached
        r2 = SearcherPool.get(path)

        assert r1 is r2
        assert mock_create.call_count == 1


class TestSearcherPoolEviction:
    """Test LRU eviction when pool is at capacity."""

    @patch("src.search.multi_index.create_searcher")
    def test_lru_eviction_at_max_size(self, mock_create):
        original_max = SearcherPool._max_size
        try:
            SearcherPool._max_size = 3
            call_count = 0

            def side_effect(index_path, **kwargs):
                nonlocal call_count
                call_count += 1
                return _fake_searcher(f"evict-{call_count}")

            mock_create.side_effect = side_effect

            # Fill pool to capacity
            s1 = SearcherPool.get(Path("/e1"))
            s2 = SearcherPool.get(Path("/e2"))
            s3 = SearcherPool.get(Path("/e3"))
            assert SearcherPool.size() == 3

            # Adding a 4th should evict the oldest (/e1)
            s4 = SearcherPool.get(Path("/e4"))
            assert SearcherPool.size() == 3
            assert s4._label == "evict-4"

            # /e1 should be evicted, /e2 and /e3 should still be cached
            assert mock_create.call_count == 4  # 4 create calls

            # Re-access /e2 — should be cached (no new create call)
            s2_again = SearcherPool.get(Path("/e2"))
            assert s2_again is s2
            assert mock_create.call_count == 4  # no new call

            # /e1 was evicted — accessing it creates new
            s1_new = SearcherPool.get(Path("/e1"))
            assert s1_new is not s1
            assert mock_create.call_count == 5
        finally:
            SearcherPool._max_size = original_max


class TestSearcherPoolEvict:
    """Test targeted eviction of specific paths."""

    @patch("src.search.multi_index.create_searcher")
    def test_evict_existing_path(self, mock_create):
        mock_create.return_value = _fake_searcher("to_evict")
        path = Path("/evict_me")

        SearcherPool.get(path)
        assert SearcherPool.size() == 1

        result = SearcherPool.evict(path)
        assert result is True
        assert SearcherPool.size() == 0

    def test_evict_nonexistent_path(self):
        result = SearcherPool.evict(Path("/no_such_path"))
        assert result is False


class TestSearcherPoolClearAndSize:
    """Test clear() and size() methods."""

    @patch("src.search.multi_index.create_searcher")
    def test_clear_empties_pool(self, mock_create):
        mock_create.return_value = _fake_searcher("tmp")
        SearcherPool.get(Path("/a"))
        SearcherPool.get(Path("/b"))
        assert SearcherPool.size() == 2

        SearcherPool.clear()
        assert SearcherPool.size() == 0

    @patch("src.search.multi_index.create_searcher")
    def test_size_tracks_entries(self, mock_create):
        mock_create.return_value = _fake_searcher("s")
        assert SearcherPool.size() == 0

        SearcherPool.get(Path("/sz1"))
        assert SearcherPool.size() == 1

        SearcherPool.get(Path("/sz2"))
        assert SearcherPool.size() == 2

        SearcherPool.clear()
        assert SearcherPool.size() == 0


class TestSearcherPoolIntegration:
    """Test SearcherPool integration with MultiIndexSearcher._search_single_index."""

    @patch("src.search.multi_index.create_searcher")
    def test_search_single_index_uses_pool(self, mock_create):
        from src.search.bm25_search import PaginatedResults, SearchPreview
        from src.search.multi_index import MultiIndexSearcher
        from src.search.unified import SearchSource

        mock_searcher = MagicMock()
        preview = SearchPreview(
            doc_id="d1", title="T1", score=5.0,
            snippet="hi", source_path=Path("f.md"),
        )
        mock_searcher.search.return_value = PaginatedResults(
            results=[preview], total=1, offset=0, limit=20,
            has_more=False, query="test", execution_time=0.01,
        )
        mock_create.return_value = mock_searcher

        # First call — creates and caches
        r1 = MultiIndexSearcher._search_single_index(
            Path("/pool_int"), "test", 20, "idx",
        )
        assert len(r1) == 1
        assert mock_create.call_count == 1

        # Second call — should use cached, no new create_searcher call
        r2 = MultiIndexSearcher._search_single_index(
            Path("/pool_int"), "test2", 20, "idx",
        )
        assert len(r2) == 1
        assert mock_create.call_count == 1  # no additional call
