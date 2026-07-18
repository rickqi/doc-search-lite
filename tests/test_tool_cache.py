"""Tests for ToolCache TTL/LRU cache and its integration with GrepTool/SearchTool."""

import time
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.base import ToolCache, ToolResult
from src.agent.tools.grep import GrepTool
from src.agent.tools.search import SearchTool

# ---------------------------------------------------------------------------
# 1. ToolCache unit tests
# ---------------------------------------------------------------------------


class TestToolCacheMakeKey:
    """Tests for ToolCache.make_key determinism and uniqueness."""

    def test_make_key_deterministic(self) -> None:
        """Same inputs always produce the same key."""
        k1 = ToolCache.make_key("grep", {"pattern": "test"})
        k2 = ToolCache.make_key("grep", {"pattern": "test"})
        assert k1 == k2

    def test_make_key_differs_for_different_tools(self) -> None:
        """Different tool names produce different keys even with same kwargs."""
        k1 = ToolCache.make_key("grep", {"pattern": "test"})
        k2 = ToolCache.make_key("search", {"pattern": "test"})
        assert k1 != k2

    def test_make_key_order_independent(self) -> None:
        """Kwargs with different insertion order produce the same key."""
        k1 = ToolCache.make_key("grep", {"a": 1, "b": 2})
        k2 = ToolCache.make_key("grep", {"b": 2, "a": 1})
        assert k1 == k2


class TestToolCachePutGet:
    """Tests for basic put / get behaviour."""

    def test_cache_miss_returns_none(self) -> None:
        cache = ToolCache()
        assert cache.get("nonexistent") is None

    def test_put_then_get_returns_result(self) -> None:
        cache = ToolCache()
        result = ToolResult.ok(data="hello")
        cache.put("k1", result)
        got = cache.get("k1")
        assert got is not None
        assert got.success is True
        assert got.data == "hello"

    def test_cache_hit_returns_same_tool_result(self) -> None:
        cache = ToolCache()
        result = ToolResult.ok(data={"count": 42}, metadata={"t": 1.0})
        cache.put("k1", result)
        got = cache.get("k1")
        assert got is result  # same object reference

    def test_clear_removes_all_entries(self) -> None:
        cache = ToolCache()
        cache.put("a", ToolResult.ok())
        cache.put("b", ToolResult.ok())
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestToolCacheTTL:
    """Tests for TTL expiry."""

    def test_expired_entry_returns_none(self) -> None:
        cache = ToolCache(ttl=0.05)
        cache.put("k", ToolResult.ok(data="soon-gone"))
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_non_expired_entry_still_available(self) -> None:
        cache = ToolCache(ttl=10)
        cache.put("k", ToolResult.ok(data="still-here"))
        assert cache.get("k") is not None


class TestToolCacheMaxSize:
    """Tests for LRU eviction when max_size is exceeded."""

    def test_lru_eviction_oldest_removed(self) -> None:
        cache = ToolCache(max_size=3)
        cache.put("a", ToolResult.ok(data="1"))
        cache.put("b", ToolResult.ok(data="2"))
        cache.put("c", ToolResult.ok(data="3"))
        # Adding a 4th should evict "a" (oldest)
        cache.put("d", ToolResult.ok(data="4"))
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None
        assert cache.get("d") is not None

    def test_access_promotes_entry(self) -> None:
        cache = ToolCache(max_size=3)
        cache.put("a", ToolResult.ok(data="1"))
        cache.put("b", ToolResult.ok(data="2"))
        cache.put("c", ToolResult.ok(data="3"))
        # Access "a" to promote it (move to most-recent)
        cache.get("a")
        # Adding a new entry should evict "b" (now the LRU)
        cache.put("d", ToolResult.ok(data="4"))
        assert cache.get("a") is not None  # promoted, still alive
        assert cache.get("b") is None  # evicted as LRU
        assert cache.get("c") is not None
        assert cache.get("d") is not None


# ---------------------------------------------------------------------------
# 2. Integration tests with GrepTool (mocked filesystem)
# ---------------------------------------------------------------------------


class TestGrepToolCacheIntegration:
    """Test that GrepTool uses cache when set_cache() is called."""

    def test_grep_cache_hit_returns_cached_result(self, tmp_path: Path) -> None:
        # Create a file to search
        (tmp_path / "doc.md").write_text("hello world\nfoo bar\n", encoding="utf-8")
        tool = GrepTool(raw_dir=tmp_path)
        cache = ToolCache()
        tool.set_cache(cache)

        # First call: cache miss, does actual work
        r1 = tool.execute(pattern="hello")
        assert r1.success
        assert "cache_hit" not in r1.metadata

        # Second call with same kwargs: cache hit
        r2 = tool.execute(pattern="hello")
        assert r2.success
        assert r2.metadata.get("cache_hit") is True
        # Same data
        assert r2.data == r1.data

    def test_grep_no_cache_works_as_before(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("hello world\n", encoding="utf-8")
        tool = GrepTool(raw_dir=tmp_path)
        # No set_cache() called — should work normally
        r = tool.execute(pattern="hello")
        assert r.success
        assert "cache_hit" not in r.metadata


# ---------------------------------------------------------------------------
# 3. Integration tests with SearchTool (mocked BM25Searcher)
# ---------------------------------------------------------------------------


class TestSearchToolCacheIntegration:
    """Test that SearchTool uses cache when set_cache() is called."""

    def _make_mock_searcher(self) -> MagicMock:
        from src.search.bm25_search import PaginatedResults, SearchPreview

        preview = SearchPreview(
            doc_id="d1",
            title="Test",
            score=1.0,
            snippet="hello",
            source_path=None,
            highlights=[],
        )
        paginated = PaginatedResults(
            results=[preview],
            total=1,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.01,
        )
        searcher = MagicMock()
        searcher.search.return_value = paginated
        return searcher

    def test_search_cache_hit_returns_cached_result(self) -> None:
        searcher = self._make_mock_searcher()
        tool = SearchTool(searcher=searcher)
        cache = ToolCache()
        tool.set_cache(cache)

        r1 = tool.execute(query="test")
        assert r1.success
        assert "cache_hit" not in r1.metadata

        r2 = tool.execute(query="test")
        assert r2.success
        assert r2.metadata.get("cache_hit") is True

        # BM25Searcher.search should have been called only once
        assert searcher.search.call_count == 1

    def test_search_no_cache_works_as_before(self) -> None:
        searcher = self._make_mock_searcher()
        tool = SearchTool(searcher=searcher)
        r = tool.execute(query="test")
        assert r.success
        assert "cache_hit" not in r.metadata
        assert searcher.search.call_count == 1
