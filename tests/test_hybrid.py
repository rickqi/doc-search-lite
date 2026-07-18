"""Unit tests for HybridSearcher — parallel BM25 + Grep with RRF fusion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.base import ToolResult
from src.search.bm25_search import PaginatedResults, SearchPreview
from src.search.hybrid import PROFILE_WEIGHTS, HybridSearcher
from src.search.unified import SearchSource, UnifiedSearchResult
from src.storage.base import SearchResult


def _make_preview(doc_id, title, score, source_path=None, snippet="", highlights=None):
    return SearchPreview(
        doc_id=doc_id,
        title=title,
        score=score,
        snippet=snippet,
        source_path=source_path,
        highlights=highlights or [],
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


def _make_grep_tool_result(data, success=True):
    return ToolResult(
        success=success,
        data=data,
        error="" if success else "failed",
        metadata={"execution_time": 0.005},
    )


class TestHybridSearcherInit:
    """Test HybridSearcher construction."""

    def test_default_weights(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))
        assert searcher._bm25_weight == 1.0
        assert searcher._grep_weight == 0.5

    def test_custom_weights(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(
            bm25_searcher=bm25,
            grep_raw_dir=Path("/raw"),
            bm25_weight=2.0,
            grep_weight=1.5,
        )
        assert searcher._bm25_weight == 2.0
        assert searcher._grep_weight == 1.5

    def test_rrf_k_constant(self):
        assert HybridSearcher.RRF_K == 60


class TestConvertBM25:
    """Test _convert_bm25 method."""

    def test_convert_single_result(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        preview = _make_preview("d1", "Title", 10.0, source_path=Path("a.md"), snippet="hi")
        paginated = _make_paginated([preview])

        results = searcher._convert_bm25(paginated)
        assert len(results) == 1
        assert results[0].doc_id == "d1"
        assert results[0].raw_score == 10.0
        assert results[0].search_source == SearchSource.BM25
        assert results[0].source_path == Path("a.md")

    def test_convert_multiple_results(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        previews = [
            _make_preview("d1", "A", 5.0),
            _make_preview("d2", "B", 3.0),
        ]
        paginated = _make_paginated(previews)

        results = searcher._convert_bm25(paginated)
        assert len(results) == 2
        assert results[0].raw_score == 5.0
        assert results[1].raw_score == 3.0

    def test_convert_empty_results(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        paginated = _make_paginated([])
        results = searcher._convert_bm25(paginated)
        assert results == []


class TestConvertGrep:
    """Test _convert_grep method."""

    def test_convert_grep_output(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        grep_data = "dir/file1.md:10: match text here\ndir/file1.md:25: another match"
        tool_result = _make_grep_tool_result(grep_data)

        results = searcher._convert_grep(tool_result)
        assert len(results) == 1
        assert results[0].search_source == SearchSource.GREP
        assert results[0].grep_matches == 2
        assert len(results[0].grep_line_matches) == 2
        assert results[0].title == "file1"

    def test_convert_grep_multiple_files(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        grep_data = "a.md:1: text\nb.md:5: more"
        tool_result = _make_grep_tool_result(grep_data)

        results = searcher._convert_grep(tool_result)
        assert len(results) == 2

    def test_convert_grep_failed_tool(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        tool_result = _make_grep_tool_result("", success=False)
        results = searcher._convert_grep(tool_result)
        assert results == []

    def test_convert_grep_no_matches(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        tool_result = _make_grep_tool_result("No matches found.")
        results = searcher._convert_grep(tool_result)
        assert results == []

    def test_synthetic_score_increases_with_matches(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        # 1 match
        data1 = "a.md:1: text"
        r1 = searcher._convert_grep(_make_grep_tool_result(data1))

        # 10 matches
        data10 = "\n".join(f"a.md:{i}: text" for i in range(10))
        r10 = searcher._convert_grep(_make_grep_tool_result(data10))

        assert r10[0].raw_score > r1[0].raw_score


class TestRRFMerge:
    """Test _rrf_merge method."""

    def test_rrf_combines_both_sources(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        bm25_r = UnifiedSearchResult(
            doc_id="d1", source_path=Path("a.md"),
            search_source=SearchSource.BM25,
        )
        grep_r = UnifiedSearchResult(
            doc_id="d2", source_path=Path("b.md"),
            search_source=SearchSource.GREP,
        )

        merged = searcher._rrf_merge([bm25_r], [grep_r])
        assert len(merged) == 2
        # Both should have positive RRF scores
        for r in merged:
            assert r.rrf_score > 0

    def test_rrf_dedup_by_source_path(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        bm25_r = UnifiedSearchResult(
            doc_id="d1", source_path=Path("same.md"), title="Title",
            search_source=SearchSource.BM25,
        )
        grep_r = UnifiedSearchResult(
            doc_id="d2", source_path=Path("same.md"),
            search_source=SearchSource.GREP,
            grep_matches=3,
            grep_line_matches=[{"file": "same.md", "line_no": 1, "content": "x"}],
        )

        merged = searcher._rrf_merge([bm25_r], [grep_r])
        assert len(merged) == 1
        # RRF score = bm25_weight/(60+1) + grep_weight/(60+1)
        expected = 1.0 / 61 + 0.5 / 61
        assert abs(merged[0].rrf_score - expected) < 1e-9
        # Grep data should be enriched onto the BM25 entry
        assert merged[0].grep_matches == 3

    def test_rrf_higher_rank_higher_score(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        r_rank1 = UnifiedSearchResult(doc_id="d1", source_path=Path("a.md"))
        r_rank5 = UnifiedSearchResult(doc_id="d2", source_path=Path("b.md"))

        merged = searcher._rrf_merge([r_rank1, r_rank5], [])
        scores = {r.source_path: r.rrf_score for r in merged}
        assert scores[Path("a.md")] > scores[Path("b.md")]

    def test_rrf_empty_inputs(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        merged = searcher._rrf_merge([], [])
        assert merged == []

    def test_rrf_only_bm25(self):
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))

        r = UnifiedSearchResult(doc_id="d1", source_path=Path("a.md"))
        merged = searcher._rrf_merge([r], [])
        assert len(merged) == 1
        assert merged[0].rrf_score == pytest.approx(1.0 / 61)


class TestHybridSearchIntegration:
    """Test full search() method with mocked backends."""

    @patch("src.search.hybrid.HybridSearcher._run_grep")
    def test_search_both_sources(self, mock_grep):
        bm25_mock = MagicMock()
        preview = _make_preview("d1", "T", 5.0, source_path=Path("a.md"))
        bm25_mock.search.return_value = _make_paginated([preview])

        grep_data = "a.md:5: match text"
        mock_grep.return_value = _make_grep_tool_result(grep_data)

        searcher = HybridSearcher(bm25_searcher=bm25_mock, grep_raw_dir=Path("/raw"))
        result = searcher.search("test", limit=10)

        assert result.query == "test"
        assert len(result.results) >= 1
        assert result.total >= 1
        assert "bm25" in result.sources_used or "grep" in result.sources_used

    @patch("src.search.hybrid.HybridSearcher._run_grep")
    def test_search_empty_bm25(self, mock_grep):
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = _make_paginated([])

        grep_data = "b.md:3: found"
        mock_grep.return_value = _make_grep_tool_result(grep_data)

        searcher = HybridSearcher(bm25_searcher=bm25_mock, grep_raw_dir=Path("/raw"))
        result = searcher.search("test")

        assert len(result.results) >= 1
        assert result.grep_count >= 1

    @patch("src.search.hybrid.HybridSearcher._run_grep")
    def test_search_both_empty(self, mock_grep):
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = _make_paginated([])
        mock_grep.return_value = _make_grep_tool_result("No matches found.")

        searcher = HybridSearcher(bm25_searcher=bm25_mock, grep_raw_dir=Path("/raw"))
        result = searcher.search("nothing")

        assert len(result.results) == 0
        assert result.total == 0
        assert result.sources_used == []

    @patch("src.search.hybrid.HybridSearcher._run_grep")
    def test_search_limit_applied(self, mock_grep):
        bm25_mock = MagicMock()
        previews = [_make_preview(f"d{i}", f"T{i}", float(i)) for i in range(20)]
        bm25_mock.search.return_value = _make_paginated(previews)

        mock_grep.return_value = _make_grep_tool_result("No matches found.")

        searcher = HybridSearcher(bm25_searcher=bm25_mock, grep_raw_dir=Path("/raw"))
        result = searcher.search("test", limit=5)

        assert len(result.results) <= 5

    @patch("src.search.hybrid.HybridSearcher._run_grep")
    def test_search_ranks_assigned(self, mock_grep):
        bm25_mock = MagicMock()
        previews = [_make_preview(f"d{i}", f"T{i}", float(10 - i)) for i in range(3)]
        bm25_mock.search.return_value = _make_paginated(previews)
        mock_grep.return_value = _make_grep_tool_result("No matches found.")

        searcher = HybridSearcher(bm25_searcher=bm25_mock, grep_raw_dir=Path("/raw"))
        result = searcher.search("test", limit=10)

        for i, r in enumerate(result.results):
            assert r.rank == i + 1

    @patch("src.search.hybrid.HybridSearcher._run_grep")
    def test_search_execution_time_positive(self, mock_grep):
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = _make_paginated([])
        mock_grep.return_value = _make_grep_tool_result("No matches found.")

        searcher = HybridSearcher(bm25_searcher=bm25_mock, grep_raw_dir=Path("/raw"))
        result = searcher.search("test")

        assert result.execution_time >= 0


# ──────────────────────────────────────────────────────────────────
# F3-1: Search profiles and BM25 field boosting
# ──────────────────────────────────────────────────────────────────


class TestSearchProfiles:
    """Search profile weight presets and override behaviour."""

    def test_profile_weights_legal(self):
        """Legal profile: bm25=1.0, grep=0.3 (exact match preferred)."""
        bm25 = MagicMock()
        searcher = HybridSearcher(
            bm25_searcher=bm25, grep_raw_dir=Path("/raw"), profile="legal"
        )
        assert searcher._bm25_weight == 1.0
        assert searcher._grep_weight == 0.3
        assert searcher.profile == "legal"

    def test_profile_weights_technical(self):
        """Technical profile: bm25=1.0, grep=0.5."""
        bm25 = MagicMock()
        searcher = HybridSearcher(
            bm25_searcher=bm25, grep_raw_dir=Path("/raw"), profile="technical"
        )
        assert searcher._bm25_weight == 1.0
        assert searcher._grep_weight == 0.5

    def test_profile_weights_faq(self):
        """FAQ profile: bm25=0.6, grep=0.8 (fuzzy matching more important)."""
        bm25 = MagicMock()
        searcher = HybridSearcher(
            bm25_searcher=bm25, grep_raw_dir=Path("/raw"), profile="faq"
        )
        assert searcher._bm25_weight == 0.6
        assert searcher._grep_weight == 0.8

    def test_profile_weights_general_default(self):
        """Default profile is 'general' with historic weights."""
        bm25 = MagicMock()
        searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path("/raw"))
        assert searcher._bm25_weight == 1.0
        assert searcher._grep_weight == 0.5
        assert searcher.profile == "general"

    def test_explicit_weight_overrides_profile(self):
        """Explicit bm25_weight/grep_weight override profile defaults."""
        bm25 = MagicMock()
        searcher = HybridSearcher(
            bm25_searcher=bm25,
            grep_raw_dir=Path("/raw"),
            bm25_weight=3.0,
            grep_weight=2.0,
            profile="legal",  # profile says 1.0 / 0.3
        )
        assert searcher._bm25_weight == 3.0
        assert searcher._grep_weight == 2.0

    def test_explicit_bm25_only_overrides_profile(self):
        """Override only bm25_weight; grep_weight comes from profile."""
        bm25 = MagicMock()
        searcher = HybridSearcher(
            bm25_searcher=bm25,
            grep_raw_dir=Path("/raw"),
            bm25_weight=5.0,
            profile="faq",  # faq: bm25=0.6, grep=0.8
        )
        assert searcher._bm25_weight == 5.0
        assert searcher._grep_weight == 0.8  # from profile

    def test_unknown_profile_uses_general(self):
        """Unknown profile name falls back to 'general' weights."""
        bm25 = MagicMock()
        searcher = HybridSearcher(
            bm25_searcher=bm25, grep_raw_dir=Path("/raw"), profile="nonexistent"
        )
        assert searcher._bm25_weight == PROFILE_WEIGHTS["general"]["bm25"]
        assert searcher._grep_weight == PROFILE_WEIGHTS["general"]["grep"]
        assert searcher.profile == "nonexistent"

    def test_get_profile_weights_returns_copy(self):
        """get_profile_weights returns a mutable copy (no side effects)."""
        weights = HybridSearcher.get_profile_weights("legal")
        assert weights["bm25"] == 1.0
        assert weights["grep"] == 0.3
        # Mutating the copy must not affect the original
        weights["bm25"] = 999.0
        assert PROFILE_WEIGHTS["legal"]["bm25"] == 1.0

    def test_get_profile_weights_unknown_falls_back(self):
        """get_profile_weights with unknown profile returns general."""
        weights = HybridSearcher.get_profile_weights("zzz")
        assert weights == PROFILE_WEIGHTS["general"]


class TestFieldBoosting:
    """BM25 title field boosting via TantivyIndexManager._apply_title_boost."""

    def test_title_boost_no_boost_when_one(self):
        """title_boost=1.0 means no boosting applied."""
        from src.storage.index import TantivyIndexManager

        mgr = TantivyIndexManager.__new__(TantivyIndexManager)
        result = mgr._apply_title_boost("年假", 1.0)
        assert result == "年假"

    def test_title_boost_no_boost_when_below_one(self):
        """title_boost < 1.0 should not modify query."""
        from src.storage.index import TantivyIndexManager

        mgr = TantivyIndexManager.__new__(TantivyIndexManager)
        result = mgr._apply_title_boost("年假", 0.5)
        assert result == "年假"

    def test_title_boost_applies_for_positive_boost(self):
        """title_boost > 1.0 adds a title-boosted clause."""
        from src.storage.index import TantivyIndexManager

        mgr = TantivyIndexManager.__new__(TantivyIndexManager)
        result = mgr._apply_title_boost("年假", 2.0)
        assert "title:" in result
        assert "^2.0" in result
        assert "年假" in result

    def test_title_boost_empty_query_unchanged(self):
        """Empty query should not be boosted."""
        from src.storage.index import TantivyIndexManager

        mgr = TantivyIndexManager.__new__(TantivyIndexManager)
        result = mgr._apply_title_boost("", 2.0)
        assert result == ""

    def test_title_boost_whitespace_query_unchanged(self):
        """Whitespace-only query should not be boosted."""
        from src.storage.index import TantivyIndexManager

        mgr = TantivyIndexManager.__new__(TantivyIndexManager)
        result = mgr._apply_title_boost("   ", 2.0)
        assert result == "   "

    def test_title_boost_multi_term_query(self):
        """Multi-term (jieba-segmented) queries get boosted correctly."""
        from src.storage.index import TantivyIndexManager

        mgr = TantivyIndexManager.__new__(TantivyIndexManager)
        result = mgr._apply_title_boost("年假 如何 申请", 1.5)
        assert "title:" in result
        assert "^1.5" in result
        # Original terms preserved
        assert "年假" in result
        assert "如何" in result


class TestBM25SearcherTitleBoost:
    """BM25Searcher stores and passes title_boost to index.search()."""

    def test_default_title_boost_is_one(self):
        """Default title_boost is 1.0 (no boost, backward compatible)."""
        from src.storage.index import TantivyIndexManager

        idx_mgr = MagicMock(spec=TantivyIndexManager)
        idx_mgr.search.return_value = SearchResult(
            hits=[], total=0, query="", execution_time=0.0
        )
        from src.search.bm25_search import BM25Searcher

        searcher = BM25Searcher(index_manager=idx_mgr)
        assert searcher._title_boost == 1.0

    def test_custom_title_boost_stored(self):
        """Custom title_boost is stored on the searcher."""
        from src.storage.index import TantivyIndexManager

        idx_mgr = MagicMock(spec=TantivyIndexManager)
        idx_mgr.search.return_value = SearchResult(
            hits=[], total=0, query="", execution_time=0.0
        )
        from src.search.bm25_search import BM25Searcher

        searcher = BM25Searcher(index_manager=idx_mgr, title_boost=2.0)
        assert searcher._title_boost == 2.0

    def test_title_boost_passed_to_index_search(self):
        """title_boost is forwarded to TantivyIndexManager.search()."""
        from src.storage.index import TantivyIndexManager

        idx_mgr = MagicMock(spec=TantivyIndexManager)
        idx_mgr.search.return_value = SearchResult(
            hits=[], total=0, query="", execution_time=0.0
        )
        from src.search.bm25_search import BM25Searcher

        searcher = BM25Searcher(index_manager=idx_mgr, title_boost=1.5)
        searcher.search("test query")

        idx_mgr.search.assert_called_once()
        call_kwargs = idx_mgr.search.call_args
        assert call_kwargs[1].get("title_boost") == 1.5 or (
            len(call_kwargs[0]) > 3 and call_kwargs[0][0] == "test query"
        )


class TestProfileWeightsConstants:
    """Verify PROFILE_WEIGHTS dictionary structure and values."""

    def test_all_profiles_have_required_keys(self):
        """Every profile must define bm25, grep, and title_boost."""
        for name, weights in PROFILE_WEIGHTS.items():
            assert "bm25" in weights, f"Profile {name!r} missing 'bm25'"
            assert "grep" in weights, f"Profile {name!r} missing 'grep'"
            assert "title_boost" in weights, f"Profile {name!r} missing 'title_boost'"

    def test_general_profile_matches_historic_defaults(self):
        """'general' profile must match the historic bm25=1.0, grep=0.5."""
        g = PROFILE_WEIGHTS["general"]
        assert g["bm25"] == 1.0
        assert g["grep"] == 0.5

    def test_four_profiles_defined(self):
        """Exactly four profiles: legal, technical, faq, general."""
        assert set(PROFILE_WEIGHTS.keys()) == {
            "legal", "technical", "faq", "general"
        }

    def test_legal_profile_prefers_bm25(self):
        """Legal docs prefer exact BM25 matches (grep weight low)."""
        lp = PROFILE_WEIGHTS["legal"]
        assert lp["bm25"] > lp["grep"]

    def test_faq_profile_prefers_grep(self):
        """FAQ docs benefit from fuzzy grep matching (grep weight high)."""
        fp = PROFILE_WEIGHTS["faq"]
        assert fp["grep"] > fp["bm25"]

