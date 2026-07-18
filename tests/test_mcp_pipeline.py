"""Tests for src/mcp_server.py — MCP pipeline helper functions.

Tests verify:
- _grep_fallback: Grep fallback when BM25 index unavailable
- _quick_search: Fast BM25 pre-check with multi-index support
- _run_agent_with_timeout: Async agent execution with hard timeout
- _fast_agent_pipeline: Query rewriting + multi-query BM25 pipeline

Uses real Tantivy indexes for _quick_search (no mocking of search engine).
All LLM calls are mocked — no real API calls.
"""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    import src.mcp_server as mcp_mod
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

pytestmark = pytest.mark.skipif(not _HAS_MCP, reason="mcp package not installed")


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_caches():
    """Clear module-level caches before each test."""
    mcp_mod._bm25_cache.clear()
    mcp_mod._hybrid_cache.clear()
    mcp_mod._agent_cache.clear()
    mcp_mod._config_cache = None
    yield
    mcp_mod._bm25_cache.clear()
    mcp_mod._hybrid_cache.clear()
    mcp_mod._agent_cache.clear()
    mcp_mod._config_cache = None


def _make_raw_dir(tmp_path: Path, name: str = "raw") -> Path:
    """Create a raw directory for grep testing."""
    raw = tmp_path / name
    raw.mkdir()
    return raw


def _make_index(tmp_path: Path, name: str = "index") -> Path:
    """Create a real Tantivy index with test data.

    Returns the Path to the index directory.
    """
    from src.storage.index import TantivyIndexManager

    raw = tmp_path / ("raw_" + name)
    raw.mkdir()
    (raw / "doc1.md").write_text(
        "# 年假制度\n\n年假天数为5天。员工可以申请年假。", encoding="utf-8"
    )

    idx = tmp_path / name
    mgr = TantivyIndexManager(idx)
    mgr.add_document(
        doc_id="abc123",
        title="年假制度",
        content="年假天数为5天。员工可以申请年假。",
        metadata={"filename": "doc1.md", "source_path": "doc1.md", "keywords": []},
    )
    mgr.commit()
    # Release writer reference so GC can clean up the lock
    mgr._writer = None
    return idx


# ── _grep_fallback tests ──────────────────────────────────────

class TestGrepFallback:
    """Tests for _grep_fallback()."""

    def test_grep_fallback_basic(self, tmp_path):
        """Basic grep search should find matches in .md files."""
        raw = _make_raw_dir(tmp_path)
        (raw / "leave.md").write_text(
            "# Annual Leave\n\nEmployees get 15 days of annual leave.",
            encoding="utf-8",
        )

        result = mcp_mod._grep_fallback("annual leave", str(raw), limit=5)

        assert "Found" in result
        assert "0 matches" not in result

    def test_grep_fallback_multi_word_or(self, tmp_path):
        """Multi-word query without regex metacharacters → auto-convert to OR pattern."""
        raw = _make_raw_dir(tmp_path)
        (raw / "doc1.md").write_text("# 年假制度\n\n年假天数为5天。", encoding="utf-8")
        (raw / "doc2.md").write_text("# 申请流程\n\n提交申请表。", encoding="utf-8")

        # "年假 申请" has space but no regex metacharacters → becomes "年假|申请"
        result = mcp_mod._grep_fallback("年假 申请", str(raw), limit=10)

        assert "Found" in result
        assert "0 matches" not in result

    def test_grep_fallback_nonexistent_dir(self, tmp_path):
        """Nonexistent raw_dir should return error string containing 'does not exist'."""
        result = mcp_mod._grep_fallback(
            "test", str(tmp_path / "nonexistent"), limit=5
        )

        assert "does not exist" in result

    def test_grep_fallback_no_matches(self, tmp_path):
        """Existing .md files but query doesn't match → returns '0 matches' string."""
        raw = _make_raw_dir(tmp_path)
        (raw / "doc.md").write_text(
            "# Some Document\n\nHello world content here.", encoding="utf-8"
        )

        result = mcp_mod._grep_fallback(
            "nonexistent_keyword_xyz", str(raw), limit=5
        )

        assert "0 matches" in result

    def test_grep_fallback_regex_meta_not_converted(self, tmp_path):
        """Query with regex metacharacters should NOT auto-convert to OR."""
        raw = _make_raw_dir(tmp_path)
        (raw / "test.md").write_text("test123\ntest456\nother", encoding="utf-8")

        # "test.*" contains regex metacharacters → used as-is (matches "test123", "test456")
        result = mcp_mod._grep_fallback("test.*", str(raw), limit=5)

        assert "Found" in result
        assert "0 matches" not in result

    def test_grep_fallback_comma_separated_dir(self, tmp_path):
        """Comma-separated raw_dir should use the first path only."""
        raw1 = _make_raw_dir(tmp_path, "raw1")
        raw2 = _make_raw_dir(tmp_path, "raw2")
        (raw1 / "doc1.md").write_text("unique_content_alpha", encoding="utf-8")
        (raw2 / "doc2.md").write_text("unique_content_beta", encoding="utf-8")

        # Comma-separated: should only search raw1 (first)
        result = mcp_mod._grep_fallback("unique_content", f"{raw1},{raw2}", limit=5)

        assert "Found" in result
        # alpha from raw1 should be found
        assert "alpha" in result


# ── _quick_search tests ───────────────────────────────────────

class TestQuickSearch:
    """Tests for _quick_search()."""

    def test_quick_search_basic(self, tmp_path):
        """Real index should return search results for matching query."""
        idx = _make_index(tmp_path)

        results = mcp_mod._quick_search("年假", str(idx), limit=3)

        assert results is not None
        assert len(results) > 0

    def test_quick_search_empty_index_no_match(self, tmp_path):
        """Index exists but query doesn't match → returns empty list []."""
        idx = _make_index(tmp_path)

        results = mcp_mod._quick_search(
            "nonexistent_xyz_keyword", str(idx), limit=3
        )

        assert results == []

    def test_quick_search_empty_path_returns_none(self, tmp_path):
        """Empty index_path string → returns None (empty idx_list)."""
        results = mcp_mod._quick_search("test", "", limit=3)

        assert results is None

    def test_quick_search_comma_only_path_returns_none(self, tmp_path):
        """Path with only commas/whitespace → returns None (empty idx_list)."""
        results = mcp_mod._quick_search("test", " , , ", limit=3)

        assert results is None

    def test_quick_search_nonexistent_path_returns_empty(self, tmp_path):
        """Nonexistent index path → inner exception caught, returns []."""
        bad_path = str(tmp_path / "no_such_index_dir")
        results = mcp_mod._quick_search("test", bad_path, limit=3)

        # Inner try/except catches, returns empty list (not None)
        assert results == []

    def test_quick_search_multi_index_one_fails(self, tmp_path):
        """Comma-separated: one index fails → skips failed, returns from working."""
        idx = _make_index(tmp_path)
        bad_path = str(tmp_path / "bad_index")

        multi = f"{idx},{bad_path}"
        results = mcp_mod._quick_search("年假", multi, limit=3)

        assert results is not None
        assert len(results) > 0  # Got results from working index

    def test_quick_search_multi_index_both_work(self, tmp_path):
        """Comma-separated: both indexes work → merges results from both."""
        idx1 = _make_index(tmp_path, "index1")
        idx2 = _make_index(tmp_path, "index2")

        multi = f"{idx1},{idx2}"
        results = mcp_mod._quick_search("年假", multi, limit=5)

        assert results is not None
        assert len(results) >= 1


# ── _run_agent_with_timeout tests ─────────────────────────────

class TestRunAgentWithTimeout:
    """Tests for _run_agent_with_timeout()."""

    def test_run_agent_success(self):
        """Agent.run() returns immediately → result returned."""
        agent = MagicMock()
        expected = MagicMock(answer="test answer", success=True)
        agent.run.return_value = expected

        result = asyncio.run(
            mcp_mod._run_agent_with_timeout(
                agent, "test query", None, timeout_seconds=10
            )
        )

        assert result is expected
        agent.run.assert_called_once_with(query="test query", skill=None)

    def test_run_agent_timeout(self):
        """Agent.run() takes too long → raises TimeoutError."""
        agent = MagicMock()

        def slow_run(**kwargs):
            time.sleep(2)

        agent.run.side_effect = slow_run

        with pytest.raises(TimeoutError, match="timed out"):
            asyncio.run(
                mcp_mod._run_agent_with_timeout(
                    agent, "test query", None, timeout_seconds=1
                )
            )

    def test_run_agent_passes_skill(self):
        """Agent.run() should receive the skill parameter."""
        agent = MagicMock()
        agent.run.return_value = "result"

        asyncio.run(
            mcp_mod._run_agent_with_timeout(
                agent, "query", "summarize", timeout_seconds=10
            )
        )

        agent.run.assert_called_once_with(query="query", skill="summarize")

    def test_run_agent_default_timeout_used(self):
        """When timeout_seconds not specified, uses _AGENT_TIMEOUT_SECONDS default."""
        agent = MagicMock()
        agent.run.return_value = "ok"

        result = asyncio.run(
            mcp_mod._run_agent_with_timeout(agent, "q", None)
        )

        assert result == "ok"


# ── _fast_agent_pipeline tests ────────────────────────────────

class TestFastAgentPipeline:
    """Tests for _fast_agent_pipeline() — error paths and early returns."""

    def setup_method(self):
        """Disable AgentMemory before each test to avoid real DB interference."""
        mcp_mod._agent_memory = MagicMock()
        mcp_mod._agent_memory.recall.return_value = None

    def test_fast_pipeline_no_results(self, tmp_path):
        """When _quick_search returns [], pipeline returns 'not found' message."""
        mock_config = SimpleNamespace()

        with patch.object(mcp_mod, "_get_config", return_value=mock_config), \
             patch.object(mcp_mod, "_quick_search", return_value=[]), \
             patch("src.agent.llm_client.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_resp = MagicMock()
            mock_resp.content = "keyword1\nkeyword2"
            mock_llm.chat.return_value = mock_resp
            MockLLM.return_value = mock_llm

            result = asyncio.run(
                mcp_mod._fast_agent_pipeline(
                    "test query", str(tmp_path / "index"), None, None
                )
            )

            assert "未找到" in result or "索引中" in result

    def test_fast_pipeline_query_rewrite_timeout(self, tmp_path):
        """When LLM query rewriting times out, pipeline continues with original query."""
        mock_config = SimpleNamespace()

        with patch.object(mcp_mod, "_get_config", return_value=mock_config), \
             patch.object(mcp_mod, "_quick_search", return_value=[]) as mock_qs, \
             patch("src.agent.llm_client.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            # Simulate timeout from the LLM call (caught by except clause)
            mock_llm.chat.side_effect = asyncio.TimeoutError()
            MockLLM.return_value = mock_llm

            result = asyncio.run(
                mcp_mod._fast_agent_pipeline(
                    "test query", str(tmp_path / "index"), None, None
                )
            )

            # Pipeline should not crash — returns not found message
            assert "未找到" in result or "索引中" in result
            # _quick_search was still called (with original query)
            assert mock_qs.called

    def test_fast_pipeline_query_rewrite_exception(self, tmp_path):
        """When LLM query rewriting raises generic exception, pipeline continues."""
        mock_config = SimpleNamespace()

        with patch.object(mcp_mod, "_get_config", return_value=mock_config), \
             patch.object(mcp_mod, "_quick_search", return_value=[]) as mock_qs, \
             patch("src.agent.llm_client.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = RuntimeError("connection failed")
            MockLLM.return_value = mock_llm

            result = asyncio.run(
                mcp_mod._fast_agent_pipeline(
                    "test query", str(tmp_path / "index"), None, None
                )
            )

            # Pipeline should not crash
            assert isinstance(result, str)
            assert mock_qs.called

    def test_fast_pipeline_legal_domain_filtering(self, tmp_path):
        """Legal-domain query triggers index filtering but pipeline still returns gracefully."""
        mock_config = SimpleNamespace()

        with patch.object(mcp_mod, "_get_config", return_value=mock_config), \
             patch.object(mcp_mod, "_quick_search", return_value=[]) as mock_qs, \
             patch("src.agent.llm_client.LLMClient") as MockLLM:
            mock_llm = MagicMock()
            mock_resp = MagicMock()
            mock_resp.content = "民法典 夫妻共同债务"
            mock_llm.chat.return_value = mock_resp
            MockLLM.return_value = mock_llm

            # Query with legal signal word
            result = asyncio.run(
                mcp_mod._fast_agent_pipeline(
                    "民法典第1064条关于夫妻共同债务",
                    str(tmp_path / "legal_kb" ) + "," + str(tmp_path / "medical_kb"),
                    None,
                    None,
                )
            )

            # No results → returns not found message
            assert isinstance(result, str)
            assert mock_qs.called


# ── _smoke_test_indexes tests ─────────────────────────────────

class TestSmokeTestIndexes:
    """Tests for _smoke_test_indexes()."""

    def test_smoke_test_empty_index(self, tmp_path, monkeypatch):
        """Empty DEFAULT_INDEX → early return, no crash."""
        monkeypatch.setattr(mcp_mod, "DEFAULT_INDEX", "")
        mcp_mod._smoke_test_indexes()  # Should not raise

    def test_smoke_test_nonexistent(self, tmp_path, monkeypatch):
        """Nonexistent index path → exception caught internally, no crash."""
        monkeypatch.setattr(
            mcp_mod, "DEFAULT_INDEX", str(tmp_path / "nonexistent_index")
        )
        monkeypatch.setattr(mcp_mod, "DEFAULT_RAW", "")
        mcp_mod._smoke_test_indexes()  # Should not raise

    def test_smoke_test_with_real_index(self, tmp_path, monkeypatch, caplog):
        """Real index with data matching smoke queries → no SMOKE TEST warnings."""
        import logging

        from src.storage.index import TantivyIndexManager

        raw = tmp_path / "raw_smoke"
        raw.mkdir()
        (raw / "doc1.md").write_text(
            "# 管理制度\n\n公司管理制度和流程规范。", encoding="utf-8"
        )
        idx = tmp_path / "index_smoke"
        mgr = TantivyIndexManager(idx)
        mgr.add_document(
            doc_id="smoke1",
            title="管理制度",
            content="公司管理制度和流程规范。",
            metadata={
                "filename": "doc1.md",
                "source_path": "doc1.md",
                "keywords": [],
            },
        )
        mgr.commit()
        mgr._writer = None  # Release writer lock

        monkeypatch.setattr(mcp_mod, "DEFAULT_INDEX", str(idx))
        monkeypatch.setattr(mcp_mod, "DEFAULT_RAW", "")

        with caplog.at_level(logging.WARNING, logger="src.mcp_server"):
            mcp_mod._smoke_test_indexes()

        # No SMOKE TEST failure warnings should be logged
        smoke_warnings = [
            r for r in caplog.records if "SMOKE TEST" in r.message
        ]
        assert len(smoke_warnings) == 0


# ── _get_agent tests ──────────────────────────────────────────

class TestGetAgent:
    """Tests for _get_agent()."""

    @patch("src.agent.search_agent.create_search_agent")
    def test_get_agent_caches(self, mock_create, tmp_path):
        """Same index_path → returns cached agent instance."""
        mock_create.return_value = MagicMock()
        mock_config = SimpleNamespace()
        with patch.object(mcp_mod, "_get_config", return_value=mock_config):
            a1 = mcp_mod._get_agent(str(tmp_path / "idx1"), None, False)
            a2 = mcp_mod._get_agent(str(tmp_path / "idx1"), None, False)
            assert a1 is a2  # Cached
        mock_create.assert_called_once()

    @patch("src.agent.search_agent.create_search_agent")
    def test_get_agent_different_index(self, mock_create, tmp_path):
        """Different index paths → different agent instances."""
        agent1 = MagicMock()
        agent2 = MagicMock()
        mock_create.side_effect = [agent1, agent2]
        mock_config = SimpleNamespace()
        with patch.object(mcp_mod, "_get_config", return_value=mock_config):
            a1 = mcp_mod._get_agent(str(tmp_path / "idx1"), None, False)
            a2 = mcp_mod._get_agent(str(tmp_path / "idx2"), None, False)
            assert a1 is agent1
            assert a2 is agent2

    @patch("src.agent.search_agent.create_search_agent")
    def test_get_agent_no_log(self, mock_create, tmp_path):
        """Agent should have _no_log attribute set to True."""
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent
        mock_config = SimpleNamespace()
        with patch.object(mcp_mod, "_get_config", return_value=mock_config):
            agent = mcp_mod._get_agent(str(tmp_path / "idx1"), None, False)
            assert getattr(agent, "_no_log", None) is True


# ── _format_search_results_from_list tests ────────────────────

class TestFormatSearchResultsFromList:
    """Tests for _format_search_results_from_list()."""

    def test_format_from_list_with_items(self):
        """List of mock results → formatted string with titles and scores."""
        r1 = SimpleNamespace(
            title="Annual Leave Policy",
            score=0.95,
            doc_id="abc123",
            snippet="Employees get 15 days of annual leave per year.",
            source_path="hr/leave.md",
        )
        r2 = SimpleNamespace(
            title="Sick Leave",
            score=0.72,
            doc_id="def456",
            snippet="Sick leave requires a medical certificate.",
            source_path="hr/sick.md",
        )

        result = mcp_mod._format_search_results_from_list([r1, r2], 10)

        assert "Found 2 results" in result
        assert "Annual Leave Policy" in result
        assert "Sick Leave" in result
        assert "abc123" in result
        assert "0.950" in result
        assert "hr/leave.md" in result

    def test_format_from_list_empty(self):
        """Empty list → 'Found 0 results' message."""
        result = mcp_mod._format_search_results_from_list([], 10)

        assert "Found 0 results" in result


# ── _write_status tests ───────────────────────────────────────

class TestWriteStatus:
    """Tests for _write_status()."""

    def test_write_status_basic(self, tmp_path, monkeypatch):
        """Write status with state → JSON file with expected fields."""
        import json

        status_file = tmp_path / "status.json"
        monkeypatch.setattr(mcp_mod, "_MCP_STATUS_FILE", status_file)
        monkeypatch.setattr(mcp_mod, "DEFAULT_INDEX", "")

        mcp_mod._write_status("running")

        assert status_file.exists()
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["state"] == "running"
        assert "pid" in data
        assert "timestamp" in data
        assert data["indexes"] == 0

    def test_write_status_with_extra(self, tmp_path, monkeypatch):
        """Write status with extra dict → extra fields merged into JSON."""
        import json

        status_file = tmp_path / "status.json"
        monkeypatch.setattr(mcp_mod, "_MCP_STATUS_FILE", status_file)
        monkeypatch.setattr(mcp_mod, "DEFAULT_INDEX", "")

        mcp_mod._write_status(
            "ready", extra={"version": "1.0", "tool": "doc_search"}
        )

        assert status_file.exists()
        data = json.loads(status_file.read_text(encoding="utf-8"))
        assert data["state"] == "ready"
        assert data["version"] == "1.0"
        assert data["tool"] == "doc_search"
