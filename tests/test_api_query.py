"""
Tests for query endpoints: POST /query (BM25), POST /query/agent, POST /api/analyze.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import _searchers, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_api_searchers():
    """Clear the module-level _searchers cache to prevent cross-test pollution."""
    _searchers.clear()
    yield
    _searchers.clear()


# ── Helpers ───────────────────────────────────────────────────


def _build_index(raw_dir: Path) -> str:
    """Create .md files under *raw_dir*, build a Tantivy index, return index path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    r = client.post("/api/admin/build-index", params={"raw_dir": str(raw_dir)})
    assert r.status_code == 200, f"build-index failed: {r.text}"
    return str(raw_dir / "index")


# ── POST /query (BM25) ────────────────────────────────────────


class TestQueryBM25:
    """Tests for POST /query — BM25 keyword search."""

    def test_query_search_basic(self, tmp_path):
        """Basic BM25 search returns results from a real index."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "annual_leave.md").write_text(
            "# 年假制度\n\n员工年假申请流程和审批要求。\n", encoding="utf-8"
        )
        (raw / "travel.md").write_text(
            "# 差旅报销\n\n出差费用报销标准。\n", encoding="utf-8"
        )
        idx = _build_index(raw)

        r = client.post("/query", json={
            "query": "年假",
            "index_path": idx,
            "limit": 5,
        })
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert data["query"] == "年假"
        assert data["total"] >= 1
        assert len(data["results"]) >= 1

    def test_query_response_structure(self, tmp_path):
        """Response contains all required QueryResponse fields."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc.md").write_text("# Test\n\nSearchable content here.\n", encoding="utf-8")
        idx = _build_index(raw)

        r = client.post("/query", json={
            "query": "Searchable",
            "index_path": idx,
        })
        assert r.status_code == 200
        data = r.json()
        for field in ("results", "total", "limit", "has_more", "query", "execution_time"):
            assert field in data, f"Missing field: {field}"
        if data["results"]:
            first = data["results"][0]
            for field in ("doc_id", "title", "score", "snippet"):
                assert field in first, f"Missing result field: {field}"

    def test_query_no_matches(self, tmp_path):
        """Query with no matches returns total=0 and empty results."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc.md").write_text("# Travel Policy\n\nBusiness travel.\n", encoding="utf-8")
        idx = _build_index(raw)

        r = client.post("/query", json={
            "query": "不存在的关键词xyz123",
            "index_path": idx,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_query_nonexistent_index(self):
        """Non-existent index path returns 404."""
        r = client.post("/query", json={
            "query": "test",
            "index_path": "/nonexistent/index/path_xyz",
        })
        assert r.status_code == 404

    def test_query_with_limit(self, tmp_path):
        """Limit parameter caps the number of returned results."""
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(5):
            (raw / f"doc{i}.md").write_text(
                f"# Document {i}\n\n年假 policy content variant {i}.\n", encoding="utf-8"
            )
        idx = _build_index(raw)

        r = client.post("/query", json={
            "query": "年假",
            "index_path": idx,
            "limit": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["limit"] == 2
        assert len(data["results"]) <= 2

    def test_query_default_limit(self, tmp_path):
        """Default limit is 10."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "only.md").write_text("# Only Doc\n\nunique content.\n", encoding="utf-8")
        idx = _build_index(raw)

        r = client.post("/query", json={
            "query": "unique",
            "index_path": idx,
        })
        assert r.status_code == 200
        assert r.json()["limit"] == 10


# ── POST /query/agent ─────────────────────────────────────────


class TestQueryAgent:
    """Tests for POST /query/agent — AI agent search."""

    @patch("src.agent.search_agent.create_search_agent")
    def test_agent_search_success(self, mock_create, tmp_path):
        """Agent search returns answer and execution_mode when mocked."""
        idx = tmp_path / "index"
        idx.mkdir()

        mock_agent = MagicMock()
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {
            "success": True,
            "answer": "年假可以按以下流程申请...",
            "sources": ["annual_leave.md"],
            "confidence": 0.85,
        }
        mock_resp.sources = ["annual_leave.md"]
        mock_agent.run.return_value = mock_resp
        mock_create.return_value = mock_agent

        r = client.post("/query/agent", json={
            "query": "年假如何申请",
            "index_path": str(idx),
        })
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert data["success"] is True
        assert "answer" in data
        assert data["execution_mode"] == "search_agent"
        mock_agent.run.assert_called_once()

    def test_agent_nonexistent_index(self):
        """Agent with non-existent index returns 404."""
        r = client.post("/query/agent", json={
            "query": "test query",
            "index_path": "/nonexistent/path/abc_xyz",
        })
        assert r.status_code == 404

    @patch("src.agent.search_agent.create_search_agent")
    def test_agent_no_log(self, mock_create, tmp_path):
        """Agent with log=False sets _no_log=True on agent instance."""
        idx = tmp_path / "index"
        idx.mkdir()

        mock_agent = MagicMock()
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {"success": True, "answer": "test", "sources": []}
        mock_resp.sources = []
        mock_agent.run.return_value = mock_resp
        mock_create.return_value = mock_agent

        r = client.post("/query/agent", json={
            "query": "test query",
            "index_path": str(idx),
            "log": False,
        })
        assert r.status_code == 200
        assert mock_agent._no_log is True

    @patch("src.agent.search_agent.create_search_agent")
    def test_agent_with_rerank(self, mock_create, tmp_path):
        """Agent with use_rerank=True passes flag through to create_search_agent."""
        idx = tmp_path / "index"
        idx.mkdir()

        mock_agent = MagicMock()
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {"success": True, "answer": "ok", "sources": []}
        mock_resp.sources = []
        mock_agent.run.return_value = mock_resp
        mock_create.return_value = mock_agent

        r = client.post("/query/agent", json={
            "query": "差旅标准",
            "index_path": str(idx),
            "use_rerank": True,
        })
        assert r.status_code == 200
        _, kwargs = mock_create.call_args
        assert kwargs.get("use_rerank") is True

    @patch("src.agent.search_agent.create_search_agent")
    def test_agent_multi_index_one_exists(self, mock_create, tmp_path):
        """Comma-separated index paths: at least one existing → no 404."""
        real_idx = tmp_path / "real_index"
        real_idx.mkdir()

        mock_agent = MagicMock()
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {"success": True, "answer": "ok", "sources": []}
        mock_resp.sources = []
        mock_agent.run.return_value = mock_resp
        mock_create.return_value = mock_agent

        r = client.post("/query/agent", json={
            "query": "test",
            "index_path": f"/nonexistent/xyz,{real_idx}",
        })
        assert r.status_code == 200

    @patch("src.agent.search_agent.create_search_agent")
    def test_agent_log_generates_srch_id(self, mock_create, tmp_path):
        """Agent with log=True generates a non-empty srch_session_id on agent."""
        idx = tmp_path / "index"
        idx.mkdir()

        mock_agent = MagicMock()
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {"success": True, "answer": "ok", "sources": []}
        mock_resp.sources = []
        mock_agent.run.return_value = mock_resp
        mock_create.return_value = mock_agent

        r = client.post("/query/agent", json={
            "query": "test",
            "index_path": str(idx),
            "log": True,
        })
        assert r.status_code == 200
        assert mock_agent._srch_session_id  # truthy = non-empty string


# ── POST /api/analyze ─────────────────────────────────────────


class TestAnalyze:
    """Tests for POST /api/analyze — document deep analysis."""

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_summarize(self, mock_func, tmp_path):
        """Summarize mode returns structured response."""
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {
            "success": True,
            "answer": "Document summary...",
            "mode": "summarize",
        }
        mock_func.return_value = mock_resp

        r = client.post("/api/analyze", json={
            "query": "总结文档",
            "index_path": str(tmp_path / "index"),
            "mode": "summarize",
        })
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert data["success"] is True
        assert "answer" in data
        mock_func.assert_called_once()

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_extract(self, mock_func, tmp_path):
        """Extract mode returns structured response."""
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {
            "success": True,
            "answer": "Extracted data...",
            "mode": "extract",
        }
        mock_func.return_value = mock_resp

        r = client.post("/api/analyze", json={
            "query": "提取关键信息",
            "index_path": str(tmp_path / "index"),
            "mode": "extract",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mock_func.assert_called_once()

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_compare(self, mock_func, tmp_path):
        """Compare mode (auto-search, no doc_ids) returns comparison results."""
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {
            "success": True,
            "answer": "Comparison results...",
            "mode": "compare",
        }
        mock_func.return_value = mock_resp

        r = client.post("/api/analyze", json={
            "query": "对比差旅标准",
            "index_path": str(tmp_path / "index"),
            "mode": "compare",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mock_func.assert_called_once()

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_table(self, mock_func, tmp_path):
        """Table mode returns table data."""
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {
            "success": True,
            "answer": "Table data...",
            "mode": "table",
        }
        mock_func.return_value = mock_resp

        r = client.post("/api/analyze", json={
            "query": "提取表格数据",
            "index_path": str(tmp_path / "index"),
            "mode": "table",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        mock_func.assert_called_once()

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_error(self, mock_func):
        """Analyze with HTTPException returns proper error response."""
        from fastapi import HTTPException

        mock_func.side_effect = HTTPException(status_code=500, detail="Index not found")

        r = client.post("/api/analyze", json={
            "query": "test",
            "index_path": "/nonexistent",
            "mode": "summarize",
        })
        assert r.status_code == 500
        assert "Index not found" in r.json()["detail"]

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_unhandled_error_propagates(self, mock_func):
        """Analyze with non-HTTP exception propagates (not silently swallowed)."""
        mock_func.side_effect = RuntimeError("Unexpected internal error")

        with pytest.raises(RuntimeError, match="Unexpected internal error"):
            client.post("/api/analyze", json={
                "query": "test",
                "index_path": "/nonexistent",
                "mode": "summarize",
            })

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_no_log(self, mock_func, tmp_path):
        """Analyze with log=False still returns 200."""
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {
            "success": True,
            "answer": "Summary",
            "mode": "summarize",
        }
        mock_func.return_value = mock_resp

        r = client.post("/api/analyze", json={
            "query": "test",
            "index_path": str(tmp_path / "index"),
            "mode": "summarize",
            "log": False,
        })
        assert r.status_code == 200
        assert r.json()["success"] is True

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_passes_mode_and_top_k(self, mock_func, tmp_path):
        """Analyze passes correct mode and top_k=3 to search_and_analyze."""
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {"success": True, "answer": "ok", "mode": "extract"}
        mock_func.return_value = mock_resp

        client.post("/api/analyze", json={
            "query": "提取信息",
            "index_path": str(tmp_path / "index"),
            "mode": "extract",
        })
        _, kwargs = mock_func.call_args
        assert kwargs.get("mode") == "extract"
        assert kwargs.get("top_k") == 3

    @patch("src.agent.analysis_agent.search_and_analyze")
    def test_analyze_passes_aspect(self, mock_func, tmp_path):
        """Analyze passes aspect parameter for compare mode."""
        mock_resp = MagicMock()
        mock_resp.to_dict.return_value = {"success": True, "answer": "ok", "mode": "compare"}
        mock_func.return_value = mock_resp

        client.post("/api/analyze", json={
            "query": "对比方案",
            "index_path": str(tmp_path / "index"),
            "mode": "compare",
            "aspect": "价格",
        })
        _, kwargs = mock_func.call_args
        assert kwargs.get("aspect") == "价格"
