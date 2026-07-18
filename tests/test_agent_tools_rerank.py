"""Tests for RerankTool — Agent reranking tool wrapper.

Tests verify:
- Candidate cap (max 64 documents forwarded to reranker)
- JSON string and Python list document input
- Parameter validation (query, documents, top_n)
- Reranker success/failure handling
- Formatted output structure
"""

import json
from unittest.mock import MagicMock, patch

from src.agent.tools.rerank import RerankTool
from src.search.reranker import RerankResult


def _make_mock_reranker(available=True):
    """Create a mock reranker with configurable availability."""
    r = MagicMock()
    r.available = available
    r.rerank.return_value = [
        RerankResult(index=1, relevance_score=0.95),
        RerankResult(index=0, relevance_score=0.78),
    ]
    return r


class TestRerankToolBasics:
    """Basic RerankTool property tests."""

    def test_name(self):
        tool = RerankTool(_make_mock_reranker())
        assert tool.name == "rerank"

    def test_description_contains_keyword(self):
        tool = RerankTool(_make_mock_reranker())
        assert "重排序" in tool.description or "rerank" in tool.description.lower()


class TestRerankToolExecute:
    """RerankTool.execute() tests."""

    def test_execute_success_with_list(self):
        """Execute with Python list documents returns formatted results."""
        mock_reranker = _make_mock_reranker()
        tool = RerankTool(mock_reranker)

        result = tool.execute(
            query="年假有几天",
            documents=["年假申请流程", "年假天数规定"],
            top_n=2,
        )

        assert result.success is True
        assert isinstance(result.data, str)
        assert "0.95" in result.data or "0.78" in result.data
        assert result.metadata["total_documents"] == 2
        assert result.metadata["top_n"] == 2
        mock_reranker.rerank.assert_called_once()

    def test_execute_success_with_json_string(self):
        """Execute with JSON string documents parses correctly."""
        mock_reranker = _make_mock_reranker()
        tool = RerankTool(mock_reranker)

        docs_json = json.dumps(["doc1 text", "doc2 text"])
        result = tool.execute(
            query="test query",
            documents=docs_json,
            top_n=2,
        )

        assert result.success is True
        mock_reranker.rerank.assert_called_once_with(
            query="test query",
            documents=["doc1 text", "doc2 text"],
            top_n=2,
        )

    def test_execute_caps_64_candidates(self):
        """When >64 documents provided, only 64 are forwarded to reranker."""
        mock_reranker = _make_mock_reranker()
        # Override rerank to capture actual document count
        actual_docs = []
        def capture_rerank(query, documents, top_n):
            actual_docs.extend(documents)
            return [RerankResult(index=0, relevance_score=0.9)]
        mock_reranker.rerank.side_effect = capture_rerank

        tool = RerankTool(mock_reranker)
        docs = [f"document_{i}" for i in range(100)]

        result = tool.execute(query="test", documents=docs, top_n=1)

        assert result.success is True
        assert len(actual_docs) == 64  # Capped to 64
        assert result.metadata["total_documents"] == 100  # Original count recorded

    def test_execute_does_not_cap_under_64(self):
        """When <=64 documents, all are forwarded (no cap)."""
        mock_reranker = _make_mock_reranker()
        actual_docs = []
        def capture_rerank(query, documents, top_n):
            actual_docs.extend(documents)
            return [RerankResult(index=0, relevance_score=0.9)]
        mock_reranker.rerank.side_effect = capture_rerank

        tool = RerankTool(mock_reranker)
        docs = [f"doc_{i}" for i in range(10)]

        result = tool.execute(query="test", documents=docs, top_n=1)

        assert result.success is True
        assert len(actual_docs) == 10  # Not capped

    def test_execute_empty_query_fails(self):
        """Empty query returns failure."""
        tool = RerankTool(_make_mock_reranker())
        result = tool.execute(query="", documents=["doc"], top_n=1)
        assert result.success is False
        assert "query" in result.error.lower()

    def test_execute_empty_documents_fails(self):
        """Empty documents list returns failure."""
        tool = RerankTool(_make_mock_reranker())
        result = tool.execute(query="test", documents=[], top_n=1)
        assert result.success is False
        assert "documents" in result.error.lower()

    def test_execute_invalid_top_n_fails(self):
        """top_n < 1 returns failure."""
        tool = RerankTool(_make_mock_reranker())
        result = tool.execute(query="test", documents=["doc"], top_n=0)
        assert result.success is False
        assert "top_n" in result.error.lower()

    def test_execute_invalid_json_string_fails(self):
        """Malformed JSON string documents returns failure."""
        tool = RerankTool(_make_mock_reranker())
        result = tool.execute(query="test", documents="{invalid json", top_n=1)
        assert result.success is False
        assert "json" in result.error.lower()

    def test_execute_non_string_documents_fails(self):
        """Non-string, non-list documents returns failure."""
        tool = RerankTool(_make_mock_reranker())
        result = tool.execute(query="test", documents=12345, top_n=1)
        assert result.success is False

    def test_execute_non_string_element_fails(self):
        """List with non-string element returns failure."""
        tool = RerankTool(_make_mock_reranker())
        result = tool.execute(query="test", documents=["ok", 123], top_n=1)
        assert result.success is False

    def test_execute_reranker_exception_handled(self):
        """Reranker exception is caught and returns failure."""
        mock_reranker = _make_mock_reranker()
        mock_reranker.rerank.side_effect = RuntimeError("API timeout")
        tool = RerankTool(mock_reranker)

        result = tool.execute(query="test", documents=["doc1"], top_n=1)

        assert result.success is False
        assert "API timeout" in result.error or "rerank" in result.error.lower()

    def test_execution_time_in_metadata(self):
        """Metadata includes execution_time."""
        tool = RerankTool(_make_mock_reranker())
        result = tool.execute(query="test", documents=["doc"], top_n=1)
        assert "execution_time" in result.metadata
        assert result.metadata["execution_time"] >= 0


class TestRerankToolRegistration:
    """Verify conditional registration behavior in create_search_agent."""

    def test_rerank_registered_when_available(self):
        """RerankTool registered when reranker.available=True."""
        from types import SimpleNamespace

        from src.agent.search_agent import create_search_agent

        config = SimpleNamespace(
            glm_api_key="test-key",
            glm_base_url="https://open.bigmodel.cn/api/paas/v4",
            deepseek_api_key="",
            deepseek_base_url="",
            litellm_model="zai/glm-4",
            llm_temperature=0.7,
            llm_max_tokens=4096,
            llm_fast_model="zai/glm-4-flash",
            llm_power_model="zai/glm-4-plus",
            provider="glm",
            active_api_key="test-key",
            active_base_url="https://open.bigmodel.cn/api/paas/v4",
            tiered_routing=False,
        )

        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            idx = Path(tmp) / "index"
            idx.mkdir()
            raw = Path(tmp) / "raw"
            raw.mkdir()
            os.environ["PYTEST_CURRENT_TEST"] = "1"  # Disable litellm Router

            try:
                agent = create_search_agent(
                    config=config,
                    index_path=idx,
                    raw_dir=raw,
                )
                tool_names = [t.name for t in agent._tools.values()]
                assert "rerank" in tool_names
            finally:
                os.environ.pop("PYTEST_CURRENT_TEST", None)

    def test_rerank_not_registered_when_unavailable(self):
        """RerankTool NOT registered when reranker.available=False (no API key)."""
        from types import SimpleNamespace

        from src.agent.search_agent import create_search_agent

        config = SimpleNamespace(
            glm_api_key="test-key",
            glm_base_url="https://open.bigmodel.cn/api/paas/v4",
            deepseek_api_key="",
            deepseek_base_url="",
            litellm_model="zai/glm-4",
            llm_temperature=0.7,
            llm_max_tokens=4096,
            llm_fast_model="zai/glm-4-flash",
            llm_power_model="zai/glm-4-plus",
            provider="glm",
            active_api_key="test-key",
            active_base_url="https://open.bigmodel.cn/api/paas/v4",
            tiered_routing=False,
        )

        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            idx = Path(tmp) / "index"
            idx.mkdir()
            raw = Path(tmp) / "raw"
            raw.mkdir()
            os.environ["PYTEST_CURRENT_TEST"] = "1"

            # Mock create_reranker to return unavailable reranker
            unavailable_reranker = MagicMock()
            unavailable_reranker.available = False

            try:
                with patch("src.search.local_reranker.create_reranker", return_value=unavailable_reranker):
                    agent = create_search_agent(
                        config=config,
                        index_path=idx,
                        raw_dir=raw,
                    )
                    tool_names = [t.name for t in agent._tools.values()]
                    assert "rerank" not in tool_names
            finally:
                os.environ.pop("PYTEST_CURRENT_TEST", None)
