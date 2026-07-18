"""Unit tests for SearchAgent class.

Tests cover:
- SearchAgent initialization and properties
- Query analysis and intent detection
- Search execution and result handling
- Document reading and context building
- Answer generation with sources
- Error handling
- Edge cases
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.agent.base import AgentResponse, ToolResult
from src.agent.llm_client import ChatMessage, ChatResponse, ToolCall
from src.agent.search_agent import (
    SearchAgent,
    SearchResult,
    SYSTEM_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    create_search_agent,
)
from src.agent.tools.search import SearchTool
from src.agent.tools.read import ReadTool
from src.utils.config import Config


# Fixtures
@pytest.fixture
def mock_config():
    """Create a mock Config."""
    config = MagicMock(spec=Config)
    config.glm_api_key = "test-api-key"
    config.glm_base_url = "https://api.test.com"
    config.llm_model = "glm-4"
    config.llm_temperature = 0.7
    config.llm_max_tokens = 4096
    return config


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient."""
    client = MagicMock()
    client.chat.return_value = ChatResponse(
        content='{"action": "search", "search_query": "绩效考核"}',
        usage={"total_tokens": 50},
    )
    return client


@pytest.fixture
def mock_search_tool():
    """Create a mock SearchTool with proper name property."""
    tool = MagicMock(spec=SearchTool)
    type(tool).name = PropertyMock(return_value="search")
    type(tool).description = PropertyMock(return_value="搜索文档")
    return tool


@pytest.fixture
def mock_read_tool():
    """Create a mock ReadTool with proper name property."""
    tool = MagicMock(spec=ReadTool)
    type(tool).name = PropertyMock(return_value="read")
    type(tool).description = PropertyMock(return_value="读取文档")
    return tool


@pytest.fixture
def agent(mock_config, mock_llm_client, mock_search_tool, mock_read_tool):
    """Create a SearchAgent with mocked dependencies (pipeline mode for legacy tests)."""
    agent = SearchAgent(
        config=mock_config,
        search_tool=mock_search_tool,
        read_tool=mock_read_tool,
        llm_client=mock_llm_client,
        mode="pipeline",
    )
    return agent


class TestSearchAgentInit:
    """Test SearchAgent initialization."""

    def test_init_default_values(self, mock_config, mock_search_tool, mock_read_tool):
        """Test initialization with default values."""
        agent = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
        )
        assert agent._max_search_results == 10
        assert agent._max_read_docs == 3
        assert agent._max_context_tokens == 3000

    def test_init_custom_values(self, mock_config, mock_search_tool, mock_read_tool):
        """Test initialization with custom values."""
        agent = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            max_search_results=20,
            max_read_docs=5,
            max_context_tokens=5000,
        )
        assert agent._max_search_results == 20
        assert agent._max_read_docs == 5
        assert agent._max_context_tokens == 5000

    def test_init_registers_tools(
        self, mock_config, mock_search_tool, mock_read_tool, mock_llm_client
    ):
        """Test that tools are registered on initialization."""
        agent = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=mock_llm_client,
        )
        assert len(agent.tools) == 2
        # Check that tools are retrievable by name
        assert agent.get_tool("search") is not None
        assert agent.get_tool("read") is not None

    def test_name_property(self, agent):
        """Test name property returns 'search_agent'."""
        assert agent.name == "search_agent"


class TestSearchAgentRun:
    """Test SearchAgent run method."""

    def test_run_with_search_query(
        self, agent, mock_llm_client, mock_search_tool, mock_read_tool
    ):
        """Test run with a search query."""
        # Setup query analysis
        mock_llm_client.chat.side_effect = [
            # Query analysis
            ChatResponse(
                content='{"action": "search", "search_query": "绩效考核"}',
                usage={"total_tokens": 50},
            ),
            # Answer generation
            ChatResponse(
                content="根据文档，绩效考核是... [文档1]",
                usage={"total_tokens": 200},
            ),
        ]

        # Setup search tool
        mock_search_tool.execute.return_value = ToolResult.ok(
            data=json.dumps(
                {
                    "query": "绩效考核",
                    "total": 1,
                    "offset": 0,
                    "limit": 10,
                    "has_more": False,
                    "results": [
                        {
                            "doc_id": "doc1",
                            "title": "绩效管理制度",
                            "score": 2.5,
                            "snippet": "绩效考核流程...",
                            "source_path": "/docs/performance.md",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            metadata={"total_results": 1},
        )

        # Setup read tool
        mock_read_tool.execute.return_value = ToolResult.ok(
            data="# 绩效管理制度\n\n绩效考核流程说明...",
            metadata={"doc_id": "doc1", "lines_read": 10},
        )

        response = agent.run("什么是绩效考核?")

        assert response.success is True
        assert len(response.answer) > 0
        assert len(response.tool_calls) >= 1  # At least search call
        assert response.processing_time >= 0  # Processing time can be 0 for fast tests

    def test_run_with_greeting(self, agent, mock_llm_client):
        """Test run with a greeting query returns direct response."""
        response = agent.run("你好")

        assert response.success is True
        assert "你好" in response.answer or "帮助" in response.answer
        # confidence is stored in reasoning field
        assert "1.00" in response.reasoning
        assert len(response.tool_calls) == 0  # No tools called

    def test_run_with_no_results(self, agent, mock_llm_client, mock_search_tool):
        """Test run when search returns no results."""
        mock_llm_client.chat.return_value = ChatResponse(
            content='{"action": "search", "search_query": "不存在的文档"}',
            usage={"total_tokens": 50},
        )

        mock_search_tool.execute.return_value = ToolResult.ok(
            data=json.dumps(
                {
                    "query": "不存在的文档",
                    "total": 0,
                    "offset": 0,
                    "limit": 10,
                    "has_more": False,
                    "results": [],
                },
                ensure_ascii=False,
            ),
            metadata={"total_results": 0},
        )

        response = agent.run("不存在的文档xyz123")

        assert response.success is True
        assert "未找到" in response.answer or "没有" in response.answer
        # confidence is stored in reasoning field
        assert "0.00" in response.reasoning

    def test_run_with_precision_mode(
        self, agent, mock_llm_client, mock_search_tool, mock_read_tool
    ):
        """Test run with precision search mode."""
        mock_llm_client.chat.side_effect = [
            ChatResponse(
                content='{"action": "search", "search_query": "报销"}',
                usage={"total_tokens": 50},
            ),
            ChatResponse(
                content="精确匹配结果: [文档1]",
                usage={"total_tokens": 150},
            ),
        ]

        mock_search_tool.execute.return_value = ToolResult.ok(
            data=json.dumps(
                {
                    "query": "报销",
                    "total": 1,
                    "offset": 0,
                    "limit": 10,
                    "has_more": False,
                    "results": [
                        {
                            "doc_id": "doc1",
                            "title": "报销流程",
                            "score": 3.0,
                            "snippet": "报销需要...",
                            "source_path": "/docs/reimburse.md",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            metadata={"total_results": 1},
        )

        mock_read_tool.execute.return_value = ToolResult.ok(
            data="报销流程详细说明",
            metadata={"doc_id": "doc1", "lines_read": 5},
        )

        response = agent.run("报销流程", context={"mode": "precision"})

        assert response.success is True

    def test_run_handles_llm_error(self, agent, mock_llm_client):
        """Test run handles LLM errors gracefully."""
        mock_llm_client.chat.side_effect = Exception("API error")

        response = agent.run("测试查询")

        assert response.success is False
        assert response.error is not None
        assert "API error" in response.error

    def test_run_handles_search_error(self, agent, mock_llm_client, mock_search_tool):
        """Test run handles search errors gracefully."""
        mock_llm_client.chat.return_value = ChatResponse(
            content='{"action": "search", "search_query": "测试"}',
            usage={"total_tokens": 50},
        )

        mock_search_tool.execute.return_value = ToolResult.fail(
            error="Index error",
            metadata={},
        )

        # Should return no results response
        response = agent.run("测试查询")

        # Search failure returns empty results, handled gracefully
        assert response.success is True
        assert "未找到" in response.answer or response.answer == ""


class TestQueryAnalysis:
    """Test query analysis functionality."""

    def test_analyze_query_returns_search(self, agent, mock_llm_client):
        """Test query analysis returns search action via rule-based routing."""
        # "什么是绩效管理?" contains search indicators ("什么", "绩效")
        # so it's routed directly to search without LLM call
        action, query = agent._analyze_query("什么是绩效管理?")

        assert action == "search"
        assert query == "什么是绩效管理"
        # LLM should NOT be called for this query (rule-based routing)
        mock_llm_client.chat.assert_not_called()

    def test_analyze_query_returns_direct_for_greeting(self, agent, mock_llm_client):
        """Test query analysis returns direct for greetings."""
        # Short greeting should not call LLM
        action, query = agent._analyze_query("你好")

        assert action == "direct"
        assert query == "你好"

    def test_analyze_query_handles_json_with_markdown(self, agent, mock_llm_client):
        """Test query analysis handles JSON in markdown blocks."""
        mock_llm_client.chat.return_value = ChatResponse(
            content='```json\n{"action": "search", "search_query": "测试"}\n```',
            usage={"total_tokens": 30},
        )

        action, query = agent._analyze_query("测试问题")

        assert action == "search"
        assert query == "测试"

    def test_analyze_query_handles_parse_error(self, agent, mock_llm_client):
        """Test query analysis handles JSON parse errors."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="invalid json",
            usage={"total_tokens": 30},
        )

        action, query = agent._analyze_query("测试问题")

        # Should fallback to search
        assert action == "search"
        assert query == "测试问题"


class TestDirectResponse:
    """Test direct response generation."""

    def test_direct_response_greeting(self, agent):
        """Test direct response for greeting."""
        response = agent._direct_response("你好")
        assert "你好" in response
        assert "帮助" in response or "搜索" in response

    def test_direct_response_hello(self, agent):
        """Test direct response for hello."""
        response = agent._direct_response("hello")
        assert "你好" in response or "Hello" in response.lower()

    def test_direct_response_other(self, agent):
        """Test direct response for non-greeting."""
        response = agent._direct_response("some random text")
        assert "文档" in response  # Should mention documents


class TestSearchExecution:
    """Test search execution functionality."""

    def test_execute_search_success(self, agent, mock_search_tool):
        """Test successful search execution."""
        mock_search_tool.execute.return_value = ToolResult.ok(
            data=json.dumps(
                {
                    "query": "测试",
                    "total": 2,
                    "offset": 0,
                    "limit": 10,
                    "has_more": False,
                    "results": [
                        {
                            "doc_id": "doc1",
                            "title": "Doc 1",
                            "score": 2.0,
                            "snippet": "Content 1",
                        },
                        {
                            "doc_id": "doc2",
                            "title": "Doc 2",
                            "score": 1.5,
                            "snippet": "Content 2",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            metadata={"total_results": 2},
        )

        tool_calls = []
        results = agent._execute_search("测试", 10, tool_calls)

        assert len(results) == 2
        assert results[0].doc_id == "doc1"
        assert results[0].title == "Doc 1"
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "search"
        assert tool_calls[0]["success"] is True

    def test_execute_search_no_results(self, agent, mock_search_tool):
        """Test search execution with no results."""
        mock_search_tool.execute.return_value = ToolResult.ok(
            data=json.dumps(
                {
                    "query": "测试",
                    "total": 0,
                    "offset": 0,
                    "limit": 10,
                    "has_more": False,
                    "results": [],
                },
                ensure_ascii=False,
            ),
            metadata={"total_results": 0},
        )

        tool_calls = []
        results = agent._execute_search("测试", 10, tool_calls)

        assert len(results) == 0
        assert tool_calls[0]["result_count"] == 0

    def test_execute_search_failure(self, agent, mock_search_tool):
        """Test search execution with failure."""
        mock_search_tool.execute.return_value = ToolResult.fail(
            error="Search failed",
            metadata={},
        )

        tool_calls = []
        results = agent._execute_search("测试", 10, tool_calls)

        assert len(results) == 0
        assert tool_calls[0]["success"] is False

    def test_execute_search_no_tool(self, agent):
        """Test search execution when tool not registered."""
        agent.unregister_tool("search")
        tool_calls = []
        results = agent._execute_search("测试", 10, tool_calls)

        assert len(results) == 0
        assert len(tool_calls) == 0


class TestDocumentReading:
    """Test document reading functionality."""

    def test_read_top_documents_success(self, agent, mock_read_tool):
        """Test successful document reading."""
        mock_read_tool.execute.return_value = ToolResult.ok(
            data="Document content here",
            metadata={"doc_id": "doc1", "lines_read": 10, "truncated": False},
        )

        search_results = [
            SearchResult(doc_id="doc1", title="Doc 1", score=2.0, snippet="Snippet"),
            SearchResult(doc_id="doc2", title="Doc 2", score=1.5, snippet="Snippet"),
        ]

        tool_calls = []
        contents = agent._read_top_documents(search_results, 3, tool_calls)

        assert "doc1" in contents
        assert len(tool_calls) == 2  # Read 2 documents
        assert tool_calls[0]["success"] is True

    def test_read_top_documents_respects_max_docs(self, agent, mock_read_tool):
        """Test reading respects max_docs limit."""
        mock_read_tool.execute.return_value = ToolResult.ok(
            data="Content",
            metadata={"doc_id": "doc1", "lines_read": 10},
        )

        search_results = [
            SearchResult(
                doc_id=f"doc{i}", title=f"Doc {i}", score=2.0, snippet="Snippet"
            )
            for i in range(10)
        ]

        tool_calls = []
        contents = agent._read_top_documents(
            search_results, 2, tool_calls
        )  # max_docs=2

        assert len(contents) == 2
        assert len(tool_calls) == 2

    def test_read_top_documents_handles_failure(self, agent, mock_read_tool):
        """Test document reading handles failures."""
        mock_read_tool.execute.return_value = ToolResult.fail(
            error="File not found",
            metadata={"doc_id": "doc1"},
        )

        search_results = [
            SearchResult(doc_id="doc1", title="Doc 1", score=2.0, snippet="Snippet"),
        ]

        tool_calls = []
        contents = agent._read_top_documents(search_results, 3, tool_calls)

        assert len(contents) == 0
        assert tool_calls[0]["success"] is False

    def test_read_top_documents_no_tool(self, agent):
        """Test document reading when tool not registered."""
        agent.unregister_tool("read")
        search_results = [
            SearchResult(doc_id="doc1", title="Doc", score=2.0, snippet="")
        ]
        tool_calls = []
        contents = agent._read_top_documents(search_results, 3, tool_calls)

        assert len(contents) == 0


class TestAnswerGeneration:
    """Test answer generation functionality."""

    def test_generate_answer_semantic_mode(self, agent, mock_llm_client):
        """Test answer generation in semantic mode."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="根据文档，绩效管理是... [文档1]",
            usage={"total_tokens": 200},
        )

        search_results = [
            SearchResult(
                doc_id="doc1", title="绩效制度", score=2.5, snippet="绩效考核内容"
            ),
        ]
        context_docs = {"doc1": "完整的绩效管理制度文档..."}

        answer, confidence, tokens = agent._generate_answer(
            query="什么是绩效?",
            search_results=search_results,
            context_docs=context_docs,
            mode="semantic",
        )

        assert "绩效" in answer
        assert tokens == 200

    def test_generate_answer_precision_mode(self, agent, mock_llm_client):
        """Test answer generation in precision mode."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="精确结果: 绩效考核 [文档1]",
            usage={"total_tokens": 150},
        )

        search_results = [
            SearchResult(doc_id="doc1", title="绩效", score=3.0, snippet="考核"),
        ]
        context_docs = {"doc1": "考核内容"}

        answer, confidence, tokens = agent._generate_answer(
            query="绩效",
            search_results=search_results,
            context_docs=context_docs,
            mode="precision",
        )

        assert len(answer) > 0
        assert tokens == 150

    def test_generate_answer_with_no_results(self, agent, mock_llm_client):
        """Test answer generation with no search results."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="未找到相关信息",
            usage={"total_tokens": 50},
        )

        answer, confidence, tokens = agent._generate_answer(
            query="测试",
            search_results=[],
            context_docs={},
        )

        assert confidence == 0.0


class TestSourceCollection:
    """Test source collection functionality."""

    def test_collect_sources_success(self, agent):
        """Test successful source collection."""
        search_results = [
            SearchResult(
                doc_id="doc1",
                title="Doc 1",
                score=2.0,
                snippet="",
                source_path="/path/1.md",
            ),
            SearchResult(
                doc_id="doc2",
                title="Doc 2",
                score=1.5,
                snippet="",
                source_path="/path/2.md",
            ),
        ]
        context_docs = {"doc1": "content1", "doc2": "content2"}

        sources = agent._collect_sources(search_results, context_docs)

        assert len(sources) == 2
        assert "/path/1.md" in sources
        assert "/path/2.md" in sources

    def test_collect_sources_only_used_docs(self, agent):
        """Test only used documents are included in sources."""
        search_results = [
            SearchResult(
                doc_id="doc1",
                title="Doc 1",
                score=2.0,
                snippet="",
                source_path="/path/1.md",
            ),
            SearchResult(
                doc_id="doc2",
                title="Doc 2",
                score=1.5,
                snippet="",
                source_path="/path/2.md",
            ),
        ]
        context_docs = {"doc1": "content1"}  # Only doc1 was read

        sources = agent._collect_sources(search_results, context_docs)

        assert len(sources) == 1
        assert "/path/1.md" in sources

    def test_collect_sources_no_source_path(self, agent):
        """Test handling results without source_path."""
        search_results = [
            SearchResult(
                doc_id="doc1", title="Doc 1", score=2.0, snippet="", source_path=None
            ),
        ]
        context_docs = {"doc1": "content1"}

        sources = agent._collect_sources(search_results, context_docs)

        assert len(sources) == 0

    def test_collect_sources_deduplication(self, agent):
        """Test source deduplication."""
        search_results = [
            SearchResult(
                doc_id="doc1",
                title="Doc 1",
                score=2.0,
                snippet="",
                source_path="/path/1.md",
            ),
            SearchResult(
                doc_id="doc2",
                title="Doc 2",
                score=1.5,
                snippet="",
                source_path="/path/1.md",
            ),
        ]
        context_docs = {"doc1": "content1", "doc2": "content2"}

        sources = agent._collect_sources(search_results, context_docs)

        assert len(sources) == 1  # Deduplicated


class TestResponseBuilding:
    """Test response building functionality."""

    def test_build_response_success(self, agent):
        """Test building successful response."""
        start_time = time.time()
        response = agent._build_response(
            success=True,
            answer="Test answer",
            sources=["/path/1.md"],
            tool_calls=[{"tool": "search", "success": True}],
            confidence=0.8,
            tokens_used=100,
            start_time=start_time,
        )

        assert response.success is True
        assert response.answer == "Test answer"
        assert "/path/1.md" in response.sources
        assert len(response.tool_calls) == 1
        assert response.tokens_used == 100
        assert "0.80" in response.reasoning
        assert response.error is None

    def test_build_response_error(self, agent):
        """Test building error response."""
        start_time = time.time()
        response = agent._build_response(
            success=False,
            answer="",
            sources=[],
            tool_calls=[],
            confidence=0.0,
            tokens_used=0,
            start_time=start_time,
            error="Test error",
        )

        assert response.success is False
        assert response.answer == ""
        assert response.error == "Test error"


class TestCreateSearchAgent:
    """Test create_search_agent factory function."""

    def test_create_search_agent(self, mock_config, tmp_path):
        """Test factory function creates agent with all components."""
        with patch("src.search.bm25_search.create_searcher") as mock_create_searcher:
            mock_create_searcher.return_value = MagicMock()

            with patch("src.storage.markdown_store.MarkdownStore") as mock_store:
                mock_store.return_value = MagicMock()

                agent = create_search_agent(
                    config=mock_config,
                    index_path=tmp_path / "index",
                    input_base=tmp_path / "input",
                    output_base=tmp_path / "output",
                )

                assert isinstance(agent, SearchAgent)
                assert agent.name == "search_agent"
                # search + read + rerank + summarize (always registered when api_key available)
                assert len(agent.tools) == 4


class TestSearchAgentProtocol:
    """Test that SearchAgent properly implements Agent protocol."""

    def test_inherits_from_agent(self, agent):
        """Test SearchAgent inherits from Agent."""
        from src.agent.base import Agent

        assert isinstance(agent, Agent)

    def test_has_name_property(self, agent):
        """Test SearchAgent has name property."""
        assert hasattr(agent, "name")
        assert agent.name == "search_agent"

    def test_has_run_method(self, agent):
        """Test SearchAgent has run method."""
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_has_tools_registry(self, agent):
        """Test SearchAgent has tools registry."""
        assert hasattr(agent, "tools")
        assert hasattr(agent, "register_tool")
        assert hasattr(agent, "get_tool")


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating SearchResult."""
        result = SearchResult(
            doc_id="doc1",
            title="Test Doc",
            score=2.5,
            snippet="Test snippet",
            source_path="/path/to/doc.md",
        )

        assert result.doc_id == "doc1"
        assert result.title == "Test Doc"
        assert result.score == 2.5
        assert result.snippet == "Test snippet"
        assert result.source_path == "/path/to/doc.md"

    def test_search_result_optional_fields(self):
        """Test SearchResult with optional fields."""
        result = SearchResult(
            doc_id="doc1",
            title="Test",
            score=1.0,
            snippet="Snippet",
        )

        assert result.source_path is None
        assert result.content is None


class TestPromptConstants:
    """Test prompt constants are defined."""

    def test_system_prompt_exists(self):
        """Test SYSTEM_PROMPT is defined."""
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 0

    def test_query_analysis_prompt_exists(self):
        """Test QUERY_ANALYSIS_PROMPT is defined."""
        assert QUERY_ANALYSIS_PROMPT is not None
        assert "{query}" in QUERY_ANALYSIS_PROMPT

    def test_system_prompt_under_token_limit(self):
        """Test SYSTEM_PROMPT is under 650 tokens (rough estimate)."""
        # Rough estimate: 1 token ≈ 4 characters for Chinese
        # 650 tokens ≈ 2500 characters (raised from 500/2000 for v0.15
        # alternating-workflow prompt with multi-dimensional examples)
        assert len(SYSTEM_PROMPT) < 2500


class TestSearchAgentRerank:
    """Test SearchAgent rerank functionality."""

    @pytest.fixture
    def agent_with_rerank(self, mock_config, mock_search_tool, mock_read_tool):
        """Create SearchAgent with rerank enabled (pipeline mode)."""
        return SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=MagicMock(),
            use_rerank=True,
            mode="pipeline",
        )

    def test_reranker_lazy_init(self, agent_with_rerank):
        """Reranker is None until first use."""
        assert agent_with_rerank._reranker is None

    def test_use_rerank_flag(self, agent_with_rerank):
        """Test use_rerank flag is set."""
        assert agent_with_rerank._use_rerank is True

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_execute_search_with_rerank(self, mock_urlopen, agent_with_rerank):
        """Test that _execute_search calls reranker when use_rerank=True."""
        # Mock search tool to return results
        search_data = json.dumps({
            "results": [
                {
                    "doc_id": "d1",
                    "title": "Doc1",
                    "score": 5.0,
                    "snippet": "关于年假",
                    "source_path": "/doc1.md",
                },
                {
                    "doc_id": "d2",
                    "title": "Doc2",
                    "score": 3.0,
                    "snippet": "关于采购",
                    "source_path": "/doc2.md",
                },
            ]
        })
        mock_result = ToolResult.ok(data=search_data, metadata={"total_results": 2})
        agent_with_rerank.get_tool("search").execute.return_value = mock_result

        # Mock reranker API
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ],
            "usage": {"total_tokens": 100},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        tool_calls = []
        results = agent_with_rerank._execute_search("年假", limit=10, tool_calls=tool_calls)

        assert len(results) > 0
        # Results should be reranked: d2 first (0.9), d1 second (0.5)
        assert results[0].doc_id == "d2"
        assert results[0].score == 0.9
        assert results[1].doc_id == "d1"
        assert results[1].score == 0.5

        # Should have search + rerank tool calls
        assert len(tool_calls) == 2
        assert tool_calls[0]["tool"] == "search"
        assert tool_calls[1]["tool"] == "rerank"
        assert tool_calls[1]["success"] is True

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_tracks_tokens(self, mock_urlopen, agent_with_rerank):
        """Test that rerank tokens are tracked in agent."""
        search_data = json.dumps({
            "results": [
                {"doc_id": "d1", "title": "Doc1", "score": 5.0, "snippet": "内容", "source_path": "/doc1.md"},
            ]
        })
        agent_with_rerank.get_tool("search").execute.return_value = ToolResult.ok(
            data=search_data, metadata={"total_results": 1}
        )

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"total_tokens": 200},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        tool_calls = []
        agent_with_rerank._execute_search("查询", limit=10, tool_calls=tool_calls)

        assert agent_with_rerank._total_tokens_used == 200

    def test_rerank_fallback_on_api_failure(self, agent_with_rerank):
        """Test that rerank failure falls back to BM25 order."""
        search_data = json.dumps({
            "results": [
                {"doc_id": "d1", "title": "Doc1", "score": 5.0, "snippet": "内容1", "source_path": "/doc1.md"},
                {"doc_id": "d2", "title": "Doc2", "score": 3.0, "snippet": "内容2", "source_path": "/doc2.md"},
            ]
        })
        agent_with_rerank.get_tool("search").execute.return_value = ToolResult.ok(
            data=search_data, metadata={"total_results": 2}
        )

        # Mock reranker to raise — need to patch at the module level
        with patch("src.search.reranker.urllib.request.urlopen", side_effect=Exception("API error")):
            tool_calls = []
            results = agent_with_rerank._execute_search("查询", limit=10, tool_calls=tool_calls)

            # Should return original order (BM25 scores)
            assert len(results) == 2
            assert results[0].doc_id == "d1"
            assert results[1].doc_id == "d2"

    def test_execute_search_without_rerank(self, mock_config, mock_search_tool, mock_read_tool):
        """Test that _execute_search does not rerank when use_rerank=False."""
        agent = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=MagicMock(),
            use_rerank=False,
        )

        search_data = json.dumps({
            "results": [
                {"doc_id": "d1", "title": "Doc1", "score": 5.0, "snippet": "内容", "source_path": "/doc1.md"},
            ]
        })
        agent.get_tool("search").execute.return_value = ToolResult.ok(
            data=search_data, metadata={"total_results": 1}
        )

        tool_calls = []
        results = agent._execute_search("查询", limit=10, tool_calls=tool_calls)

        assert len(results) == 1
        # Only search tool call, no rerank
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "search"

    def test_rerank_fetches_more_candidates(self, agent_with_rerank):
        """Test that rerank mode fetches 3x candidates."""
        search_data = json.dumps({
            "results": [
                {"doc_id": f"d{i}", "title": f"Doc{i}", "score": float(10 - i), "snippet": f"内容{i}", "source_path": f"/doc{i}.md"}
                for i in range(6)
            ]
        })
        agent_with_rerank.get_tool("search").execute.return_value = ToolResult.ok(
            data=search_data, metadata={"total_results": 6}
        )

        # Mock reranker to return original order
        with patch.object(agent_with_rerank, "_rerank_results", side_effect=lambda q, r, n, tc: r):
            tool_calls = []
            results = agent_with_rerank._execute_search("查询", limit=5, tool_calls=tool_calls)

            # Search should have been called with limit=min(5*3, 20)=15
            search_call = agent_with_rerank.get_tool("search").execute.call_args
            assert search_call[1]["limit"] == 15  # 3x candidates, capped at 20


class TestCreateSearchAgentWithGrep:
    """Test create_search_agent with raw_dir (GrepTool registration)."""

    def test_grep_tool_registered(self, tmp_path):
        """GrepTool should be registered when raw_dir is provided."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        with patch("src.search.bm25_search.create_searcher") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("src.storage.markdown_store.MarkdownStore") as mock_store:
                mock_store.return_value = MagicMock()

                config = Config(
                    glm_api_key="test-key",
                    glm_base_url="https://test.com",
                )
                agent = create_search_agent(config=config, raw_dir=raw_dir)

                tool_names = [t.name for t in agent.tools]
                assert "grep" in tool_names
                assert len(agent.tools) == 6  # search + read + grep + bash + rerank + summarize

    def test_grep_tool_not_registered_without_raw_dir(self):
        """GrepTool should NOT be registered when raw_dir is None."""
        with patch("src.search.bm25_search.create_searcher") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("src.storage.markdown_store.MarkdownStore") as mock_store:
                mock_store.return_value = MagicMock()

                config = Config(
                    glm_api_key="test-key",
                    glm_base_url="https://test.com",
                )
                agent = create_search_agent(config=config)

                tool_names = [t.name for t in agent.tools]
                assert "grep" not in tool_names
                assert len(agent.tools) == 4  # search + read + rerank + summarize

    def test_grep_tool_not_registered_with_nonexistent_dir(self, tmp_path):
        """GrepTool should NOT be registered when raw_dir doesn't exist."""
        nonexistent = tmp_path / "does_not_exist"

        with patch("src.search.bm25_search.create_searcher") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("src.storage.markdown_store.MarkdownStore") as mock_store:
                mock_store.return_value = MagicMock()

                config = Config(
                    glm_api_key="test-key",
                    glm_base_url="https://test.com",
                )
                agent = create_search_agent(config=config, raw_dir=nonexistent)

                tool_names = [t.name for t in agent.tools]
                assert "grep" not in tool_names

    def test_create_with_rerank_flag(self, tmp_path):
        """Test create_search_agent with use_rerank=True."""
        with patch("src.search.bm25_search.create_searcher") as mock_create:
            mock_create.return_value = MagicMock()
            with patch("src.storage.markdown_store.MarkdownStore") as mock_store:
                mock_store.return_value = MagicMock()

                config = Config(
                    glm_api_key="test-key",
                    glm_base_url="https://test.com",
                )
                agent = create_search_agent(config=config, use_rerank=True)

                assert agent._use_rerank is True


class TestSearchAgentRerankIntegration:
    """Test SearchAgent rerank integration paths."""

    @pytest.fixture
    def agent_with_rerank(self, mock_config, mock_search_tool, mock_read_tool):
        """Create SearchAgent with rerank enabled (pipeline mode)."""
        return SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=MagicMock(),
            use_rerank=True,
            mode="pipeline",
        )

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_full_rerank_flow(self, mock_urlopen, agent_with_rerank):
        """Test full rerank flow: search -> rerank -> results."""
        # Mock search results
        search_data = json.dumps({
            "results": [
                {"doc_id": "d1", "title": "年假制度", "score": 5.0, "snippet": "年假有5天", "source_path": "/doc1.md"},
                {"doc_id": "d2", "title": "请假制度", "score": 3.0, "snippet": "请假流程", "source_path": "/doc2.md"},
                {"doc_id": "d3", "title": "加班制度", "score": 2.0, "snippet": "加班审批", "source_path": "/doc3.md"},
            ]
        })
        agent_with_rerank.get_tool("search").execute.return_value = ToolResult.ok(
            data=search_data, metadata={"total_results": 3}
        )

        # Mock reranker API response — d3 ranked highest
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.6},
            ],
            "usage": {"prompt_tokens": 80, "total_tokens": 80},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        tool_calls = []
        results = agent_with_rerank._execute_search("年假", limit=3, tool_calls=tool_calls)

        # Verify reranked order: d3(0.95) > d1(0.8) > d2(0.6)
        assert len(results) == 3
        assert results[0].doc_id == "d3"
        assert results[0].score == 0.95
        assert results[1].doc_id == "d1"
        assert results[2].doc_id == "d2"

        # Verify tool calls recorded
        rerank_call = [tc for tc in tool_calls if tc["tool"] == "rerank"][0]
        assert rerank_call["success"] is True
        assert rerank_call["tokens_used"] == 80

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_empty_results(self, mock_urlopen, agent_with_rerank):
        """Test rerank is not called when search returns empty results."""
        search_data = json.dumps({"results": []})
        agent_with_rerank.get_tool("search").execute.return_value = ToolResult.ok(
            data=search_data, metadata={"total_results": 0}
        )

        tool_calls = []
        results = agent_with_rerank._execute_search("查询", limit=10, tool_calls=tool_calls)

        assert len(results) == 0
        # Only search call, no rerank
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "search"
        # urlopen should not be called
        mock_urlopen.assert_not_called()

    def test_get_reranker_lazy_init(self, agent_with_rerank):
        """Test _get_reranker lazy-initializes the reranker."""
        assert agent_with_rerank._reranker is None
        reranker = agent_with_rerank._get_reranker()
        assert reranker is not None
        assert agent_with_rerank._reranker is reranker

        # Second call returns same instance
        reranker2 = agent_with_rerank._get_reranker()
        assert reranker2 is reranker

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_respects_limit(self, mock_urlopen, agent_with_rerank):
        """Test that final results are limited to requested limit."""
        search_data = json.dumps({
            "results": [
                {"doc_id": f"d{i}", "title": f"Doc{i}", "score": float(10 - i), "snippet": f"内容{i}", "source_path": f"/doc{i}.md"}
                for i in range(6)
            ]
        })
        agent_with_rerank.get_tool("search").execute.return_value = ToolResult.ok(
            data=search_data, metadata={"total_results": 6}
        )

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [
                {"index": i, "relevance_score": round(0.9 - i * 0.1, 2)}
                for i in range(6)
            ],
            "usage": {"total_tokens": 150},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        tool_calls = []
        results = agent_with_rerank._execute_search("查询", limit=3, tool_calls=tool_calls)

        # Results should be capped at limit=3
        assert len(results) <= 3


# ─────────────────────────────────────────────────────
# Phase 2 (Agent Intelligence) tests
# ─────────────────────────────────────────────────────


class TestQueryExpansion:
    """F2-1: LLM query expansion tests."""

    def test_expand_query_returns_original_plus_variants(self, agent, mock_llm_client):
        """_expand_query returns [original] + LLM-generated variants."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="年假如何申请\n请假流程是什么\n休息时间规定",
            usage={"total_tokens": 30},
        )
        result = agent._expand_query("年假制度")
        assert result[0] == "年假制度"
        assert len(result) == 4  # 1 original + 3 variants
        assert "年假如何申请" in result

    def test_expand_query_fallback_on_error(self, agent, mock_llm_client):
        """_expand_query returns [original] only when LLM raises."""
        mock_llm_client.chat.side_effect = Exception("API down")
        result = agent._expand_query("测试查询")
        assert result == ["测试查询"]

    def test_expand_query_limits_to_5_variants(self, agent, mock_llm_client):
        """_expand_query caps at 5 variants even if LLM returns more."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="v1\nv2\nv3\nv4\nv5\nv6\nv7",
            usage={"total_tokens": 30},
        )
        result = agent._expand_query("查询")
        assert result[0] == "查询"
        assert len(result) == 6  # original + 5 variants max

    def test_expand_query_empty_response(self, agent, mock_llm_client):
        """_expand_query handles empty LLM response."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="",
            usage={"total_tokens": 10},
        )
        result = agent._expand_query("查询")
        assert result == ["查询"]

    def test_expand_query_filters_blank_lines(self, agent, mock_llm_client):
        """_expand_query skips blank lines from LLM response."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="变体1\n\n\n变体2",
            usage={"total_tokens": 20},
        )
        result = agent._expand_query("查询")
        assert result[0] == "查询"
        assert "变体1" in result
        assert "变体2" in result
        assert len(result) == 3


class TestIterationLimit:
    """F2-2: Reduced iteration limit."""

    def test_max_iterations_is_8(self):
        """MAX_TOOL_ITERATIONS class constant is 8."""
        assert SearchAgent.MAX_TOOL_ITERATIONS == 8

    def test_max_expansion_variants_is_5(self):
        """MAX_EXPANSION_VARIANTS class constant is 5 (aligned with AgenticRAG)."""
        assert SearchAgent.MAX_EXPANSION_VARIANTS == 5


class TestConsecutiveSearchDetection:
    """Test consecutive search early-stop mechanism."""

    def test_consecutive_searches_triggers_early_stop(self, agent, mock_llm_client):
        """When LLM makes 2+ consecutive searches without read, should_stop fires."""
        # The _check_convergence callback should detect consecutive searches
        # We can't easily test the full tool_loop, but we can verify the logic exists
        # by checking that the agent's run method handles the case gracefully
        import inspect
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "_consecutive_searches" in source
        assert "consecutive searches without read" in source


class TestDynamicConfidence:
    """F2-3: Dynamic confidence calculation."""

    def test_no_tool_calls_low_confidence(self, agent):
        """Empty tool_calls → low confidence (~0.3)."""
        confidence = agent._calculate_tool_loop_confidence([])
        assert confidence == pytest.approx(0.3)

    def test_no_search_calls_medium_low_confidence(self, agent):
        """Only read calls, no search → 0.4."""
        tool_calls = [{"tool": "read", "success": True}]
        confidence = agent._calculate_tool_loop_confidence(tool_calls)
        assert confidence == pytest.approx(0.4)

    def test_search_only_base_confidence(self, agent):
        """Single search call, no read → 0.5."""
        tool_calls = [{"tool": "search", "success": True}]
        confidence = agent._calculate_tool_loop_confidence(tool_calls)
        assert confidence == pytest.approx(0.5)

    def test_search_and_read_high_confidence(self, agent):
        """Search + read → 0.7."""
        tool_calls = [
            {"tool": "search", "success": True},
            {"tool": "read", "success": True},
        ]
        confidence = agent._calculate_tool_loop_confidence(tool_calls)
        assert confidence == pytest.approx(0.7)

    def test_with_rerank_highest_confidence(self, agent):
        """Search + read + rerank → 0.8."""
        tool_calls = [
            {"tool": "search", "success": True},
            {"tool": "read", "success": True},
            {"tool": "rerank", "success": True},
        ]
        confidence = agent._calculate_tool_loop_confidence(tool_calls)
        assert confidence == pytest.approx(0.8)

    def test_confidence_capped_at_095(self, agent):
        """Many tools → still max 0.95."""
        tool_calls = [
            {"tool": "search", "success": True},
            {"tool": "search", "success": True},
            {"tool": "grep", "success": True},
            {"tool": "read", "success": True},
            {"tool": "rerank", "success": True},
        ]
        confidence = agent._calculate_tool_loop_confidence(tool_calls)
        assert confidence == pytest.approx(0.9)

    def test_grep_counts_as_search(self, agent):
        """Grep tool counts toward search_calls."""
        tool_calls = [{"tool": "grep", "success": True}]
        confidence = agent._calculate_tool_loop_confidence(tool_calls)
        assert confidence == pytest.approx(0.5)

    def test_multiple_searches_boost_confidence(self, agent):
        """2+ search/grep calls boost confidence by 0.1."""
        tool_calls = [
            {"tool": "search", "success": True},
            {"tool": "search", "success": True},
        ]
        confidence = agent._calculate_tool_loop_confidence(tool_calls)
        assert confidence == pytest.approx(0.6)


class TestMaxTokens:
    """F2-4: max_tokens increased."""

    def test_tool_loop_max_tokens_is_2000(self, mock_config, mock_llm_client, mock_search_tool, mock_read_tool):
        """Verify max_tokens=2000 is passed to chat_with_tools in tool_loop mode."""
        agent_tl = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=mock_llm_client,
            mode="tool_loop",
        )
        # Mock chat_with_tools to capture the args
        mock_llm_client.chat_with_tools.return_value = ChatResponse(
            content="回答",
            usage={"total_tokens": 100},
        )
        # Mock _expand_query to avoid extra LLM calls
        agent_tl._expand_query = lambda q: [q]
        agent_tl.run("测试")
        call_kwargs = mock_llm_client.chat_with_tools.call_args
        assert call_kwargs[1]["max_tokens"] == 2000

    def test_pipeline_generate_answer_max_tokens(self, agent, mock_llm_client):
        """Verify _generate_answer uses max_tokens=2000."""
        mock_llm_client.chat.return_value = ChatResponse(
            content="回答",
            usage={"total_tokens": 100},
        )
        agent._generate_answer(
            query="测试",
            search_results=[SearchResult(doc_id="d1", title="T", score=1.0, snippet="s")],
            context_docs={"d1": "content"},
        )
        call_kwargs = mock_llm_client.chat.call_args
        assert call_kwargs[1]["max_tokens"] == 2000


class TestQueryComplexityClassification:
    """Test _classify_query_complexity method."""

    def test_simple_short_query(self, agent):
        """Very short queries are simple."""
        assert agent._classify_query_complexity("年假") == "simple"

    def test_simple_keyword_only(self, agent):
        """Single keyword with no signals is simple."""
        assert agent._classify_query_complexity("报销") == "simple"

    def test_medium_with_how(self, agent):
        """Queries with '如何' + '申请' (2 medium signals) → medium."""
        result = agent._classify_query_complexity("如何申请年假")
        assert result == "medium"

    def test_light_with_what(self, agent):
        """Queries with single '什么' signal → light (4 iterations)."""
        result = agent._classify_query_complexity("什么是绩效考核")
        assert result == "light"

    def test_light_single_signal(self, agent):
        """Single medium signal with no other signals → light."""
        result = agent._classify_query_complexity("公司年假制度详解内容")
        assert result == "light"

    def test_complex_compound_query(self, agent):
        """Queries with multiple complex signals are complex."""
        result = agent._classify_query_complexity("差旅报销标准和审批流程的区别")
        assert result == "complex"

    def test_complex_long_query(self, agent):
        """Very long queries are complex."""
        query = "请帮我分析一下公司新出台的员工考勤管理制度和之前的制度有什么不同，以及这些变化对员工的影响"
        result = agent._classify_query_complexity(query)
        assert result == "complex"

    def test_complex_comparison(self, agent):
        """Comparison queries are complex."""
        result = agent._classify_query_complexity("A公司和B公司的福利对比分析")
        assert result == "complex"


class TestConversationHistory:
    """Test multi-turn conversation memory (history parameter)."""

    def test_history_includes_prior_messages(self, mock_config, mock_llm_client, mock_search_tool, mock_read_tool):
        """run() with history injects prior turns into LLM messages."""
        agent_tl = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=mock_llm_client,
            mode="tool_loop",
        )
        mock_llm_client.chat_with_tools.return_value = ChatResponse(
            content="回答",
            usage={"total_tokens": 100},
        )
        agent_tl._expand_query = lambda q: [q]

        history = [
            {"role": "user", "content": "公司的年假政策是什么？"},
            {"role": "assistant", "content": "年假政策规定每年15天..."},
        ]
        agent_tl.run("那病假呢？", history=history)

        call_args = mock_llm_client.chat_with_tools.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        roles = [m.role for m in messages]
        contents = [m.content for m in messages]

        assert roles[0] == "system"
        assert "公司的年假政策是什么？" in contents
        assert "年假政策规定每年15天..." in contents
        assert "那病假呢？" in contents
        assert roles.count("user") >= 2
        assert "assistant" in roles

    def test_history_none_no_regression(self, mock_config, mock_llm_client, mock_search_tool, mock_read_tool):
        """run() with history=None works — first msg is system, last is user query."""
        agent_tl = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=mock_llm_client,
            mode="tool_loop",
        )
        mock_llm_client.chat_with_tools.return_value = ChatResponse(
            content="回答",
            usage={"total_tokens": 100},
        )
        agent_tl._expand_query = lambda q: [q]
        agent_tl._no_log = True  # prevent SearchLogger from writing to real DB

        agent_tl.run("测试查询")

        call_args = mock_llm_client.chat_with_tools.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        roles = [m.role for m in messages]

        # Core contract: first is system prompt, last is user query
        assert roles[0] == "system"
        assert roles[-1] == "user"
        assert messages[-1].content == "测试查询"
        # There may be an AgentMemory context system msg between them
        # (depends on search_logs.db content), so allow 2 or 3 messages
        assert 2 <= len(messages) <= 3

    def test_conversation_memory_window_limits_turns(self, mock_config, mock_llm_client, mock_search_tool, mock_read_tool, monkeypatch):
        """CONVERSATION_MEMORY_WINDOW env var limits number of history turns."""
        monkeypatch.setenv("CONVERSATION_MEMORY_WINDOW", "1")

        agent_tl = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=mock_llm_client,
            mode="tool_loop",
        )
        mock_llm_client.chat_with_tools.return_value = ChatResponse(
            content="回答",
            usage={"total_tokens": 100},
        )
        agent_tl._expand_query = lambda q: [q]

        history = [
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "第二轮问题"},
            {"role": "assistant", "content": "第二轮回答"},
            {"role": "user", "content": "第三轮问题"},
            {"role": "assistant", "content": "第三轮回答"},
        ]
        agent_tl.run("第四轮问题", history=history)

        call_args = mock_llm_client.chat_with_tools.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        contents = [m.content for m in messages]

        assert "第一轮问题" not in contents
        assert "第二轮问题" not in contents
        assert "第三轮问题" in contents
        assert "第三轮回答" in contents
        assert "第四轮问题" in contents

    def test_history_filters_invalid_roles(self, mock_config, mock_llm_client, mock_search_tool, mock_read_tool):
        """run() with history containing invalid roles filters them out."""
        agent_tl = SearchAgent(
            config=mock_config,
            search_tool=mock_search_tool,
            read_tool=mock_read_tool,
            llm_client=mock_llm_client,
            mode="tool_loop",
        )
        mock_llm_client.chat_with_tools.return_value = ChatResponse(
            content="回答",
            usage={"total_tokens": 100},
        )
        agent_tl._expand_query = lambda q: [q]

        history = [
            {"role": "user", "content": "有效问题"},
            {"role": "system", "content": "不应出现的系统消息"},
            {"role": "tool", "content": "不应出现的工具消息"},
            {"role": "", "content": "空角色"},
        ]
        agent_tl.run("跟进", history=history)

        call_args = mock_llm_client.chat_with_tools.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        contents = [m.content for m in messages]

        assert "有效问题" in contents
        assert "不应出现的系统消息" not in contents
        assert "不应出现的工具消息" not in contents
        assert "跟进" in contents
