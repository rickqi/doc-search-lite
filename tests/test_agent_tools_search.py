"""Unit tests for SearchTool class.

Tests cover:
- SearchTool initialization and properties
- Execute method with various parameters
- Result formatting
- OpenAI tool format generation
- Error handling
- Edge cases
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.tool_types import ToolResult
from src.agent.tools.search import SearchTool, create_search_tool
from src.search.bm25_search import PaginatedResults, SearchPreview


class TestSearchToolInit:
    """Test SearchTool initialization."""

    @pytest.fixture
    def mock_searcher(self):
        """Create a mock BM25Searcher."""
        return MagicMock()

    def test_init_default_values(self, mock_searcher):
        """Test initialization with default values."""
        tool = SearchTool(mock_searcher)
        assert tool._searcher is mock_searcher
        assert tool._default_limit == 10
        assert tool._snippet_length == 200

    def test_init_custom_values(self, mock_searcher):
        """Test initialization with custom values."""
        tool = SearchTool(
            mock_searcher,
            default_limit=20,
            snippet_length=150,
        )
        assert tool._default_limit == 20
        assert tool._snippet_length == 150


class TestSearchToolProperties:
    """Test SearchTool properties."""

    @pytest.fixture
    def tool(self):
        """Create a SearchTool instance."""
        return SearchTool(MagicMock())

    def test_name_property(self, tool):
        """Test name property returns 'search'."""
        assert tool.name == "search"

    def test_description_property(self, tool):
        """Test description property returns non-empty string."""
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0
        assert "搜索" in tool.description or "BM25" in tool.description


class TestSearchToolExecute:
    """Test SearchTool execute method."""

    @pytest.fixture
    def mock_searcher(self):
        """Create a mock BM25Searcher."""
        return MagicMock()

    @pytest.fixture
    def tool(self, mock_searcher):
        """Create a SearchTool instance."""
        return SearchTool(mock_searcher)

    def test_execute_simple_query(self, mock_searcher):
        """Test simple search query execution."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[
                SearchPreview(
                    doc_id="doc1",
                    title="Test Document",
                    score=1.5,
                    snippet="This is a test snippet",
                    source_path=Path("/test.md"),
                    highlights=["test"],
                )
            ],
            total=1,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, str)

        # Parse JSON data
        data = json.loads(result.data)
        assert data["total"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["doc_id"] == "doc1"

    def test_execute_with_limit_and_offset(self, mock_searcher):
        """Test execution with custom limit and offset."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=100,
            offset=20,
            limit=10,
            has_more=True,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test", limit=10, offset=20)

        assert result.success is True
        mock_searcher.search.assert_called_once_with(
            query="test",
            limit=10,
            offset=20,
        )

        # Check metadata
        assert result.metadata["limit"] == 10
        assert result.metadata["offset"] == 20
        assert result.metadata["has_more"] is True

    def test_execute_empty_query_fails(self, tool):
        """Test that empty query returns failure."""
        result = tool.execute(query="")

        assert result.success is False
        assert result.error is not None
        assert "required" in result.error.lower()
        assert "query" in result.error.lower()

    def test_execute_missing_query_fails(self, tool):
        """Test that missing query parameter returns failure."""
        result = tool.execute(limit=10)

        assert result.success is False
        assert result.error is not None
        assert "required" in result.error.lower()

    def test_execute_invalid_query_type_fails(self, tool):
        """Test that non-string query returns failure."""
        result = tool.execute(query=123)

        assert result.success is False
        assert result.error is not None
        assert "string" in result.error.lower()

    def test_execute_invalid_limit_fails(self, tool):
        """Test that invalid limit returns failure."""
        result = tool.execute(query="test", limit=-1)

        assert result.success is False
        assert result.error is not None
        assert "limit" in result.error.lower()

    def test_execute_invalid_limit_type_fails(self, tool):
        """Test that non-integer limit returns failure."""
        result = tool.execute(query="test", limit="ten")

        assert result.success is False
        assert result.error is not None
        assert "limit" in result.error.lower()

    def test_execute_invalid_offset_fails(self, tool):
        """Test that invalid offset returns failure."""
        result = tool.execute(query="test", offset=-5)

        assert result.success is False
        assert result.error is not None
        assert "offset" in result.error.lower()

    def test_execute_uses_default_limit(self, mock_searcher):
        """Test that default limit is used when not specified."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=20,
            has_more=False,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher, default_limit=20)

        result = tool.execute(query="test")

        assert result.success is True
        mock_searcher.search.assert_called_once()
        call_args = mock_searcher.search.call_args
        assert call_args[1]["limit"] == 20

    def test_execute_handles_searcher_exception(self, mock_searcher):
        """Test that searcher exceptions are handled gracefully."""
        mock_searcher.search.side_effect = Exception("Index error")
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test")

        assert result.success is False
        assert result.error is not None
        assert "Search failed" in result.error
        assert "Index error" in result.error
        assert result.metadata["error_type"] == "Exception"


class TestSearchToolResultFormatting:
    """Test SearchTool result formatting."""

    @pytest.fixture
    def mock_searcher(self):
        """Create a mock BM25Searcher."""
        return MagicMock()

    def test_format_results_structure(self, mock_searcher):
        """Test that formatted results have correct structure."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[
                SearchPreview(
                    doc_id="doc1",
                    title="绩效管理制度",
                    score=2.5,
                    snippet="这是关于绩效考核的制度...",
                    source_path=Path("/docs/performance.md"),
                    highlights=["绩效", "考核"],
                )
            ],
            total=1,
            offset=0,
            limit=10,
            has_more=False,
            query="绩效",
            execution_time=0.05,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="绩效")

        data = json.loads(result.data)

        # Check top-level structure
        assert "query" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data
        assert "has_more" in data
        assert "results" in data

        # Check result item structure
        result_item = data["results"][0]
        assert "doc_id" in result_item
        assert "title" in result_item
        assert "score" in result_item
        assert "snippet" in result_item
        assert "source_path" in result_item
        assert "highlights" in result_item

    def test_format_results_with_chinese(self, mock_searcher):
        """Test formatting with Chinese content."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[
                SearchPreview(
                    doc_id="doc1",
                    title="中文标题测试",
                    score=1.0,
                    snippet="这是一段中文内容片段",
                    source_path=None,
                    highlights=[],
                )
            ],
            total=1,
            offset=0,
            limit=10,
            has_more=False,
            query="中文",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="中文")

        # Ensure Chinese characters are preserved
        data = json.loads(result.data)
        assert data["results"][0]["title"] == "中文标题测试"
        assert data["results"][0]["snippet"] == "这是一段中文内容片段"

    def test_format_results_multiple_items(self, mock_searcher):
        """Test formatting multiple results."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[
                SearchPreview(
                    doc_id=f"doc{i}",
                    title=f"Document {i}",
                    score=float(i),
                    snippet=f"Snippet {i}",
                    source_path=None,
                    highlights=[],
                )
                for i in range(5)
            ],
            total=5,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.02,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test")

        data = json.loads(result.data)
        assert len(data["results"]) == 5
        assert data["total"] == 5

    def test_format_results_empty(self, mock_searcher):
        """Test formatting empty results."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=10,
            has_more=False,
            query="nonexistent",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="nonexistent")

        data = json.loads(result.data)
        assert data["results"] == []
        assert data["total"] == 0

    def test_format_results_null_source_path(self, mock_searcher):
        """Test formatting with null source_path."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[
                SearchPreview(
                    doc_id="doc1",
                    title="Test",
                    score=1.0,
                    snippet="Content",
                    source_path=None,
                    highlights=[],
                )
            ],
            total=1,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test")

        data = json.loads(result.data)
        assert data["results"][0]["source_path"] is None

    def test_score_rounding(self, mock_searcher):
        """Test that scores are rounded to 4 decimal places."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[
                SearchPreview(
                    doc_id="doc1",
                    title="Test",
                    score=1.23456789,
                    snippet="Content",
                    source_path=None,
                    highlights=[],
                )
            ],
            total=1,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test")

        data = json.loads(result.data)
        assert data["results"][0]["score"] == 1.2346  # Rounded to 4 decimals


class TestSearchToolMetadata:
    """Test SearchTool metadata in results."""

    @pytest.fixture
    def mock_searcher(self):
        """Create a mock BM25Searcher."""
        return MagicMock()

    def test_metadata_contains_query(self, mock_searcher):
        """Test that metadata contains query."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=10,
            has_more=False,
            query="test query",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test query")

        assert result.metadata["query"] == "test query"

    def test_metadata_contains_pagination(self, mock_searcher):
        """Test that metadata contains pagination info."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=100,
            offset=20,
            limit=10,
            has_more=True,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test", limit=10, offset=20)

        assert result.metadata["total_results"] == 100
        assert result.metadata["returned_count"] == 0
        assert result.metadata["offset"] == 20
        assert result.metadata["limit"] == 10
        assert result.metadata["has_more"] is True

    def test_metadata_contains_timing(self, mock_searcher):
        """Test that metadata contains timing info."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.05,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test")

        assert "execution_time" in result.metadata
        assert "search_time" in result.metadata
        assert result.metadata["search_time"] == 0.05


class TestSearchToolOpenAIFormat:
    """Test SearchTool OpenAI format generation."""

    @pytest.fixture
    def tool(self):
        """Create a SearchTool instance."""
        return SearchTool(MagicMock())

    def test_to_openai_tool_structure(self, tool):
        """Test to_openai_tool returns correct structure."""
        openai_tool = tool.to_openai_tool()

        assert openai_tool["type"] == "function"
        assert "function" in openai_tool
        assert openai_tool["function"]["name"] == "search"
        assert openai_tool["function"]["description"] == tool.description

    def test_to_openai_tool_parameters(self, tool):
        """Test to_openai_tool has correct parameters schema."""
        openai_tool = tool.to_openai_tool()
        params = openai_tool["function"]["parameters"]

        assert params["type"] == "object"
        assert "properties" in params

        # Check query parameter
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert "query" in params["required"]

        # Check limit parameter
        assert "limit" in params["properties"]
        assert params["properties"]["limit"]["type"] == "integer"
        assert params["properties"]["limit"]["default"] == 10

        # Check offset parameter
        assert "offset" in params["properties"]
        assert params["properties"]["offset"]["type"] == "integer"
        assert params["properties"]["offset"]["default"] == 0

    def test_to_openai_tool_parameter_descriptions(self, tool):
        """Test that parameter descriptions are present."""
        openai_tool = tool.to_openai_tool()
        props = openai_tool["function"]["parameters"]["properties"]

        assert "description" in props["query"]
        assert "description" in props["limit"]
        assert "description" in props["offset"]


class TestCreateSearchTool:
    """Test create_search_tool factory function."""

    def test_create_search_tool_default(self):
        """Test creating search tool with default parameters."""
        with patch("src.search.bm25_search.create_searcher") as mock_create_searcher:
            mock_searcher = MagicMock()
            mock_create_searcher.return_value = mock_searcher

            tool = create_search_tool()

            assert isinstance(tool, SearchTool)
            assert tool._default_limit == 10
            mock_create_searcher.assert_called_once()

    def test_create_search_tool_custom_params(self):
        """Test creating search tool with custom parameters."""
        with patch("src.search.bm25_search.create_searcher") as mock_create_searcher:
            mock_searcher = MagicMock()
            mock_create_searcher.return_value = mock_searcher

            index_path = Path("/custom/index")
            tool = create_search_tool(
                index_path=index_path,
                use_jieba=False,
                default_limit=20,
                snippet_length=150,
            )

            assert isinstance(tool, SearchTool)
            assert tool._default_limit == 20

            mock_create_searcher.assert_called_once_with(
                index_path=index_path,
                use_jieba=False,
                snippet_length=150,
            )


class TestSearchToolEdgeCases:
    """Test edge cases for SearchTool."""

    @pytest.fixture
    def mock_searcher(self):
        """Create a mock BM25Searcher."""
        return MagicMock()

    @pytest.fixture
    def tool(self, mock_searcher):
        """Create a SearchTool instance."""
        return SearchTool(mock_searcher)

    def test_execute_whitespace_only_query(self, tool):
        """Test with whitespace-only query."""
        result = tool.execute(query="   ")

        assert result.success is False
        assert result.error is not None
        assert "required" in result.error.lower()

    def test_execute_with_zero_limit(self, mock_searcher):
        """Test with zero limit."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=10,
            offset=0,
            limit=0,
            has_more=True,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test", limit=0)

        assert result.success is True
        mock_searcher.search.assert_called_once_with(
            query="test",
            limit=0,
            offset=0,
        )

    def test_execute_with_large_offset(self, mock_searcher):
        """Test with large offset."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=100,
            offset=1000,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="test", offset=1000)

        assert result.success is True

    def test_execute_with_special_characters_in_query(self, mock_searcher):
        """Test with special characters in query."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=10,
            has_more=False,
            query='test@#$%^&*()"',
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query='test@#$%^&*()"')

        assert result.success is True

    def test_execute_with_unicode_query(self, mock_searcher):
        """Test with unicode characters in query."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[
                SearchPreview(
                    doc_id="doc1",
                    title="测试文档 🚀",
                    score=1.0,
                    snippet="包含emoji 📝的内容",
                    source_path=None,
                    highlights=[],
                )
            ],
            total=1,
            offset=0,
            limit=10,
            has_more=False,
            query="测试 🚀",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query="测试 🚀")

        assert result.success is True
        data = json.loads(result.data)
        assert "🚀" in data["results"][0]["title"]

    def test_execute_with_very_long_query(self, mock_searcher):
        """Test with very long query string."""
        long_query = "绩效" * 1000
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=10,
            has_more=False,
            query=long_query,
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        result = tool.execute(query=long_query)

        assert result.success is True

    def test_execute_extra_parameters_ignored(self, mock_searcher):
        """Test that extra parameters are ignored."""
        mock_searcher.search.return_value = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
            execution_time=0.01,
        )
        tool = SearchTool(mock_searcher)

        # Should not raise an error
        result = tool.execute(
            query="test",
            limit=10,
            extra_param="should be ignored",
        )

        assert result.success is True


class TestSearchToolIntegration:
    """Integration tests with real BM25Searcher (if available)."""

    @pytest.fixture
    def real_searcher(self, tmp_path):
        """Create a real BM25Searcher for integration tests."""
        try:
            from src.search.bm25_search import BM25Searcher
            from src.storage.index import TantivyIndexManager

            manager = TantivyIndexManager(
                index_path=tmp_path / "test_index",
                use_jieba=False,
            )

            # Add test documents
            manager.add_document(
                doc_id="doc1",
                title="绩效管理制度",
                content="这是关于绩效考核的管理制度文档。",
                metadata={"filename": "performance.md"},
            )
            manager.add_document(
                doc_id="doc2",
                title="财务报销流程",
                content="员工差旅费用报销的详细流程。",
                metadata={"filename": "reimbursement.md"},
            )
            manager.commit()

            return BM25Searcher(manager)
        except Exception:
            pytest.skip("TantivyIndexManager not available")

    def test_integration_search(self, real_searcher):
        """Test search with real BM25Searcher."""
        tool = SearchTool(real_searcher)

        result = tool.execute(query="绩效", limit=5)

        assert result.success is True
        assert result.metadata is not None
        assert "total_results" in result.metadata

    def test_integration_pagination(self, real_searcher):
        """Test pagination with real BM25Searcher."""
        tool = SearchTool(real_searcher)

        result = tool.execute(query="流程", limit=1, offset=0)

        assert result.success is True
        assert result.metadata["limit"] == 1
        assert result.metadata["offset"] == 0

    def test_integration_empty_results(self, real_searcher):
        """Test with query that returns no results."""
        tool = SearchTool(real_searcher)

        result = tool.execute(query="nonexistent_document_xyz123")

        assert result.success is True
        data = json.loads(result.data)
        assert data["results"] == []
        assert data["total"] == 0


class TestSearchToolProtocol:
    """Test that SearchTool properly implements Tool protocol."""

    def test_implements_name_property(self):
        """Test that SearchTool has name property."""
        tool = SearchTool(MagicMock())
        assert hasattr(tool, "name")
        assert callable(lambda: tool.name)

    def test_implements_description_property(self):
        """Test that SearchTool has description property."""
        tool = SearchTool(MagicMock())
        assert hasattr(tool, "description")
        assert callable(lambda: tool.description)

    def test_implements_execute_method(self):
        """Test that SearchTool has execute method."""
        tool = SearchTool(MagicMock())
        assert hasattr(tool, "execute")
        assert callable(tool.execute)

    def test_implements_to_openai_tool_method(self):
        """Test that SearchTool has to_openai_tool method."""
        tool = SearchTool(MagicMock())
        assert hasattr(tool, "to_openai_tool")
        assert callable(tool.to_openai_tool)

    def test_execute_returns_tool_result(self):
        """Test that execute returns ToolResult instance."""
        tool = SearchTool(MagicMock())
        result = tool.execute(query="test")
        assert isinstance(result, ToolResult)
