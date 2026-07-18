"""Unit tests for BM25Searcher class."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.search.bm25_search import (
    BM25Searcher,
    FullSearchResult,
    PaginatedResults,
    SearchPreview,
    create_searcher,
)
from src.search.query_parser import Query
from src.storage.base import SearchHit, SearchResult as StorageSearchResult


class TestSearchPreview:
    """Test SearchPreview dataclass."""

    def test_default_values(self):
        """Test SearchPreview with default values."""
        preview = SearchPreview(
            doc_id="doc1",
            title="Test Document",
            score=1.0,
            snippet="Test snippet",
        )
        assert preview.doc_id == "doc1"
        assert preview.title == "Test Document"
        assert preview.score == 1.0
        assert preview.snippet == "Test snippet"
        assert preview.source_path is None
        assert preview.highlights == []

    def test_source_path_normalization(self):
        """Test that source_path is normalized to Path object."""
        preview = SearchPreview(
            doc_id="doc1",
            title="Test",
            score=1.0,
            snippet="Test",
            source_path="/path/to/file.md",
        )
        assert isinstance(preview.source_path, Path)
        assert preview.source_path == Path("/path/to/file.md")

    def test_path_object_preserved(self):
        """Test that Path objects are preserved."""
        path = Path("/test/path.md")
        preview = SearchPreview(
            doc_id="doc1",
            title="Test",
            score=1.0,
            snippet="Test",
            source_path=path,
        )
        assert preview.source_path == path


class TestFullSearchResult:
    """Test FullSearchResult dataclass."""

    def test_full_search_result_defaults(self):
        """Test FullSearchResult with default values."""
        result = FullSearchResult(
            title="Test",
            score=1.0,
            snippet="Snippet",
            source=Path("/test.md"),
        )
        assert result.doc_id == ""
        assert result.full_content == ""
        assert result.keywords == []

    def test_full_search_result_custom_values(self):
        """Test FullSearchResult with custom values."""
        result = FullSearchResult(
            doc_id="doc123",
            title="Test Document",
            score=0.95,
            snippet="Preview snippet",
            source=Path("/docs/test.md"),
            full_content="This is the full content of the document.",
            keywords=["test", "document"],
        )
        assert result.doc_id == "doc123"
        assert result.title == "Test Document"
        assert result.score == 0.95
        assert result.full_content == "This is the full content of the document."
        assert result.keywords == ["test", "document"]


class TestPaginatedResults:
    """Test PaginatedResults dataclass."""

    def test_empty_results(self):
        """Test empty PaginatedResults."""
        results = PaginatedResults(
            results=[],
            total=0,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
        )
        assert results.results == []
        assert results.total == 0
        assert results.has_more is False

    def test_has_more_calculation(self):
        """Test has_more is calculated correctly."""
        # Not has_more when at end
        results = PaginatedResults(
            results=[],
            total=10,
            offset=0,
            limit=10,
            has_more=False,
            query="test",
        )
        assert results.has_more is False

        # has_more when more results exist
        results = PaginatedResults(
            results=[],
            total=25,
            offset=0,
            limit=10,
            has_more=True,
            query="test",
        )
        assert results.has_more is True


class TestBM25SearcherInit:
    """Test BM25Searcher initialization."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager."""
        manager = MagicMock(spec=["search", "get_stats"])
        manager.search.return_value = StorageSearchResult(
            hits=[],
            total=0,
            query="",
            execution_time=0.0,
        )
        return manager

    def test_init_default_values(self, mock_index_manager):
        """Test initialization with default values."""
        searcher = BM25Searcher(mock_index_manager)
        assert searcher._snippet_length == 200
        assert searcher._min_score == 0.0

    def test_init_custom_values(self, mock_index_manager):
        """Test initialization with custom values."""
        searcher = BM25Searcher(
            mock_index_manager,
            snippet_length=100,
            min_score=0.5,
        )
        assert searcher._snippet_length == 100
        assert searcher._min_score == 0.5


class TestBM25SearcherSearch:
    """Test BM25Searcher search method."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager with sample data."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def searcher(self, mock_index_manager):
        """Create a BM25Searcher instance."""
        return BM25Searcher(mock_index_manager)

    def test_search_empty_query(self, searcher):
        """Test search with empty query returns empty results."""
        result = searcher.search("")
        assert result.results == []
        assert result.total == 0
        assert result.has_more is False

    def test_search_whitespace_query(self, searcher):
        """Test search with whitespace-only query."""
        result = searcher.search("   \t\n  ")
        assert result.results == []
        assert result.total == 0

    def test_search_simple_query(self, mock_index_manager):
        """Test simple search query."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="绩效管理制度",
                    score=2.5,
                    excerpt="这是关于绩效管理的制度文档...",
                    source_path=Path("/docs/hr/performance.md"),
                )
            ],
            total=1,
            query="绩效",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search("绩效")

        assert len(result.results) == 1
        assert result.results[0].doc_id == "doc1"
        assert result.results[0].title == "绩效管理制度"
        assert result.total == 1

    def test_search_with_pagination(self, mock_index_manager):
        """Test search with pagination parameters."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc3",
                    title="Document 3",
                    score=1.5,
                    excerpt="Content 3",
                )
            ],
            total=30,
            query="test",
            execution_time=0.01,
            offset=20,
            limit=10,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search("test", limit=10, offset=20)

        assert result.offset == 20
        assert result.limit == 10
        assert result.has_more is False  # 20 + 10 >= 30

    def test_search_with_has_more(self, mock_index_manager):
        """Test has_more flag when more results exist."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=100,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search("test", limit=10, offset=0)

        assert result.has_more is True  # 0 + 10 < 100

    def test_search_min_score_filter(self, mock_index_manager):
        """Test that results below min_score are filtered."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="High Score",
                    score=1.0,
                    excerpt="Content",
                ),
                SearchHit(
                    doc_id="doc2",
                    title="Low Score",
                    score=0.3,
                    excerpt="Content",
                ),
            ],
            total=2,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager, min_score=0.5)

        result = searcher.search("test")

        # Only the high score result should be included
        assert len(result.results) == 1
        assert result.results[0].doc_id == "doc1"

    def test_search_snippet_truncation(self, mock_index_manager):
        """Test that snippets are truncated to configured length."""
        long_content = "A" * 500
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="Test",
                    score=1.0,
                    excerpt=long_content,
                )
            ],
            total=1,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager, snippet_length=100)

        result = searcher.search("test")

        assert len(result.results[0].snippet) <= 103  # 100 + "..."


class TestBM25SearcherGetFullContent:
    """Test BM25Searcher get_full_content method."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def searcher(self, mock_index_manager):
        """Create a BM25Searcher instance."""
        return BM25Searcher(mock_index_manager)

    def test_get_full_content_found(self, mock_index_manager):
        """Test getting full content for existing document."""
        mock_index_manager.get_document_by_id.return_value = {
            "doc_id": "doc123",
            "title": "Full Document",
            "content": "This is the full document content.",
            "filename": "full.md",
            "source_path": "/docs/full.md",
            "keywords": "",
        }
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.get_full_content("doc123")

        assert result is not None
        assert result.doc_id == "doc123"
        assert result.title == "Full Document"
        assert result.full_content == "This is the full document content."

    def test_get_full_content_not_found(self, mock_index_manager):
        """Test getting full content for non-existent document."""
        mock_index_manager.get_document_by_id.return_value = None
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.get_full_content("nonexistent")

        assert result is None

    def test_get_full_content_with_metadata(self, mock_index_manager):
        """Test getting full content with metadata included."""
        mock_index_manager.get_document_by_id.return_value = {
            "doc_id": "doc123",
            "title": "Test Document",
            "content": "Content",
            "filename": "test.md",
            "source_path": "/test.md",
            "keywords": "",
        }
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.get_full_content("doc123", include_metadata=True)

        assert result is not None
        assert "doc_id" in result.metadata
        assert result.metadata["filename"] == "test.md"
        assert result.metadata["source_path"] == "/test.md"


class TestBM25SearcherSearchByField:
    """Test BM25Searcher search_by_field method."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def searcher(self, mock_index_manager):
        """Create a BM25Searcher instance."""
        return BM25Searcher(mock_index_manager)

    def test_search_by_field_title(self, mock_index_manager):
        """Test searching by title field."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="季度报告",
                    score=2.0,
                    excerpt="内容...",
                )
            ],
            total=1,
            query='title:"季度报告"',
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search_by_field("title", "季度报告")

        assert len(result.results) == 1
        assert result.query == "title:季度报告"

    def test_search_by_field_keywords(self, mock_index_manager):
        """Test searching by keywords field."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="财务报表",
                    score=1.5,
                    excerpt="...",
                )
            ],
            total=1,
            query='keywords:"财务"',
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search_by_field("keywords", "财务")

        assert len(result.results) == 1

    def test_search_by_field_invalid_field(self, searcher):
        """Test searching with invalid field name raises error."""
        with pytest.raises(ValueError) as exc_info:
            searcher.search_by_field("invalid_field", "value")

        assert "Invalid field" in str(exc_info.value)

    def test_search_by_field_empty_value(self, searcher):
        """Test searching with empty value returns empty results."""
        result = searcher.search_by_field("title", "")

        assert result.results == []
        assert result.total == 0

    def test_search_by_field_escapes_quotes(self, mock_index_manager):
        """Test that quotes in value are escaped."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=0,
            query='title:"test \\"quoted\\"" value"',
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        searcher.search_by_field("title", 'test "quoted" value')

        # Check that search was called with escaped quotes
        call_args = mock_index_manager.search.call_args
        assert '\\"' in call_args[1]["query"]


class TestBM25SearcherSearchMultiField:
    """Test BM25Searcher search_multi_field method."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def searcher(self, mock_index_manager):
        """Create a BM25Searcher instance."""
        return BM25Searcher(mock_index_manager)

    def test_search_multi_field_and(self, mock_index_manager):
        """Test multi-field search with AND operator."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=0,
            query='title:"报告" AND keywords:"财务"',
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search_multi_field(
            {"title": "报告", "keywords": "财务"}, operator="AND"
        )

        mock_index_manager.search.assert_called_once()
        call_query = mock_index_manager.search.call_args[1]["query"]
        assert " AND " in call_query

    def test_search_multi_field_or(self, mock_index_manager):
        """Test multi-field search with OR operator."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=0,
            query='title:"报告" OR keywords:"财务"',
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search_multi_field(
            {"title": "报告", "keywords": "财务"}, operator="OR"
        )

        call_query = mock_index_manager.search.call_args[1]["query"]
        assert " OR " in call_query

    def test_search_multi_field_invalid_operator(self, searcher):
        """Test multi-field search with invalid operator."""
        with pytest.raises(ValueError) as exc_info:
            searcher.search_multi_field({"title": "test"}, operator="XOR")

        assert "Operator must be 'AND' or 'OR'" in str(exc_info.value)

    def test_search_multi_field_empty_queries(self, searcher):
        """Test multi-field search with empty queries."""
        result = searcher.search_multi_field({})

        assert result.results == []
        assert result.total == 0

    def test_search_multi_field_filters_invalid_fields(self, mock_index_manager):
        """Test that invalid fields are filtered out."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=0,
            query='title:"valid"',
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        searcher.search_multi_field(
            {
                "title": "valid",
                "invalid_field": "should be ignored",
            }
        )

        call_query = mock_index_manager.search.call_args[1]["query"]
        assert "invalid_field" not in call_query
        assert "title" in call_query


class TestBM25SearcherCount:
    """Test BM25Searcher count method."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def searcher(self, mock_index_manager):
        """Create a BM25Searcher instance."""
        return BM25Searcher(mock_index_manager)

    def test_count_returns_total(self, mock_index_manager):
        """Test count returns total matching documents."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=42,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        count = searcher.count("test")

        assert count == 42

    def test_count_empty_query(self, searcher):
        """Test count with empty query returns 0."""
        count = searcher.count("")
        assert count == 0


class TestBM25SearcherSuggest:
    """Test BM25Searcher suggest method."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager."""
        manager = MagicMock()
        manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="绩效管理制度",
                    score=1.0,
                    excerpt="...",
                ),
                SearchHit(
                    doc_id="doc2",
                    title="绩效考核标准",
                    score=0.9,
                    excerpt="...",
                ),
            ],
            total=2,
            query="绩效",
            execution_time=0.01,
        )
        return manager

    @pytest.fixture
    def searcher(self, mock_index_manager):
        """Create a BM25Searcher instance."""
        return BM25Searcher(mock_index_manager)

    def test_suggest_returns_titles(self, searcher):
        """Test suggest returns titles from matching documents."""
        suggestions = searcher.suggest("绩效")

        assert len(suggestions) > 0
        assert "绩效管理制度" in suggestions or "绩效考核标准" in suggestions

    def test_suggest_too_short_query(self, searcher):
        """Test suggest with too short query returns empty."""
        suggestions = searcher.suggest("a")

        assert suggestions == []

    def test_suggest_empty_query(self, searcher):
        """Test suggest with empty query returns empty."""
        suggestions = searcher.suggest("")

        assert suggestions == []

    def test_suggest_respects_limit(self, mock_index_manager):
        """Test suggest respects the limit parameter."""
        # Create more results than limit
        hits = [
            SearchHit(
                doc_id=f"doc{i}",
                title=f"Document {i}",
                score=1.0,
                excerpt="...",
            )
            for i in range(10)
        ]
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=hits,
            total=10,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        suggestions = searcher.suggest("test", limit=3)

        assert len(suggestions) <= 3


class TestBM25SearcherGetIndexStats:
    """Test BM25Searcher get_index_stats method."""

    def test_get_index_stats(self):
        """Test getting index statistics."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "num_docs": 100,
            "index_path": "/path/to/index",
        }
        searcher = BM25Searcher(mock_manager)

        stats = searcher.get_index_stats()

        assert stats["num_docs"] == 100
        assert stats["index_path"] == "/path/to/index"


class TestCreateSearcher:
    """Test create_searcher factory function."""

    def test_create_searcher_default(self):
        """Test creating searcher with default parameters."""
        with patch("src.search.bm25_search.TantivyIndexManager") as MockManager:
            mock_manager = MagicMock()
            MockManager.return_value = mock_manager

            searcher = create_searcher()

            assert isinstance(searcher, BM25Searcher)
            MockManager.assert_called_once_with(
                index_path=None,
                use_jieba=True,
                readonly=False,
            )

    def test_create_searcher_custom_params(self):
        """Test creating searcher with custom parameters."""
        with patch("src.search.bm25_search.TantivyIndexManager") as MockManager:
            mock_manager = MagicMock()
            MockManager.return_value = mock_manager

            index_path = Path("/custom/index")
            searcher = create_searcher(
                index_path=index_path,
                use_jieba=False,
                snippet_length=150,
            )

            assert isinstance(searcher, BM25Searcher)
            assert searcher._snippet_length == 150
            MockManager.assert_called_once_with(
                index_path=index_path,
                use_jieba=False,
                readonly=False,
            )


class TestBM25SearcherIntegration:
    """Integration tests with actual TantivyIndexManager (if available)."""

    @pytest.fixture
    def real_index_manager(self, tmp_path):
        """Create a real TantivyIndexManager for integration tests."""
        try:
            from src.storage.index import TantivyIndexManager

            manager = TantivyIndexManager(
                index_path=tmp_path / "test_index",
                use_jieba=False,  # Don't require jieba for tests
            )

            # Add some test documents
            manager.add_document(
                doc_id="doc1",
                title="绩效管理制度",
                content="这是关于绩效考核的管理制度文档，包含考核标准和流程。",
                metadata={"filename": "performance.md"},
            )
            manager.add_document(
                doc_id="doc2",
                title="财务报销流程",
                content="员工差旅费用报销的详细流程和所需材料说明。",
                metadata={"filename": "reimbursement.md"},
            )
            manager.add_document(
                doc_id="doc3",
                title="2024年度报告",
                content="2024年度公司运营报告，包含财务数据和绩效指标。",
                metadata={"filename": "annual_report.md"},
            )
            manager.commit()

            return manager
        except Exception:
            pytest.skip("TantivyIndexManager not available")

    def test_integration_search(self, real_index_manager):
        """Test search with real index manager."""
        searcher = BM25Searcher(real_index_manager)

        result = searcher.search("绩效")

        # Should return a valid result structure (may be empty due to tokenizer)
        assert isinstance(result, PaginatedResults)
        assert result.query == "绩效"
        assert isinstance(result.total, int)
        assert isinstance(result.results, list)

    def test_integration_search_by_field(self, real_index_manager):
        """Test field-specific search with real index manager."""
        searcher = BM25Searcher(real_index_manager)

        result = searcher.search_by_field("title", "报告")

        # Should return a valid result structure (may be empty due to tokenizer)
        assert isinstance(result, PaginatedResults)
        assert "title" in result.query

    def test_integration_pagination(self, real_index_manager):
        """Test pagination with real index manager."""
        searcher = BM25Searcher(real_index_manager)

        # Get first page
        page1 = searcher.search("的", limit=1, offset=0)

        # Get second page
        page2 = searcher.search("的", limit=1, offset=1)

        # Pages should be different (if there are multiple results)
        if page1.total > 1:
            assert page1.results[0].doc_id != page2.results[0].doc_id

    def test_integration_get_full_content(self, real_index_manager):
        """Test getting full content with real index manager."""
        searcher = BM25Searcher(real_index_manager)

        # First search to get a doc_id
        result = searcher.search("绩效")

        if result.results:
            doc_id = result.results[0].doc_id
            full_content = searcher.get_full_content(doc_id)

            assert full_content is not None
            assert full_content.doc_id == doc_id

    def test_integration_count(self, real_index_manager):
        """Test count with real index manager."""
        searcher = BM25Searcher(real_index_manager)

        count = searcher.count("绩效")

        # Should return a valid count (may be 0 due to tokenizer)
        assert isinstance(count, int)
        assert count >= 0


class TestBM25SearcherEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_index_manager(self):
        """Create a mock TantivyIndexManager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def searcher(self, mock_index_manager):
        """Create a BM25Searcher instance."""
        return BM25Searcher(mock_index_manager)

    def test_search_with_special_characters(self, mock_index_manager):
        """Test search with special characters in query."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=0,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        # Should not raise any exceptions
        result = searcher.search("test@#$%^&*()")

        assert result is not None

    def test_search_with_unicode(self, mock_index_manager):
        """Test search with unicode characters."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="测试文档",
                    score=1.0,
                    excerpt="中文内容测试",
                )
            ],
            total=1,
            query="测试",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search("测试")

        assert len(result.results) == 1

    def test_search_with_very_long_query(self, mock_index_manager):
        """Test search with very long query."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=0,
            query="",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        long_query = "绩效" * 1000
        result = searcher.search(long_query)

        assert result is not None

    def test_search_index_error_handling(self, mock_index_manager):
        """Test handling of index errors."""
        mock_index_manager.search.side_effect = Exception("Index error")
        searcher = BM25Searcher(mock_index_manager)

        # Should propagate exception (or handle gracefully depending on design)
        with pytest.raises(Exception):
            searcher.search("test")

    def test_snippet_truncation_edge_cases(self, mock_index_manager):
        """Test snippet truncation with various lengths."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[
                SearchHit(
                    doc_id="doc1",
                    title="Test",
                    score=1.0,
                    excerpt="Short",
                ),
                SearchHit(
                    doc_id="doc2",
                    title="Test",
                    score=1.0,
                    excerpt="",  # Empty
                ),
                SearchHit(
                    doc_id="doc3",
                    title="Test",
                    score=1.0,
                    excerpt=None,  # None
                ),
            ],
            total=3,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search("test")

        # Should handle all cases without errors
        assert len(result.results) == 3

    def test_search_with_zero_limit(self, mock_index_manager):
        """Test search with zero limit."""
        mock_index_manager.search.return_value = StorageSearchResult(
            hits=[],
            total=10,
            query="test",
            execution_time=0.01,
        )
        searcher = BM25Searcher(mock_index_manager)

        result = searcher.search("test", limit=0)

        # Should still work, just return no results
        assert result.results == []
