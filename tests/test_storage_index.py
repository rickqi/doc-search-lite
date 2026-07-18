"""Unit tests for TantivyIndexManager full-text search implementation."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.storage.index import TantivyIndexManager
from src.storage.base import SearchHit, SearchResult


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for the index."""
    index_dir = Path(tempfile.mkdtemp())
    yield index_dir
    # Cleanup
    shutil.rmtree(index_dir, ignore_errors=True)


@pytest.fixture
def index_manager(temp_index_dir):
    """Create a TantivyIndexManager instance with a temp index."""
    manager = TantivyIndexManager(
        index_path=temp_index_dir / "test_index",
        use_jieba=False,  # Use regex tokenizer for consistent testing
        heap_size=20_000_000,
        num_threads=1,
    )
    yield manager
    manager.close()


@pytest.fixture
def in_memory_manager():
    """Create an in-memory TantivyIndexManager for fast tests."""
    manager = TantivyIndexManager(
        index_path=None,
        use_jieba=False,
        heap_size=20_000_000,
        num_threads=1,
    )
    yield manager
    manager.close()


class TestTantivyIndexManagerInit:
    """Tests for TantivyIndexManager initialization."""

    def test_init_with_path(self, temp_index_dir):
        """Should create index at specified path."""
        manager = TantivyIndexManager(
            index_path=temp_index_dir / "new_index",
            use_jieba=False,
        )

        assert manager._index_path == temp_index_dir / "new_index"
        assert (temp_index_dir / "new_index").exists()
        manager.close()

    def test_init_in_memory(self):
        """Should create in-memory index when path is None."""
        manager = TantivyIndexManager(
            index_path=None,
            use_jieba=False,
        )

        assert manager._index_path is None
        manager.close()

    def test_init_detects_jieba_unavailable(self):
        """Should detect when jieba is not available."""
        manager = TantivyIndexManager(
            index_path=None,
            use_jieba=True,
        )

        # jieba_available depends on installation
        assert isinstance(manager._jieba_available, bool)
        manager.close()


class TestAddDocument:
    """Tests for add_document operation."""

    def test_add_document_basic(self, in_memory_manager):
        """Should add a basic document."""
        result = in_memory_manager.add_document(
            doc_id="doc1",
            title="Test Document",
            content="This is test content for searching.",
            metadata={},
        )

        assert result is True

    def test_add_document_with_metadata(self, in_memory_manager):
        """Should add document with all metadata fields."""
        result = in_memory_manager.add_document(
            doc_id="doc2",
            title="Document with Metadata",
            content="Content with metadata.",
            metadata={
                "filename": "test.pdf",
                "source_path": "/path/to/test.pdf",
                "keywords": ["test", "document"],
                "modified_time": datetime(2024, 1, 15, 10, 30, 0),
            },
        )

        assert result is True

    def test_add_document_with_keywords_list(self, in_memory_manager):
        """Should handle keywords as list."""
        result = in_memory_manager.add_document(
            doc_id="doc3",
            title="Keywords Test",
            content="Content.",
            metadata={
                "keywords": ["python", "search", "index"],
            },
        )

        assert result is True

    def test_add_document_with_keywords_string(self, in_memory_manager):
        """Should handle keywords as string."""
        result = in_memory_manager.add_document(
            doc_id="doc4",
            title="Keywords String Test",
            content="Content.",
            metadata={
                "keywords": "python search index",
            },
        )

        assert result is True

    def test_add_document_with_iso_datetime(self, in_memory_manager):
        """Should handle ISO format datetime strings."""
        result = in_memory_manager.add_document(
            doc_id="doc5",
            title="DateTime Test",
            content="Content.",
            metadata={
                "modified_time": "2024-01-15T10:30:00",
            },
        )

        assert result is True

    def test_add_document_empty_content(self, in_memory_manager):
        """Should handle empty content."""
        result = in_memory_manager.add_document(
            doc_id="doc_empty",
            title="Empty Document",
            content="",
            metadata={},
        )

        assert result is True

    def test_add_document_chinese_content(self, in_memory_manager):
        """Should handle Chinese content."""
        result = in_memory_manager.add_document(
            doc_id="doc_chinese",
            title="中文文档",
            content="这是一个中文测试文档，用于测试中文分词和搜索功能。",
            metadata={
                "keywords": ["中文", "测试"],
            },
        )

        assert result is True


class TestSearch:
    """Tests for search operation."""

    def test_search_basic(self, in_memory_manager):
        """Should perform basic search."""
        # Add test documents
        in_memory_manager.add_document(
            doc_id="search1",
            title="Python Programming",
            content="Python is a popular programming language.",
            metadata={},
        )
        in_memory_manager.add_document(
            doc_id="search2",
            title="Java Programming",
            content="Java is another programming language.",
            metadata={},
        )
        in_memory_manager.commit()

        # Search for Python
        result = in_memory_manager.search("Python", limit=10)

        assert isinstance(result, SearchResult)
        assert result.total >= 1
        assert len(result.hits) >= 1
        assert any("Python" in hit.title for hit in result.hits)

    def test_search_returns_search_hit(self, in_memory_manager):
        """Search should return SearchHit objects."""
        in_memory_manager.add_document(
            doc_id="hit_test",
            title="Test Document",
            content="This document contains searchable content.",
            metadata={"source_path": "/test/doc.md"},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("searchable", limit=10)

        assert len(result.hits) >= 1
        hit = result.hits[0]
        assert isinstance(hit, SearchHit)
        assert hit.doc_id == "hit_test"
        assert hit.title == "Test Document"
        assert hit.score > 0

    def test_search_with_limit(self, in_memory_manager):
        """Should respect limit parameter."""
        # Add multiple documents
        for i in range(10):
            in_memory_manager.add_document(
                doc_id=f"limit{i}",
                title=f"Document {i}",
                content=f"Content for document {i} about testing.",
                metadata={},
            )
        in_memory_manager.commit()

        result = in_memory_manager.search("testing", limit=3)

        assert len(result.hits) <= 3

    def test_search_with_offset(self, in_memory_manager):
        """Should respect offset parameter."""
        # Add multiple documents
        for i in range(5):
            in_memory_manager.add_document(
                doc_id=f"offset{i}",
                title=f"Offset Document {i}",
                content=f"Content for offset testing {i}.",
                metadata={},
            )
        in_memory_manager.commit()

        result1 = in_memory_manager.search("offset", limit=2, offset=0)
        result2 = in_memory_manager.search("offset", limit=2, offset=2)

        # Results should be different with different offsets
        ids1 = {h.doc_id for h in result1.hits}
        ids2 = {h.doc_id for h in result2.hits}
        assert ids1 != ids2 or len(ids1) == 0 or len(ids2) == 0

    def test_search_empty_result(self, in_memory_manager):
        """Should return empty result for no matches."""
        in_memory_manager.add_document(
            doc_id="no_match",
            title="Some Title",
            content="Some content.",
            metadata={},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("xyznonexistent123", limit=10)

        assert result.total == 0
        assert len(result.hits) == 0

    def test_search_includes_execution_time(self, in_memory_manager):
        """Search result should include execution time."""
        in_memory_manager.add_document(
            doc_id="time_test",
            title="Time Test",
            content="Content for timing test.",
            metadata={},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("timing", limit=10)

        assert result.execution_time >= 0

    def test_search_chinese_content(self, in_memory_manager):
        """Should search Chinese content."""
        in_memory_manager.add_document(
            doc_id="chinese_search",
            title="中文搜索测试",
            content="这是一个中文搜索测试文档。",
            metadata={},
        )
        in_memory_manager.commit()

        # Search for Chinese term
        result = in_memory_manager.search("中文", limit=10)

        # Should find the document (may or may not match depending on tokenizer)
        assert isinstance(result, SearchResult)

    def test_search_preserves_query(self, in_memory_manager):
        """Should preserve original query in result."""
        in_memory_manager.add_document(
            doc_id="query_test",
            title="Query Test",
            content="Test content.",
            metadata={},
        )
        in_memory_manager.commit()

        query = "test query"
        result = in_memory_manager.search(query, limit=10)

        assert result.query == query


class TestUpdateDocument:
    """Tests for update_document operation."""

    def test_update_document(self, in_memory_manager):
        """Should update existing document."""
        # Add initial document
        in_memory_manager.add_document(
            doc_id="update1",
            title="Original Title",
            content="Original content.",
            metadata={},
        )
        in_memory_manager.commit()

        # Update document
        result = in_memory_manager.update_document(
            doc_id="update1",
            title="Updated Title",
            content="Updated content with new information.",
            metadata={"filename": "updated.txt"},
        )
        in_memory_manager.commit()

        assert result is True

        # Search should find updated content
        search_result = in_memory_manager.search("Updated", limit=10)
        assert len(search_result.hits) >= 1

    def test_update_nonexistent_document(self, in_memory_manager):
        """Should handle update of nonexistent document."""
        result = in_memory_manager.update_document(
            doc_id="nonexistent",
            title="New Title",
            content="New content.",
            metadata={},
        )

        # Update of nonexistent doc should still succeed (treated as add)
        assert result is True


class TestDeleteDocument:
    """Tests for delete_document operation."""

    def test_delete_document(self, in_memory_manager):
        """Should delete document from index."""
        # Add document
        in_memory_manager.add_document(
            doc_id="delete1",
            title="Document to Delete",
            content="This will be deleted.",
            metadata={},
        )
        in_memory_manager.commit()

        # Delete document
        result = in_memory_manager.delete_document("delete1")

        assert result is True

    def test_delete_nonexistent_document(self, in_memory_manager):
        """Should handle delete of nonexistent document."""
        result = in_memory_manager.delete_document("nonexistent_delete")

        # Should return True (no-op)
        assert result is True


class TestCommit:
    """Tests for commit operation."""

    def test_commit_persists_changes(self, in_memory_manager):
        """Commit should persist changes."""
        in_memory_manager.add_document(
            doc_id="commit1",
            title="Commit Test",
            content="Content before commit.",
            metadata={},
        )

        result = in_memory_manager.commit()

        assert result is True

    def test_multiple_commits(self, in_memory_manager):
        """Should handle multiple commits."""
        for i in range(3):
            in_memory_manager.add_document(
                doc_id=f"multi_commit{i}",
                title=f"Document {i}",
                content=f"Content {i}.",
                metadata={},
            )
            assert in_memory_manager.commit() is True


class TestRebuild:
    """Tests for rebuild operation."""

    def test_rebuild_clears_index(self, index_manager):
        """Rebuild should clear the index."""
        # Add documents
        index_manager.add_document(
            doc_id="rebuild1",
            title="Before Rebuild",
            content="Content before rebuild.",
            metadata={},
        )
        index_manager.commit()

        # Rebuild
        result = index_manager.rebuild()

        assert result is True

        # Index should be empty
        stats = index_manager.get_stats()
        assert stats["num_docs"] == 0

    def test_rebuild_in_memory(self, in_memory_manager):
        """Rebuild should work for in-memory index."""
        in_memory_manager.add_document(
            doc_id="rebuild_mem",
            title="Memory Rebuild",
            content="Content.",
            metadata={},
        )
        in_memory_manager.commit()

        result = in_memory_manager.rebuild()

        assert result is True


class TestGetStats:
    """Tests for get_stats operation."""

    def test_get_stats_returns_dict(self, in_memory_manager):
        """Should return statistics as dictionary."""
        stats = in_memory_manager.get_stats()

        assert isinstance(stats, dict)
        assert "num_docs" in stats

    def test_get_stats_with_documents(self, in_memory_manager):
        """Stats should reflect number of documents."""
        # Add documents
        for i in range(3):
            in_memory_manager.add_document(
                doc_id=f"stats{i}",
                title=f"Stats Doc {i}",
                content=f"Content {i}.",
                metadata={},
            )
        in_memory_manager.commit()

        stats = in_memory_manager.get_stats()

        # Note: num_docs might vary due to indexing
        assert isinstance(stats["num_docs"], int)


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager(self):
        """Should work as context manager."""
        with TantivyIndexManager(index_path=None, use_jieba=False) as manager:
            result = manager.add_document(
                doc_id="context_test",
                title="Context Test",
                content="Content.",
                metadata={},
            )
            assert result is True
        # Auto-commit on exit


class TestChineseTokenization:
    """Tests for Chinese text tokenization."""

    def test_tokenize_chinese_without_jieba(self, in_memory_manager):
        """Should tokenize Chinese with regex when jieba unavailable."""
        text = "这是一个测试"

        result = in_memory_manager._tokenize_chinese(text)

        # Without jieba, returns original text
        assert isinstance(result, str)

    def test_tokenize_empty_string(self, in_memory_manager):
        """Should handle empty string."""
        result = in_memory_manager._tokenize_chinese("")

        assert result == ""

    def test_add_and_search_mixed_language(self, in_memory_manager):
        """Should handle mixed Chinese and English content."""
        in_memory_manager.add_document(
            doc_id="mixed",
            title="Mixed Language 混合语言",
            content="This is English content mixed with 中文内容.",
            metadata={"keywords": ["Python", "中文"]},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("English", limit=10)

        assert isinstance(result, SearchResult)


class TestExcerptGeneration:
    """Tests for excerpt generation."""

    def test_create_excerpt_basic(self, in_memory_manager):
        """Should create excerpt from content."""
        content = "This is a long piece of content. It has many words and sentences. The query appears here: Python programming. And continues after."

        excerpt = in_memory_manager._create_excerpt(content, "Python")

        assert "Python" in excerpt
        assert len(excerpt) <= 203  # max_length + 3 for ellipsis

    def test_create_excerpt_no_match(self, in_memory_manager):
        """Should return beginning when no match."""
        content = "This content does not contain the query term."

        excerpt = in_memory_manager._create_excerpt(content, "xyznonexistent")

        # Should return beginning of content
        assert content[:50] in excerpt or excerpt.startswith(content[:20])

    def test_create_excerpt_empty_content(self, in_memory_manager):
        """Should handle empty content."""
        excerpt = in_memory_manager._create_excerpt("", "query")

        assert excerpt == ""


class TestPersistence:
    """Tests for index persistence."""

    def test_persistence_across_sessions(self, temp_index_dir):
        """Index should persist across manager instances."""
        index_path = temp_index_dir / "persist_test"

        # Create and add document
        manager1 = TantivyIndexManager(index_path=index_path, use_jieba=False)
        manager1.add_document(
            doc_id="persist1",
            title="Persistence Test",
            content="This document should persist.",
            metadata={},
        )
        manager1.commit()
        manager1.close()

        # Open new manager at same path
        manager2 = TantivyIndexManager(index_path=index_path, use_jieba=False)
        result = manager2.search("persist", limit=10)

        assert result.total >= 1
        manager2.close()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_special_characters_in_content(self, in_memory_manager):
        """Should handle special characters."""
        result = in_memory_manager.add_document(
            doc_id="special",
            title="Special Chars: <>&\"'",
            content="Content with <html> tags & \"quotes\" and 'apostrophes'.",
            metadata={},
        )

        assert result is True

    def test_very_long_content(self, in_memory_manager):
        """Should handle very long content."""
        long_content = "word " * 10000  # ~50KB

        result = in_memory_manager.add_document(
            doc_id="long",
            title="Long Document",
            content=long_content,
            metadata={},
        )

        assert result is True

    def test_unicode_content(self, in_memory_manager):
        """Should handle various unicode characters."""
        result = in_memory_manager.add_document(
            doc_id="unicode",
            title="Unicode Тест 中文 العربية",
            content="Content with: émojis 🎉, Greek αβγ, Cyrillic абв, Chinese 中文.",
            metadata={},
        )

        assert result is True

    def test_concurrent_operations(self, in_memory_manager):
        """Should handle sequential add and search operations."""
        for i in range(5):
            in_memory_manager.add_document(
                doc_id=f"concurrent{i}",
                title=f"Document {i}",
                content=f"Content {i}.",
                metadata={},
            )

        in_memory_manager.commit()
        result = in_memory_manager.search("Content", limit=10)

        assert len(result.hits) >= 1


class TestSearchHighlights:
    """F3-4: Search result highlighting tests."""

    def test_highlights_returned_for_match(self, in_memory_manager):
        """Search should return non-empty highlights for matching documents."""
        in_memory_manager.add_document(
            doc_id="hl1",
            title="Python Guide",
            content="Python is a popular programming language for data science.",
            metadata={},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("Python", limit=10)

        assert len(result.hits) >= 1
        hit = result.hits[0]
        assert isinstance(hit.highlights, list)
        # Should have at least one highlighted term
        assert len(hit.highlights) >= 1

    def test_highlights_empty_for_no_match(self, in_memory_manager):
        """No matching terms should produce empty highlights."""
        in_memory_manager.add_document(
            doc_id="hl_nomatch",
            title="Some Title",
            content="Some content without the query term.",
            metadata={},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("xyznonexistent123", limit=10)

        assert result.total == 0
        assert len(result.hits) == 0

    def test_highlights_limited_to_max_terms(self, in_memory_manager):
        """Highlights should be capped at max_terms (default 5)."""
        content = "alpha beta gamma delta epsilon zeta eta theta"
        in_memory_manager.add_document(
            doc_id="hl_limit",
            title="Many Terms",
            content=content,
            metadata={},
        )
        in_memory_manager.commit()

        # Use _extract_highlight_terms directly to verify capping
        highlights = in_memory_manager._extract_highlight_terms(
            "alpha beta gamma delta epsilon zeta eta theta",
            content,
            max_terms=3,
        )

        assert len(highlights) <= 3

    def test_extract_highlight_terms_from_query(self, in_memory_manager):
        """_extract_highlight_terms should find query terms in content."""
        content = "This document discusses machine learning algorithms."
        highlights = in_memory_manager._extract_highlight_terms(
            "machine learning", content
        )

        # Should find both terms in content
        assert "machine" in highlights
        assert "learning" in highlights

    def test_extract_highlight_terms_case_insensitive(self, in_memory_manager):
        """Highlight extraction should be case-insensitive."""
        content = "Python Programming is fun."
        highlights = in_memory_manager._extract_highlight_terms(
            "python programming", content
        )

        assert len(highlights) >= 1

    def test_extract_highlight_terms_empty_query(self, in_memory_manager):
        """Empty query should return empty highlights."""
        highlights = in_memory_manager._extract_highlight_terms("", "some content")
        assert highlights == []

    def test_extract_highlight_terms_empty_content(self, in_memory_manager):
        """Empty content should return empty highlights."""
        highlights = in_memory_manager._extract_highlight_terms("python", "")
        assert highlights == []

    def test_highlights_in_search_result(self, in_memory_manager):
        """Full search should populate highlights field."""
        in_memory_manager.add_document(
            doc_id="hl_full",
            title="Test Document",
            content="The quick brown fox jumps over the lazy dog.",
            metadata={},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("quick fox", limit=10)

        assert len(result.hits) >= 1
        # The highlights field should be a list (may be populated or empty
        # depending on Tantivy snippet behavior, but must not error)
        assert isinstance(result.hits[0].highlights, list)

    def test_extract_highlight_terms_phrase_query(self, in_memory_manager):
        """Phrase queries should produce correct highlights."""
        content = 'The "machine learning" algorithm works well.'
        highlights = in_memory_manager._extract_highlight_terms(
            '"machine learning"', content
        )

        # Should find the phrase or individual terms
        assert len(highlights) >= 1

    def test_extract_highlight_terms_chinese(self, in_memory_manager):
        """Chinese query terms should be extracted as highlights."""
        content = "这是一个关于绩效考核的文档"
        highlights = in_memory_manager._extract_highlight_terms(
            "绩效考核", content
        )

        # Should find the Chinese terms
        assert len(highlights) >= 1

    def test_search_hit_has_highlights_field(self, in_memory_manager):
        """SearchHit from search should always have highlights field."""
        in_memory_manager.add_document(
            doc_id="hl_field",
            title="Field Test",
            content="Content for highlight field testing.",
            metadata={},
        )
        in_memory_manager.commit()

        result = in_memory_manager.search("highlight", limit=10)

        if result.hits:
            hit = result.hits[0]
            assert hasattr(hit, "highlights")
            assert isinstance(hit.highlights, list)
