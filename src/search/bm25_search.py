"""
BM25 Search Executor for document search.

This module provides the BM25Searcher class for executing BM25-based searches
with support for:
- Natural language query parsing
- Pagination (limit/offset)
- Two-phase search (preview + full content)
- Field-specific searches
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.search.query_parser import QueryParser
from src.search.result_formatter import SearchResult
from src.storage.index import TantivyIndexManager


@dataclass
class SearchPreview:
    """
    Preview result for two-phase search (Phase 1).

    Contains lightweight information for displaying search results
    without loading full document content.
    """

    doc_id: str
    title: str
    score: float
    snippet: str
    source_path: Path | None = None
    highlights: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Normalize source_path to Path object."""
        if self.source_path is not None and isinstance(self.source_path, str):
            self.source_path = Path(self.source_path)


@dataclass
class FullSearchResult(SearchResult):
    """
    Full search result with complete content (Phase 2).

    Extends SearchResult with full document content for when
    user selects a specific result.
    """

    doc_id: str = ""
    full_content: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class PaginatedResults:
    """
    Container for paginated search results.

    Includes pagination metadata for UI navigation.
    """

    results: list[SearchPreview]
    total: int
    offset: int
    limit: int
    has_more: bool
    query: str
    execution_time: float = 0.0


class BM25Searcher:
    """
    BM25-based search executor with advanced features.

    This class provides a high-level search interface that:
    - Parses natural language queries using QueryParser
    - Executes searches using TantivyIndexManager
    - Supports two-phase search (preview + full content)
    - Handles pagination efficiently
    - Provides field-specific search capabilities

    Example:
        >>> index_manager = TantivyIndexManager(index_path=Path("./index"))
        >>> searcher = BM25Searcher(index_manager)
        >>>
        >>> # Phase 1: Preview search
        >>> previews = searcher.search("绩效管理", limit=10)
        >>>
        >>> # Phase 2: Get full content for selected result
        >>> full_result = searcher.get_full_content(previews.results[0].doc_id)
    """

    # Valid field names for field-specific search
    VALID_FIELDS = frozenset(
        [
            "title",
            "content",
            "keywords",
            "filename",
            "source_path",
        ]
    )

    def __init__(
        self,
        index_manager: TantivyIndexManager,
        snippet_length: int = 200,
        min_score: float = 0.0,
        title_boost: float = 1.0,
    ):
        """
        Initialize the BM25Searcher.

        Args:
            index_manager: TantivyIndexManager instance for executing searches
            snippet_length: Maximum length of content snippets in preview
            min_score: Minimum score threshold for results
            title_boost: Boost factor for title field matches (1.0 = no boost).
                Values > 1.0 give extra weight to documents whose title
                matches the query terms.
        """
        self._index = index_manager
        self._parser = QueryParser()
        self._snippet_length = snippet_length
        self._min_score = min_score
        self._title_boost = title_boost

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> PaginatedResults:
        """
        Execute a BM25 search and return paginated preview results.

        This is Phase 1 of the two-phase search, returning lightweight
        preview information without loading full document content.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)

        Returns:
            PaginatedResults containing preview results and pagination metadata

        Example:
            >>> results = searcher.search("项目进度报告", limit=10, offset=0)
            >>> for preview in results.results:
            ...     print(f"{preview.title} (score: {preview.score})")
        """
        import time

        start_time = time.time()

        # Handle empty query gracefully
        if not query or not query.strip():
            return PaginatedResults(
                results=[],
                total=0,
                offset=offset,
                limit=limit,
                has_more=False,
                query=query or "",
                execution_time=time.time() - start_time,
            )

        # Parse the query
        parsed_query = self._parser.parse(query)

        # Convert to Tantivy query string
        tantivy_query = self._parser.to_tantivy_query(parsed_query)

        # If parser returned empty query, use original query
        search_query = tantivy_query if tantivy_query else query

        # Execute search using index manager
        search_result = self._index.search(
            query=search_query,
            limit=limit,
            offset=offset,
            title_boost=self._title_boost,
        )

        # Convert SearchHit to SearchPreview
        previews = []
        for hit in search_result.hits:
            # Skip results below minimum score
            if hit.score < self._min_score:
                continue

            preview = SearchPreview(
                doc_id=hit.doc_id,
                title=hit.title,
                score=hit.score,
                snippet=self._truncate_snippet(hit.excerpt),
                source_path=hit.source_path,
                highlights=hit.highlights,
            )
            previews.append(preview)

        execution_time = time.time() - start_time

        return PaginatedResults(
            results=previews,
            total=search_result.total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < search_result.total,
            query=query,
            execution_time=execution_time,
        )

    def get_full_content(
        self,
        doc_id: str,
        include_metadata: bool = True,
    ) -> FullSearchResult | None:
        """
        Get full content for a specific document (Phase 2).

        This method loads the complete document content when a user
        selects a specific result from the preview.

        Args:
            doc_id: Document ID to retrieve
            include_metadata: Whether to include document metadata

        Returns:
            FullSearchResult with complete content, or None if not found

        Example:
            >>> previews = searcher.search("报告", limit=5)
            >>> if previews.results:
            ...     full = searcher.get_full_content(previews.results[0].doc_id)
            ...     if full:
            ...         print(full.full_content)
        """
        # Retrieve full document directly (no excerpt truncation)
        doc = self._index.get_document_by_id(doc_id)

        if doc is None:
            return None

        # Build metadata
        metadata: dict[str, Any] = {}
        if include_metadata:
            metadata["doc_id"] = doc["doc_id"]
            metadata["filename"] = doc["filename"]
            metadata["source_path"] = doc["source_path"]

        # Parse keywords
        keywords_str = doc.get("keywords", "")
        keywords = [k.strip() for k in keywords_str.split() if k.strip()] if keywords_str else []

        return FullSearchResult(
            doc_id=doc["doc_id"],
            title=doc["title"],
            score=1.0,
            snippet="",
            source=Path(doc["source_path"]) if doc["source_path"] else Path(""),
            timestamp=datetime.now(),
            metadata=metadata,
            full_content=doc["content"],
            keywords=keywords,
        )

    def search_by_field(
        self,
        field: str,
        value: str,
        limit: int = 10,
        offset: int = 0,
    ) -> PaginatedResults:
        """
        Execute a field-specific search.

        Searches within a specific field (title, content, keywords, etc.)
        rather than across all fields.

        Args:
            field: Field name to search in (title, content, keywords, filename)
            value: Value to search for
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            PaginatedResults with matching documents

        Raises:
            ValueError: If field name is not valid

        Example:
            >>> results = searcher.search_by_field("title", "季度报告")
            >>> results = searcher.search_by_field("keywords", "财务")
        """
        import time

        start_time = time.time()

        # Validate field name
        if field not in self.VALID_FIELDS:
            raise ValueError(
                f"Invalid field '{field}'. Valid fields: {', '.join(sorted(self.VALID_FIELDS))}"
            )

        # Handle empty value
        if not value or not value.strip():
            return PaginatedResults(
                results=[],
                total=0,
                offset=offset,
                limit=limit,
                has_more=False,
                query=f"{field}:{value}",
                execution_time=time.time() - start_time,
            )

        # Build field-specific query
        # Escape quotes in value for safety
        escaped_value = value.replace('"', '\\"')
        field_query = f'{field}:"{escaped_value}"'

        # Execute search
        search_result = self._index.search(
            query=field_query,
            limit=limit,
            offset=offset,
        )

        # Convert to previews
        previews = []
        for hit in search_result.hits:
            if hit.score < self._min_score:
                continue

            preview = SearchPreview(
                doc_id=hit.doc_id,
                title=hit.title,
                score=hit.score,
                snippet=self._truncate_snippet(hit.excerpt),
                source_path=hit.source_path,
                highlights=hit.highlights,
            )
            previews.append(preview)

        execution_time = time.time() - start_time

        return PaginatedResults(
            results=previews,
            total=search_result.total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < search_result.total,
            query=f"{field}:{value}",
            execution_time=execution_time,
        )

    def search_multi_field(
        self,
        field_queries: dict[str, str],
        limit: int = 10,
        offset: int = 0,
        operator: str = "AND",
    ) -> PaginatedResults:
        """
        Execute a multi-field search with combined criteria.

        Args:
            field_queries: Dictionary mapping field names to search values
            limit: Maximum number of results
            offset: Number of results to skip
            operator: Combination operator ("AND" or "OR")

        Returns:
            PaginatedResults with matching documents

        Raises:
            ValueError: If operator is not "AND" or "OR"

        Example:
            >>> results = searcher.search_multi_field({
            ...     "title": "报告",
            ...     "keywords": "财务",
            ... }, operator="AND")
        """
        import time

        start_time = time.time()

        if operator not in ("AND", "OR"):
            raise ValueError(f"Operator must be 'AND' or 'OR', got '{operator}'")

        # Filter out invalid fields
        valid_queries = {
            field: value
            for field, value in field_queries.items()
            if field in self.VALID_FIELDS and value and value.strip()
        }

        if not valid_queries:
            return PaginatedResults(
                results=[],
                total=0,
                offset=offset,
                limit=limit,
                has_more=False,
                query="",
                execution_time=time.time() - start_time,
            )

        # Build combined query
        query_parts = []
        for field, value in valid_queries.items():
            escaped_value = value.replace('"', '\\"')
            query_parts.append(f'{field}:"{escaped_value}"')

        combined_query = f" {operator} ".join(query_parts)

        # Execute search
        search_result = self._index.search(
            query=combined_query,
            limit=limit,
            offset=offset,
        )

        # Convert to previews
        previews = []
        for hit in search_result.hits:
            if hit.score < self._min_score:
                continue

            preview = SearchPreview(
                doc_id=hit.doc_id,
                title=hit.title,
                score=hit.score,
                snippet=self._truncate_snippet(hit.excerpt),
                source_path=hit.source_path,
                highlights=hit.highlights,
            )
            previews.append(preview)

        execution_time = time.time() - start_time

        return PaginatedResults(
            results=previews,
            total=search_result.total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < search_result.total,
            query=combined_query,
            execution_time=execution_time,
        )

    def count(self, query: str) -> int:
        """
        Count total matching documents for a query.

        This is useful for displaying total result counts without
        loading all results.

        Args:
            query: Search query

        Returns:
            Total number of matching documents

        Example:
            >>> total = searcher.count("项目管理")
            >>> print(f"Found {total} documents")
        """
        if not query or not query.strip():
            return 0

        # Parse and convert query
        parsed_query = self._parser.parse(query)
        tantivy_query = self._parser.to_tantivy_query(parsed_query)
        search_query = tantivy_query if tantivy_query else query

        # Search with limit=0 to just get count
        result = self._index.search(query=search_query, limit=1)

        return result.total

    def suggest(
        self,
        partial_query: str,
        limit: int = 5,
    ) -> list[str]:
        """
        Suggest query completions based on partial input.

        This is useful for implementing search-as-you-type functionality.

        Args:
            partial_query: Partial query string
            limit: Maximum number of suggestions

        Returns:
            List of suggested query completions

        Note:
            Current implementation returns terms extracted from matching documents.
            Future versions may integrate with a dedicated suggestion index.
        """
        if not partial_query or len(partial_query.strip()) < 2:
            return []

        # Search with partial query
        results = self.search(partial_query, limit=limit)

        # Extract unique terms from titles
        suggestions = set()
        for preview in results.results:
            # Add title as suggestion
            if preview.title:
                suggestions.add(preview.title)

            if len(suggestions) >= limit:
                break

        return list(suggestions)[:limit]

    def _truncate_snippet(self, text: str) -> str:
        """
        Truncate snippet to configured length.

        Args:
            text: Text to truncate

        Returns:
            Truncated text with ellipsis if needed
        """
        if not text:
            return ""

        if len(text) <= self._snippet_length:
            return text

        return text[: self._snippet_length - 3] + "..."

    def get_index_stats(self) -> dict[str, Any]:
        """
        Get statistics about the search index.

        Returns:
            Dictionary with index statistics
        """
        return self._index.get_stats()


def create_searcher(
    index_path: Path | None = None,
    use_jieba: bool = True,
    snippet_length: int = 200,
    readonly: bool = False,
) -> BM25Searcher:
    """
    Factory function to create a BM25Searcher instance.

    Args:
        index_path: Path to the index directory (None for in-memory)
        use_jieba: Whether to use jieba for Chinese tokenization
        snippet_length: Maximum length of snippets in preview
        readonly: If True, skip IndexWriter creation (search-only mode)

    Returns:
        Configured BM25Searcher instance

    Example:
        >>> searcher = create_searcher(Path("./index"))
        >>> results = searcher.search("测试查询")
    """
    index_manager = TantivyIndexManager(
        index_path=index_path,
        use_jieba=use_jieba,
        readonly=readonly,
    )

    return BM25Searcher(
        index_manager=index_manager,
        snippet_length=snippet_length,
    )
