"""Search tool for agent-based document search.

This module provides the SearchTool class that wraps BM25Searcher
to enable agent-based document search functionality.

The tool implements the Tool protocol and returns formatted JSON results
suitable for agent consumption.
"""

import json
import time
from pathlib import Path
from typing import Any

from src.agent.base import Tool
from src.agent.tool_types import ToolCache, ToolResult
from src.search.bm25_search import BM25Searcher, PaginatedResults


class SearchTool(Tool):
    """Tool for searching documents using BM25 algorithm.

    This tool wraps BM25Searcher to provide agent-compatible search
    functionality with formatted JSON output.

    Attributes:
        _searcher: BM25Searcher instance for executing searches
        _default_limit: Default number of results to return
        _snippet_length: Default snippet length for previews

    Example:
        >>> from src.search.bm25_search import BM25Searcher
        >>> from src.storage.index import TantivyIndexManager
        >>>
        >>> index_manager = TantivyIndexManager(index_path=Path("./index"))
        >>> searcher = BM25Searcher(index_manager)
        >>> tool = SearchTool(searcher)
        >>>
        >>> result = tool.execute(query="绩效管理", limit=5)
        >>> if result.success:
        ...     print(result.data)
    """

    def __init__(
        self,
        searcher: BM25Searcher,
        default_limit: int = 10,
        snippet_length: int = 200,
    ) -> None:
        """Initialize the SearchTool.

        Args:
            searcher: BM25Searcher instance for executing searches
            default_limit: Default number of results to return (default: 10)
            snippet_length: Maximum length of content snippets (default: 200)
        """
        self._searcher = searcher
        self._default_limit = default_limit
        self._snippet_length = snippet_length
        self._cache: ToolCache | None = None

    def set_cache(self, cache: ToolCache) -> None:
        """Attach an optional ToolCache for result caching (opt-in)."""
        self._cache = cache

    @property
    def name(self) -> str:
        """Unique identifier for the tool.

        Returns:
            str: The tool name 'search'
        """
        return "search"

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does.

        Returns:
            str: Description of the search tool functionality
        """
        return "搜索文档库中的相关内容。使用BM25算法进行全文检索，支持中文和混合语言查询。返回匹配文档的标题、片段和相关性得分。"

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the search tool with given parameters.

        Args:
            **kwargs: Tool arguments including:
                - query (str): Search query string (required)
                - limit (int): Maximum results to return (default: 10)
                - offset (int): Number of results to skip (default: 0)

        Returns:
            ToolResult: Result containing:
                - success: Whether search was successful
                - data: JSON string with formatted search results
                - error: Error message if failed
                - metadata: Query time, result count, pagination info

        Example:
            >>> result = tool.execute(query="绩效管理", limit=5)
            >>> if result.success:
            ...     data = json.loads(result.data)
            ...     print(f"Found {data['total']} results")
        """
        # Opt-in cache: check before doing any work
        if self._cache is not None:
            cache_key = ToolCache.make_key(self.name, kwargs)
            cached = self._cache.get(cache_key)
            if cached is not None:
                hit_meta = {**cached.metadata, "cache_hit": True}
                return ToolResult(
                    success=cached.success,
                    data=cached.data,
                    error=cached.error,
                    metadata=hit_meta,
                )

        start_time = time.time()

        # Extract parameters
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", self._default_limit)
        offset = kwargs.get("offset", 0)

        # Validate required parameters
        if not query or not isinstance(query, str) or not query.strip():
            return ToolResult.fail(
                error="Parameter 'query' is required and must be a non-empty string",
                metadata={"execution_time": time.time() - start_time},
            )

        # Validate numeric parameters
        if not isinstance(limit, int) or limit < 0:
            return ToolResult.fail(
                error="Parameter 'limit' must be a non-negative integer",
                metadata={"execution_time": time.time() - start_time},
            )

        if not isinstance(offset, int) or offset < 0:
            return ToolResult.fail(
                error="Parameter 'offset' must be a non-negative integer",
                metadata={"execution_time": time.time() - start_time},
            )

        try:
            # Execute search
            paginated_results = self._searcher.search(
                query=query,
                limit=limit,
                offset=offset,
            )

            # Format results
            formatted_data = self._format_results(paginated_results)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Build metadata
            metadata = {
                "query": query,
                "total_results": paginated_results.total,
                "returned_count": len(paginated_results.results),
                "offset": offset,
                "limit": limit,
                "has_more": paginated_results.has_more,
                "execution_time": execution_time,
                "search_time": paginated_results.execution_time,
            }

            result = ToolResult.ok(
                data=json.dumps(formatted_data, ensure_ascii=False, indent=2),
                metadata=metadata,
            )

            # P2: Structured feedback signals for LLM strategy adaptation
            total = paginated_results.total
            if total == 0:
                result = ToolResult.ok(
                    data=json.dumps(
                        {**formatted_data, "hint": (
                            "零命中建议：1) 简化关键词 2) 使用同义词 3) 尝试 grep 工具"
                        )},
                        ensure_ascii=False, indent=2,
                    ),
                    metadata=metadata,
                )
            elif total <= 2:
                result = ToolResult.ok(
                    data=json.dumps(
                        {**formatted_data, "hint": (
                            "命中较少。建议读取后判断是否需要补充搜索。"
                        )},
                        ensure_ascii=False, indent=2,
                    ),
                    metadata=metadata,
                )

            if self._cache is not None:
                self._cache.put(cache_key, result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = ToolResult.fail(
                error=f"Search failed: {str(e)}",
                metadata={
                    "query": query,
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                },
            )
            if self._cache is not None:
                self._cache.put(cache_key, result)
            return result

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert tool to OpenAI function calling format.

        Returns:
            Dict[str, Any]: Tool definition in OpenAI format with JSON schema
            for parameters including query, limit, and offset.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "BM25关键词搜索查询。⚠️ 用精炼关键词而不要用自然语言问句（'法释2018年2号 夫妻债务'优于'请搜索关于夫妻债务的司法解释'）。多个关键词用空格分隔。中文和英文均支持。",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回结果的最大数量，默认为10",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "跳过的结果数量，用于分页，默认为0",
                            "default": 0,
                            "minimum": 0,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def _format_results(self, paginated_results: PaginatedResults) -> dict[str, Any]:
        """Format paginated results into a structured dictionary.

        Args:
            paginated_results: PaginatedResults from BM25Searcher

        Returns:
            Dict[str, Any]: Formatted results dictionary with:
                - query: Original query string
                - total: Total matching documents
                - offset: Current offset
                - limit: Results limit
                - has_more: Whether more results exist
                - results: List of formatted search results
        """
        formatted_results = []

        for preview in paginated_results.results:
            result_item = {
                "doc_id": preview.doc_id,
                "title": preview.title,
                "score": round(preview.score, 4),
                "snippet": preview.snippet,
                "source_path": str(preview.source_path)
                if preview.source_path
                else None,
                "highlights": preview.highlights,
            }
            formatted_results.append(result_item)

        return {
            "query": paginated_results.query,
            "total": paginated_results.total,
            "offset": paginated_results.offset,
            "limit": paginated_results.limit,
            "has_more": paginated_results.has_more,
            "results": formatted_results,
        }


def create_search_tool(
    index_path: Path | None = None,
    use_jieba: bool = True,
    default_limit: int = 10,
    snippet_length: int = 200,
) -> SearchTool:
    """Factory function to create a SearchTool instance.

    This function creates a complete search tool with its own BM25Searcher
    and TantivyIndexManager.

    Args:
        index_path: Path to the index directory (None for in-memory)
        use_jieba: Whether to use jieba for Chinese tokenization
        default_limit: Default number of results to return
        snippet_length: Maximum length of snippets in preview

    Returns:
        SearchTool: Configured search tool instance

    Example:
        >>> tool = create_search_tool(Path("./index"))
        >>> result = tool.execute(query="测试查询")
    """
    from src.search.bm25_search import create_searcher

    searcher = create_searcher(
        index_path=index_path,
        use_jieba=use_jieba,
        snippet_length=snippet_length,
    )

    return SearchTool(
        searcher=searcher,
        default_limit=default_limit,
        snippet_length=snippet_length,
    )
