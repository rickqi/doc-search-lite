"""RerankTool - Agent tool for reranking search results using ZhipuAI Rerank API.

This module provides the RerankTool class that wraps ZhipuAIReranker
to enable agent-based reranking functionality for improving search result relevance.
"""

import json
import logging
import time
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

from src.agent.base import Tool, ToolResult
from src.search.reranker import RerankResult


class RerankTool(Tool):
    """Tool for reranking search results using ZhipuAI Rerank API.

    This tool wraps ZhipuAIReranker to provide agent-compatible reranking
    functionality with formatted output showing relevance scores and document excerpts.

    Attributes:
        _reranker: ZhipuAIReranker instance for executing reranking

    Example:
        >>> from src.search.reranker import ZhipuAIReranker
        >>>
        >>> reranker = ZhipuAIReranker()
        >>> tool = RerankTool(reranker)
        >>>
        >>> result = tool.execute(
        ...     query="年假如何申请",
        ...     documents=["doc1 text...", "doc2 text..."],
        ...     top_n=5
        ... )
        >>> if result.success:
        ...     print(result.data)
    """

    def __init__(self, reranker: Any) -> None:
        """Initialize the RerankTool.

        Args:
            reranker: Reranker instance (ZhipuAIReranker or LocalReranker).
                Must implement .rerank(query, documents, top_n) -> List[RerankResult]
        """
        self._reranker = reranker

    @property
    def name(self) -> str:
        """Unique identifier for the tool.

        Returns:
            str: The tool name 'rerank'
        """
        return "rerank"

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does.

        Returns:
            str: Description of the rerank tool functionality
        """
        return "对搜索结果进行重排序，提高相关性精度。输入查询和文档文本列表，返回按相关性排序的结果。"

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the rerank tool with given parameters.

        Args:
            **kwargs: Tool arguments including:
                - query (str): Search query string (required)
                - documents (Union[str, List[str]]): JSON string or list of document texts (required)
                - top_n (int): Number of top results to return (default: 10)

        Returns:
            ToolResult: Result containing:
                - success: Whether reranking was successful
                - data: Formatted string with reranked results showing scores and excerpts
                - error: Error message if failed
                - metadata: Tokens used, document count, top_n, execution time

        Example:
            >>> result = tool.execute(
            ...     query="年假如何申请",
            ...     documents=["doc1 text...", "doc2 text..."],
            ...     top_n=5
            ... )
            >>> if result.success:
            ...     print(result.data)
        """
        start_time = time.time()

        # Extract parameters
        query = kwargs.get("query", "")
        documents_input = kwargs.get("documents", [])
        top_n = kwargs.get("top_n", 10)

        # Validate required parameters
        if not query or not isinstance(query, str) or not query.strip():
            return ToolResult.fail(
                error="Parameter 'query' is required and must be a non-empty string",
                metadata={"execution_time": time.time() - start_time},
            )

        # Parse documents parameter (can be JSON string or list)
        if isinstance(documents_input, str):
            try:
                documents = json.loads(documents_input)
                if not isinstance(documents, list):
                    return ToolResult.fail(
                        error="Parameter 'documents' must be a JSON array",
                        metadata={"execution_time": time.time() - start_time},
                    )
            except json.JSONDecodeError as exc:
                return ToolResult.fail(
                    error=f"Failed to parse 'documents' as JSON: {exc}",
                    metadata={"execution_time": time.time() - start_time},
                )
        elif isinstance(documents_input, list):
            documents = documents_input
        else:
            return ToolResult.fail(
                error="Parameter 'documents' must be a JSON string or list",
                metadata={"execution_time": time.time() - start_time},
            )

        # Validate documents is non-empty
        if not documents:
            return ToolResult.fail(
                error="Parameter 'documents' must be a non-empty array",
                metadata={"execution_time": time.time() - start_time},
            )

        # Validate all documents are strings
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                return ToolResult.fail(
                    error=f"Document at index {i} must be a string, got {type(doc).__name__}",
                    metadata={"execution_time": time.time() - start_time},
                )

        # Validate top_n
        if not isinstance(top_n, int) or top_n < 1:
            return ToolResult.fail(
                error="Parameter 'top_n' must be a positive integer",
                metadata={"execution_time": time.time() - start_time},
            )

        total_documents = len(documents)

        # Cap candidates to reduce API cost (RAGFlow pattern: min(64, top))
        RERANK_MAX_CANDIDATES = 64
        if len(documents) > RERANK_MAX_CANDIDATES:
            documents = documents[:RERANK_MAX_CANDIDATES]
            logger.debug("Rerank capped %d → %d candidates", total_documents, RERANK_MAX_CANDIDATES)

        try:
            # Execute reranking
            reranked_results = self._reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_n,
            )

            # Format results
            formatted_output = self._format_results(
                reranked_results, documents, top_n
            )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Build metadata
            metadata = {
                "query": query,
                "total_documents": total_documents,
                "top_n": top_n,
                "returned_count": len(reranked_results),
                "tokens_used": self._reranker.tokens_used,
                "execution_time": execution_time,
            }

            return ToolResult.ok(
                data=formatted_output,
                metadata=metadata,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult.fail(
                error=f"Rerank failed: {str(e)}",
                metadata={
                    "query": query,
                    "total_documents": total_documents,
                    "top_n": top_n,
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                },
            )

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format.

        Returns:
            Dict[str, Any]: Tool definition in OpenAI format with JSON schema
            for parameters including query, documents, and top_n.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "对搜索结果进行重排序，提高相关性精度。当BM25搜索结果不够精确时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询文本",
                        },
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": "待排序的文档文本列表",
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "返回前N个最相关结果（默认10）",
                            "default": 10,
                        },
                    },
                    "required": ["query", "documents"],
                },
            },
        }

    def _format_results(
        self,
        reranked_results: List[RerankResult],
        documents: List[str],
        top_n: int,
    ) -> str:
        """Format reranked results into a structured string.

        Args:
            reranked_results: List of RerankResult from ZhipuAIReranker
            documents: Original list of document texts
            top_n: Number of top results requested

        Returns:
            str: Formatted string with reranked results showing scores and excerpts
        """
        if not reranked_results:
            return "No reranked results returned."

        lines = [f"Reranked results (top {len(reranked_results)}):"]

        for rank, result in enumerate(reranked_results, start=1):
            # Get document excerpt (first 100 chars)
            doc_index = result.index
            doc_excerpt = documents[doc_index][:100] if doc_index < len(documents) else ""

            lines.append(
                f"{rank}. [score={result.relevance_score:.4f}] Document index {result.index}: \"{doc_excerpt}\""
            )

        return "\n".join(lines)