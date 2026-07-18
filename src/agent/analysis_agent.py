"""Analysis Agent for document comparison and structured extraction.

This module provides the AnalysisAgent that supports:
- Compare: Compare multiple documents and find differences/similarities
- Extract: Extract structured data (tables, lists, key-value pairs)

All operations return results with source references.
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.agent.base import Agent, AgentResponse
from src.agent.llm_client import ChatMessage, LLMClient
from src.agent.tools.analyze import AnalyzeTool, LLMClientProtocol
from src.storage.markdown_store import MarkdownStore

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class LLMClientAdapter:
    """Adapter to make LLMClient compatible with LLMClientProtocol.

    The AnalyzeTool expects a client with generate() and count_tokens() methods,
    but LLMClient has chat() method. This adapter bridges the gap.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize adapter with LLMClient instance.

        Args:
            llm_client: The LLMClient instance to adapt
        """
        self._client = llm_client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        messages: List[ChatMessage] = []

        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

        messages.append(ChatMessage(role="user", content=prompt))

        response = self._client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.content

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple approximation of 4 characters per token.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated number of tokens
        """
        # Simple approximation: ~4 characters per token for Chinese/English
        return max(1, len(text) // 4)


class AnalysisAgent(Agent):
    """Agent for deep document analysis.

    Supports two analysis modes:
    1. compare: Compare multiple documents to find similarities and differences
    2. extract: Extract structured information from documents

    All results include source document references for traceability.

    Example:
        >>> agent = AnalysisAgent(llm_client, markdown_store)
        >>> response = agent.compare(["doc1", "doc2"], aspect="pricing")
        >>> print(response.answer)  # Comparison result
        >>> print(response.sources)  # ["doc1", "doc2"]
    """

    def __init__(
        self,
        llm_client: LLMClient,
        markdown_store: MarkdownStore,
    ):
        """Initialize the AnalysisAgent.

        Args:
            llm_client: LLM client for analysis operations
            markdown_store: Markdown store for accessing documents
        """
        super().__init__()
        self._llm_client = llm_client
        self._store = markdown_store

        # Create adapter for AnalyzeTool compatibility
        self._llm_adapter = LLMClientAdapter(llm_client)

        # Create and register the analyze tool
        self._analyze_tool = AnalyzeTool(
            markdown_store=markdown_store,
            llm_client=self._llm_adapter,
        )
        self.register_tool(self._analyze_tool)

    @property
    def name(self) -> str:
        """Unique identifier for the agent."""
        return "analysis_agent"

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Execute analysis based on query and context.

        The agent automatically detects the analysis mode from context:
        - If context contains 'doc_ids' with 2+ items: compare mode
        - If context contains 'doc_id' or 'doc_ids' with 1 item: extract mode
        - If context contains 'mode': use specified mode

        Args:
            query: The analysis query/instruction
            context: Optional context containing:
                - mode: 'compare' or 'extract' (auto-detected if not specified)
                - doc_ids: List of document IDs for compare mode
                - doc_id: Single document ID for extract mode
                - schema: JSON schema for structured extraction
                - aspect: Specific aspect to focus on (for compare)

        Returns:
            AgentResponse with analysis results
        """
        start_time = time.time()
        context = context or {}

        try:
            # Determine mode from context
            mode = context.get("mode")

            if mode is None:
                # Auto-detect mode
                doc_ids = context.get("doc_ids", [])
                if len(doc_ids) >= 2:
                    mode = "compare"
                else:
                    mode = "extract"

            if mode == "compare":
                doc_ids = context.get("doc_ids", [])
                aspect = context.get("aspect")
                return self.compare(doc_ids, aspect or query)
            else:
                # Extract mode
                doc_id = context.get("doc_id") or (
                    context.get("doc_ids", [None])[0]
                    if context.get("doc_ids")
                    else None
                )
                schema = context.get("schema")
                return self.extract(doc_id, query, schema)

        except Exception as e:
            logger.error(f"Analysis agent error: {e}")
            processing_time = time.time() - start_time
            return AgentResponse.error_response(
                error=str(e),
                processing_time=processing_time,
            )

    def compare(
        self,
        doc_ids: List[str],
        aspect: Optional[str] = None,
    ) -> AgentResponse:
        """Compare multiple documents and find differences/similarities.

        Args:
            doc_ids: List of document IDs to compare (minimum 2)
            aspect: Optional specific aspect to focus comparison on

        Returns:
            AgentResponse with:
                - answer: Comparison analysis text
                - sources: List of compared document IDs
                - confidence: Confidence score (0.0-1.0)
                - tool_calls: Record of tool execution
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not doc_ids or len(doc_ids) < 2:
                return AgentResponse.error_response(
                    error="Compare requires at least 2 document IDs",
                    processing_time=time.time() - start_time,
                )

            # Execute analyze tool in compare mode
            result = self._analyze_tool.execute(
                mode="compare",
                doc_ids=doc_ids,
                query=aspect,
            )

            processing_time = time.time() - start_time

            if not result.success:
                return AgentResponse.error_response(
                    error=result.error or "Comparison failed",
                    processing_time=processing_time,
                )

            # Build response with sources and confidence
            answer = result.data if isinstance(result.data, str) else str(result.data)
            tokens_used = result.metadata.get("tokens_used", 0)

            # Calculate confidence based on token usage and document count
            confidence = self._calculate_confidence(tokens_used, len(doc_ids))

            tool_call = self._record_tool_call(
                tool_name="analyze",
                arguments={"mode": "compare", "doc_ids": doc_ids, "query": aspect},
                result=result.to_dict(),
            )

            return AgentResponse(
                success=True,
                answer=answer,
                sources=doc_ids,
                tool_calls=[tool_call],
                reasoning=f"Compared {len(doc_ids)} documents focusing on: {aspect or 'general comparison'}",
                tokens_used=tokens_used,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Compare error: {e}")
            return AgentResponse.error_response(
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def extract(
        self,
        doc_id: Optional[str],
        query: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Extract structured information from a document.

        Args:
            doc_id: Document ID to extract from
            query: What to extract / extraction instructions
            schema: Optional JSON schema for structured output

        Returns:
            AgentResponse with:
                - answer: Extracted information (JSON if schema provided)
                - sources: List containing the source document ID
                - confidence: Confidence score (0.0-1.0)
                - tool_calls: Record of tool execution
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not doc_id:
                return AgentResponse.error_response(
                    error="Extract requires a document ID",
                    processing_time=time.time() - start_time,
                )

            # Execute analyze tool in extract mode
            result = self._analyze_tool.execute(
                mode="extract",
                doc_id=doc_id,
                query=query,
                schema=schema,
            )

            processing_time = time.time() - start_time

            if not result.success:
                return AgentResponse.error_response(
                    error=result.error or "Extraction failed",
                    processing_time=processing_time,
                )

            # Format the answer
            extracted_data = result.data
            tokens_used = result.metadata.get("tokens_used", 0)

            if isinstance(extracted_data, dict):
                answer = json.dumps(extracted_data, ensure_ascii=False, indent=2)
            elif isinstance(extracted_data, str):
                answer = extracted_data
            else:
                answer = str(extracted_data)

            # Calculate confidence based on token usage
            confidence = self._calculate_confidence(tokens_used, 1)

            tool_call = self._record_tool_call(
                tool_name="analyze",
                arguments={
                    "mode": "extract",
                    "doc_id": doc_id,
                    "query": query,
                    "schema": schema,
                },
                result=result.to_dict(),
            )

            return AgentResponse(
                success=True,
                answer=answer,
                sources=[doc_id],
                tool_calls=[tool_call],
                reasoning=f"Extracted information from document: {query or 'general extraction'}",
                tokens_used=tokens_used,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Extract error: {e}")
            return AgentResponse.error_response(
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def _calculate_confidence(self, tokens_used: int, doc_count: int) -> float:
        """Calculate confidence score based on analysis metrics.

        Args:
            tokens_used: Number of tokens used in analysis
            doc_count: Number of documents analyzed

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.7

        # Boost for more tokens (deeper analysis)
        if tokens_used > 500:
            confidence += 0.1
        if tokens_used > 1000:
            confidence += 0.1

        # Adjust for document count in compare mode
        if doc_count >= 2:
            # More documents = more comprehensive comparison
            confidence += min(0.1, doc_count * 0.02)

        return min(1.0, confidence)

    def analyze_table(self, doc_id: str) -> AgentResponse:
        """Extract table data from a document.

        Convenience method for table extraction with predefined schema.

        Args:
            doc_id: Document ID to extract tables from

        Returns:
            AgentResponse with extracted table data
        """
        table_schema = {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "headers": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "caption": {"type": "string"},
                        },
                    },
                }
            },
        }

        return self.extract(
            doc_id=doc_id,
            query="Extract all tables from the document with their headers, rows, and captions",
            schema=table_schema,
        )

    def summarize(self, doc_id: str, focus: Optional[str] = None) -> AgentResponse:
        """Generate a summary of a document.

        Convenience method for document summarization.

        Args:
            doc_id: Document ID to summarize
            focus: Optional aspect to focus the summary on

        Returns:
            AgentResponse with document summary
        """
        query = f"Summarize this document"
        if focus:
            query += f", focusing on: {focus}"

        return self.extract(doc_id=doc_id, query=query)


def create_analysis_agent(
    config: "Config",
    raw_dir: Optional[str] = None,
    output_base: Optional[str] = None,
) -> "AnalysisAgent":
    """Factory function to create a fully configured AnalysisAgent.

    Args:
        config: Configuration object with LLM credentials
        raw_dir: Directory with raw markdown files (for MarkdownStore)
        output_base: Base directory for output files

    Returns:
        Configured AnalysisAgent instance ready for compare/extract/summarize
    """
    from pathlib import Path
    from src.storage.markdown_store import MarkdownStore

    output_dir = Path(output_base) if output_base else (Path(raw_dir) if raw_dir else Path("."))
    markdown_store = MarkdownStore(input_base=output_dir, output_base=output_dir)
    llm_client = LLMClient(config=config)

    return AnalysisAgent(llm_client=llm_client, markdown_store=markdown_store)


def search_and_analyze(
    query: str,
    index_path: str,
    config: "Config",
    mode: str = "extract",
    raw_dir: Optional[str] = None,
    top_k: int = 3,
    aspect: Optional[str] = None,
) -> "AgentResponse":
    """Search for relevant documents then analyze them in one step.

    This is the main entry point for the ``analyze`` CLI/API when the user
    does not supply explicit doc_ids. It performs a BM25 search, picks the
    top-K documents, and delegates to AnalysisAgent.

    Args:
        query: Search + analysis query.
        index_path: Path to the Tantivy index (comma-separated for multi).
        config: Config with LLM credentials.
        mode: Analysis mode — compare | extract | summarize | table.
        raw_dir: Raw markdown directory (default: index parent).
        top_k: Number of top search hits to analyze (compare uses top_k≥2).
        aspect: Optional focus for compare mode.

    Returns:
        AgentResponse with analysis result.
    """
    from pathlib import Path
    from src.search.bm25_search import create_searcher

    # 1. Search
    idx = Path(index_path.strip().split(",")[0])  # primary index
    searcher = create_searcher(index_path=idx, use_jieba=True, readonly=True)
    results = searcher.search(query, limit=top_k)

    if not results.results:
        from src.agent.base import AgentResponse
        return AgentResponse.error_response(
            error="搜索无结果，无法分析",
        )

    doc_ids = [r.doc_id for r in results.results]

    # 2. Analyze
    agent = create_analysis_agent(config=config, raw_dir=raw_dir or str(idx.parent))

    if mode == "compare":
        # Need at least 2 docs; pad to top_k if needed
        if len(doc_ids) < 2:
            from src.agent.base import AgentResponse
            return AgentResponse.error_response(
                error=f"compare 模式需要至少 2 个文档，搜索仅找到 {len(doc_ids)} 个",
            )
        return agent.compare(doc_ids=doc_ids, aspect=aspect or query)
    elif mode == "summarize":
        return agent.summarize(doc_id=doc_ids[0], focus=query)
    elif mode == "table":
        return agent.analyze_table(doc_id=doc_ids[0])
    else:  # extract
        return agent.extract(doc_id=doc_ids[0], query=query)
