"""Analyze tool for document comparison and information extraction.

Supports two modes:
- compare: Compare multiple documents and find differences/similarities
- extract: Extract structured information from documents using LLM
"""

import json
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.agent.base import Tool
from src.agent.tool_types import ToolResult
from src.storage.markdown_store import MarkdownStore


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client that can be used with AnalyzeTool.

    This allows any LLM client implementation to be used,
    as long as it follows this interface.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        ...


class AnalyzeTool(Tool):
    """Tool for analyzing documents with LLM support.

    Supports two analysis modes:
    1. compare: Compare multiple documents to find similarities and differences
    2. extract: Extract structured information from a document based on a schema
    """

    def __init__(
        self,
        markdown_store: MarkdownStore,
        llm_client: Optional[LLMClientProtocol] = None,
    ):
        """Initialize the AnalyzeTool.

        Args:
            markdown_store: MarkdownStore instance for accessing documents
            llm_client: Optional LLM client for analysis (can be set later)
        """
        self._store = markdown_store
        self._llm = llm_client

    def set_llm_client(self, llm_client: LLMClientProtocol) -> None:
        """Set the LLM client.

        Args:
            llm_client: LLM client instance
        """
        self._llm = llm_client

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "analyze"

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        return "分析文档内容，支持对比分析和结构化信息提取"

    def execute(
        self,
        mode: str = "extract",
        doc_ids: Optional[List[str]] = None,
        doc_id: Optional[str] = None,
        query: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the analyze tool.

        Args:
            mode: Analysis mode - 'compare' or 'extract' (default: 'extract')
            doc_ids: List of document IDs to compare (for compare mode)
            doc_id: Single document ID (for extract mode)
            query: Specific query/question for analysis
            schema: JSON schema for structured extraction (for extract mode)
            **kwargs: Additional arguments

        Returns:
            ToolResult with:
                - success: True if analysis completed
                - data: Analysis result (JSON for extract, text for compare)
                - metadata: mode, documents analyzed, tokens used
        """
        # Validate mode
        if mode not in ("compare", "extract"):
            return ToolResult.fail(
                error=f"Invalid mode '{mode}'. Must be 'compare' or 'extract'",
                metadata={"error_type": "invalid_mode", "mode": mode},
            )

        # Check LLM client
        if self._llm is None:
            return ToolResult.fail(
                error="LLM client not configured. Please set LLM client before using analyze tool.",
                metadata={"error_type": "llm_not_configured"},
            )

        if mode == "compare":
            return self._execute_compare(doc_ids, query, **kwargs)
        else:
            return self._execute_extract(
                doc_id or (doc_ids[0] if doc_ids else None), query, schema, **kwargs
            )

    def _execute_compare(
        self,
        doc_ids: Optional[List[str]],
        query: Optional[str],
        **kwargs: Any,
    ) -> ToolResult:
        """Execute compare mode - compare multiple documents.

        Args:
            doc_ids: List of document IDs to compare
            query: Optional specific comparison focus
            **kwargs: Additional arguments

        Returns:
            ToolResult with comparison analysis
        """
        assert self._llm is not None  # Checked in execute()

        # Validate doc_ids
        if not doc_ids or len(doc_ids) < 2:
            return ToolResult.fail(
                error="Compare mode requires at least 2 documents (doc_ids)",
                metadata={"error_type": "insufficient_documents", "doc_ids": doc_ids},
            )

        # Load all documents
        documents = []
        for doc_id in doc_ids:
            result = self._store.load(doc_id)
            if result is None:
                return ToolResult.fail(
                    error=f"Document not found: {doc_id}",
                    metadata={"error_type": "document_not_found", "doc_id": doc_id},
                )
            record, content = result
            documents.append(
                {
                    "id": record.id,
                    "title": record.title,
                    "source_path": str(record.source_path),
                    "content": content,
                }
            )

        # Build comparison prompt
        prompt = self._build_compare_prompt(documents, query)

        # Get LLM analysis
        try:
            analysis = self._llm.generate(
                prompt,
                system_prompt="你是一个专业的文档分析助手。请用中文进行分析和回答。",
                temperature=0.3,
                max_tokens=3000,
            )
            tokens_used = self._llm.count_tokens(prompt + analysis)
        except Exception as e:
            return ToolResult.fail(
                error=f"LLM analysis failed: {str(e)}",
                metadata={"error_type": "llm_error", "exception": str(e)},
            )

        return ToolResult.ok(
            data=analysis,
            metadata={
                "mode": "compare",
                "doc_ids": doc_ids,
                "query": query,
                "tokens_used": tokens_used,
            },
        )

    def _execute_extract(
        self,
        doc_id: Optional[str],
        query: Optional[str],
        schema: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolResult:
        """Execute extract mode - extract structured information.

        Args:
            doc_id: Document ID to extract from
            query: What to extract / extraction instructions
            schema: JSON schema for structured output
            **kwargs: Additional arguments

        Returns:
            ToolResult with extracted data as JSON
        """
        assert self._llm is not None  # Checked in execute()

        # Validate doc_id
        if not doc_id:
            return ToolResult.fail(
                error="Extract mode requires a document ID (doc_id or doc_ids[0])",
                metadata={"error_type": "missing_document_id"},
            )

        # Load document
        result = self._store.load(doc_id)
        if result is None:
            return ToolResult.fail(
                error=f"Document not found: {doc_id}",
                metadata={"error_type": "document_not_found", "doc_id": doc_id},
            )

        record, content = result

        # Build extraction prompt
        prompt = self._build_extract_prompt(record.title, content, query, schema)

        # Get LLM analysis
        try:
            analysis = self._llm.generate(
                prompt,
                system_prompt="你是一个专业的信息提取助手。请严格按照要求的格式返回结果，如果是JSON格式，确保返回有效的JSON。",
                temperature=0.1,
                max_tokens=2000,
            )
            tokens_used = self._llm.count_tokens(prompt + analysis)
        except Exception as e:
            return ToolResult.fail(
                error=f"LLM analysis failed: {str(e)}",
                metadata={"error_type": "llm_error", "exception": str(e)},
            )

        # Try to parse as JSON if schema was provided
        extracted_data = analysis
        if schema:
            try:
                # Try to extract JSON from the response
                extracted_data = self._extract_json(analysis)
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, keep raw text
                pass

        return ToolResult.ok(
            data=extracted_data,
            metadata={
                "mode": "extract",
                "doc_id": doc_id,
                "query": query,
                "has_schema": schema is not None,
                "tokens_used": tokens_used,
            },
        )

    def _build_compare_prompt(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str],
    ) -> str:
        """Build the comparison prompt for LLM.

        Args:
            documents: List of document info dicts
            query: Optional specific comparison focus

        Returns:
            Formatted prompt string
        """
        prompt_parts = ["请对比分析以下文档:\n"]

        for i, doc in enumerate(documents, 1):
            prompt_parts.append(f"--- 文档 {i}: {doc['title']} ---")
            prompt_parts.append(f"来源: {doc['source_path']}")
            prompt_parts.append(
                f"内容:\n{doc['content'][:3000]}..."
            )  # Limit content length
            prompt_parts.append("")

        if query:
            prompt_parts.append(f"特别关注: {query}")
            prompt_parts.append("")

        prompt_parts.append("请分析并总结:")
        prompt_parts.append("1. 各文档的主要内容")
        prompt_parts.append("2. 文档之间的相似之处")
        prompt_parts.append("3. 文档之间的差异")
        prompt_parts.append("4. 综合结论")

        return "\n".join(prompt_parts)

    def _build_extract_prompt(
        self,
        title: str,
        content: str,
        query: Optional[str],
        schema: Optional[Dict[str, Any]],
    ) -> str:
        """Build the extraction prompt for LLM.

        Args:
            title: Document title
            content: Document content
            query: What to extract
            schema: JSON schema for output

        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"从以下文档中提取信息:\n"]
        prompt_parts.append(f"文档标题: {title}")
        prompt_parts.append(f"内容:\n{content[:4000]}...")  # Limit content length
        prompt_parts.append("")

        if query:
            prompt_parts.append(f"提取要求: {query}")

        if schema:
            prompt_parts.append(
                f"\n输出格式 (JSON Schema):\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
            )
            prompt_parts.append("\n请严格按照上述JSON格式返回结果。")
        else:
            prompt_parts.append("\n请以清晰的结构化格式返回提取的信息。")

        return "\n".join(prompt_parts)

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response.

        Handles cases where LLM might wrap JSON in markdown code blocks
        or include extra text.

        Args:
            text: Raw LLM response text

        Returns:
            Parsed JSON data

        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from code blocks
        import re

        # Look for ```json ... ``` or ``` ... ```
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, text)

        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Look for { } or [ ] patterns
        brace_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
        matches = re.findall(brace_pattern, text)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Fall back to original text
        raise json.JSONDecodeError("No valid JSON found", text, 0)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["compare", "extract"],
                            "description": "分析模式: compare=对比多文档, extract=提取结构化信息",
                            "default": "extract",
                        },
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "文档ID列表（compare模式至少需要2个，extract模式可传入单个ID）",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "单个文档ID（extract模式，与doc_ids二选一）",
                        },
                        "query": {
                            "type": "string",
                            "description": "分析查询/提取指令，描述你想要分析或提取的内容",
                        },
                        "schema": {
                            "type": "object",
                            "description": "JSON Schema格式，定义extract模式下期望的输出结构",
                        },
                    },
                },
            },
        }
