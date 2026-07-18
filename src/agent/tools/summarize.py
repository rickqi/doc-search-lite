"""Summarize tool for reading and condensing document content via LLM.

Inspired by Microsoft AgenticRAG's `summarize` tool — gives the agent the ability
to quickly understand long documents without consuming the full token budget on
raw content reading. Uses the fast-tier model (DeepSeek) for cost efficiency.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.base import Tool
from src.agent.tool_types import ToolResult
from src.storage.markdown_store import MarkdownStore

logger = logging.getLogger(__name__)

# Summarization prompt — structured output for agent consumption
_SUMMARIZE_PROMPT = (
    "请总结以下文档的关键内容，提取核心要点。{focus_clause}\n\n"
    "文档内容：\n"
    "---\n"
    "{content}\n"
    "---\n\n"
    "请输出：\n"
    "1. **文档主旨**（1-2句话概括）\n"
    "2. **关键要点**（3-5条核心信息）\n"
    "3. **重要数据/标准/流程**（如有具体数值、标准、步骤，请列出）\n"
    "4. **适用场景**（这份文档适用于什么情况）\n\n"
    "保持简洁准确，只基于文档内容，不要编造。"
)


class SummarizeTool(Tool):
    """Tool for reading and summarizing document content via LLM.

    Loads a document (same resolution logic as ReadTool: doc_id or source_path),
    then calls the fast-tier LLM to produce a structured summary. This lets the
    agent quickly assess whether a document is worth reading in full, saving
    token budget on long documents.

    Supports:
    - doc_id: Unique identifier from search results
    - source_path: Original file path
    - focus: Optional focus area to emphasize in the summary
    - max_lines: Maximum lines to read before summarizing (default 500)
    """

    def __init__(
        self,
        markdown_store: MarkdownStore,
        raw_dirs: list[str] | None = None,
        searcher=None,
        llm_client=None,
    ):
        """Initialize the SummarizeTool.

        Args:
            markdown_store: MarkdownStore instance for accessing documents
            raw_dirs: Optional list of raw directory paths for source_path resolution
            searcher: Optional BM25Searcher for loading documents by doc_id from index
            llm_client: LLMClient instance for summarization (uses fast tier)
        """
        self._store = markdown_store
        self._raw_dirs = raw_dirs or []
        self._searcher = searcher
        self._llm = llm_client

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "summarize"

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        return "读取并总结文档要点，快速了解文档核心内容（节省token）"

    def execute(
        self,
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
        focus: str = "",
        max_lines: int = 500,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the summarize tool.

        Args:
            doc_id: Document ID to load (from search results)
            source_path: Original file path to load
            focus: Optional focus area to emphasize in the summary
            max_lines: Maximum lines to read before summarizing (default 500)
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with:
                - success: True if document found and summarized
                - data: Structured summary text
                - metadata: doc_id, source_path, tokens_used, execution_time
        """
        start_time = time.time()

        if doc_id is None and source_path is None:
            return ToolResult.fail(
                error="Must provide either doc_id or source_path",
                metadata={"error_type": "missing_parameter", "execution_time": round(time.time() - start_time, 3)},
            )

        if self._llm is None:
            return ToolResult.fail(
                error="LLM client not configured for summarization",
                metadata={"execution_time": round(time.time() - start_time, 3)},
            )

        # --- Load document content (same resolution logic as ReadTool) ---
        content = self._load_content(doc_id, source_path)
        if content is None:
            return ToolResult.fail(
                error=f"Document not found (doc_id={doc_id}, source_path={source_path})",
                metadata={"doc_id": doc_id, "source_path": source_path, "execution_time": round(time.time() - start_time, 3)},
            )

        # Strip YAML frontmatter to avoid --- delimiter confusion in prompt
        from src.converter.frontmatter import strip_frontmatter
        _, content = strip_frontmatter(content)

        # Truncate to max_lines to control LLM input size
        lines = content.split("\n")
        total_lines = len(lines)
        truncated = False
        if total_lines > max_lines:
            content = "\n".join(lines[:max_lines])
            truncated = True

        # --- Call LLM for summarization (fast tier for cost efficiency) ---
        focus_clause = f"\n重点关注：{focus}" if focus else ""
        prompt = _SUMMARIZE_PROMPT.format(content=content, focus_clause=focus_clause)

        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                model_tier="fast",
            )
            tokens_used = response.usage.get("total_tokens", 0)
            summary = (response.content or "").strip()

            if not summary:
                return ToolResult.fail(
                    error="LLM returned empty summary",
                    metadata={"doc_id": doc_id, "execution_time": round(time.time() - start_time, 3)},
                )

            # Append truncation notice if applicable
            if truncated:
                summary += f"\n\n⚠️ 文档较长（共{total_lines}行），仅总结了前{max_lines}行。如需完整内容请使用 read 工具。"

            return ToolResult.ok(
                data=summary,
                metadata={
                    "doc_id": doc_id or "",
                    "source_path": source_path or "",
                    "total_lines": total_lines,
                    "lines_read": min(max_lines, total_lines),
                    "truncated": truncated,
                    "tokens_used": tokens_used,
                    "focus": focus,
                    "execution_time": round(time.time() - start_time, 3),
                },
            )
        except Exception as e:
            logger.warning(f"Summarize failed for doc_id={doc_id}: {e}")
            return ToolResult.fail(
                error=f"Summarization failed: {e}",
                metadata={"doc_id": doc_id, "execution_time": round(time.time() - start_time, 3)},
            )

    def _load_content(self, doc_id: Optional[str], source_path: Optional[str]) -> Optional[str]:
        """Load document content by doc_id or source_path.

        Resolution order mirrors ReadTool:
        1. doc_id → MarkdownStore.load() → Tantivy fallback
        2. source_path → MarkdownStore.load_by_source() → raw_dirs fallback
        """
        if doc_id:
            result = self._store.load(doc_id)
            if result is None and self._searcher is not None:
                try:
                    full = self._searcher.get_full_content(doc_id)
                    if full is not None and full.full_content:
                        return full.full_content
                except Exception as e:
                    logger.warning(f"Searcher fallback failed for doc_id={doc_id}: {e}")
            if result is not None:
                return result[1]

        if source_path:
            result = self._store.load_by_source(Path(source_path))
            if result is None and self._raw_dirs:
                for rd in self._raw_dirs:
                    full_path = Path(rd) / source_path
                    result = self._store.load_by_source(full_path)
                    if result is not None:
                        return result[1]
                    # Also try .md variant
                    md_path = Path(rd) / f"{source_path}.md"
                    if md_path.exists():
                        try:
                            return md_path.read_text(encoding="utf-8")
                        except Exception:
                            pass
            if result is not None:
                return result[1]

        return None

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
                        "doc_id": {
                            "type": "string",
                            "description": "文档ID（从搜索结果获取），与source_path二选一",
                        },
                        "source_path": {
                            "type": "string",
                            "description": "原始文件路径，与doc_id二选一",
                        },
                        "focus": {
                            "type": "string",
                            "description": "总结时的关注重点（可选），如'申请流程'、'报销标准'",
                            "default": "",
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "最大读取行数（默认500，避免输入过长）",
                            "default": 500,
                        },
                    },
                },
            },
        }
