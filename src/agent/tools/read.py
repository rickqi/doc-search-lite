"""Read tool for retrieving document content from Markdown files."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.base import Tool
from src.agent.tool_types import ToolResult
from src.storage.markdown_store import MarkdownStore


class ReadTool(Tool):
    """Tool for reading Markdown document content.

    Supports reading documents by:
    - doc_id: Unique identifier from search results (tries MarkdownStore, then BM25 index)
    - source_path: Original file path (tries multiple raw dirs)

    Supports pagination to avoid loading huge files at once.
    """

    def __init__(
        self,
        markdown_store: MarkdownStore,
        raw_dirs: list[str] | None = None,
        searcher=None,
    ):
        """Initialize the ReadTool.

        Args:
            markdown_store: MarkdownStore instance for accessing documents
            raw_dirs: Optional list of raw directory paths to try for source_path resolution
            searcher: Optional BM25Searcher for loading documents by doc_id from Tantivy index
        """
        self._store = markdown_store
        self._raw_dirs = raw_dirs or []
        self._searcher = searcher
        self._read_history: Dict[str, str] = {}  # P2: track {doc_id/source_path: "L{start}-L{end}"}

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "read"

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        return "读取指定文档的完整内容，支持分段读取"

    def execute(
        self,
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
        start_line: int = 0,
        max_lines: int = 500,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the read tool to retrieve document content.

        Args:
            doc_id: Document ID to load (from search results)
            source_path: Original file path to load
            start_line: Starting line number (0-indexed, default: 0)
            max_lines: Maximum number of lines to read (default: 500)
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with:
                - success: True if document found and read
                - data: File content as string
                - metadata: total_lines, file_size, truncated flag
        """
        start_time = time.time()

        # Validate parameters
        if doc_id is None and source_path is None:
            return ToolResult.fail(
                error="Must provide either doc_id or source_path",
                metadata={"error_type": "missing_parameter", "execution_time": round(time.time() - start_time, 3)},
            )

        # P2: Duplicate read detection — warn if exact same doc+page already read
        read_key = doc_id or source_path or ""
        dup_key = f"{read_key}:{start_line}"
        if read_key and dup_key in self._read_history:
            prev_range = self._read_history[read_key]
            return ToolResult.ok(
                data=(
                    f"⚠️ 此文档已读取过（{prev_range}）。"
                    "\n如需查看特定段落，请指定 start_line 参数。"
                    "\n如需查看其他文档，请搜索新的关键词。"
                ),
                metadata={"already_read": True, "doc_id": doc_id, "source_path": source_path},
            )

        # Load document
        if doc_id:
            result = self._store.load(doc_id)
            # Fallback: try loading from Tantivy index via searcher
            if result is None and self._searcher is not None:
                try:
                    full = self._searcher.get_full_content(doc_id)
                    if full is not None and full.full_content:
                        from src.storage.base import DocumentRecord
                        from datetime import datetime
                        # FullSearchResult uses 'source' not 'source_path'
                        sp = getattr(full, "source_path", None) or getattr(full, "source", None) or ""
                        record = DocumentRecord(
                            id=doc_id,
                            title=full.title,
                            source_path=Path(str(sp)) if sp else Path(""),
                            output_path=Path(str(sp)) if sp else Path(""),
                            content_hash="",
                            file_size=len(full.full_content),
                            file_mtime=datetime.now(),
                        )
                        result = (record, full.full_content)
                except Exception as fallback_exc:
                    import logging
                    logging.getLogger(__name__).warning(f"Searcher fallback failed for doc_id={doc_id}: {fallback_exc}")
        elif source_path:
            # Try with store first
            result = self._store.load_by_source(Path(source_path))
            # If not found, try each raw directory
            if result is None and self._raw_dirs:
                sp = Path(source_path).as_posix()
                for rd in self._raw_dirs:
                    full_path = Path(rd) / source_path
                    result = self._store.load_by_source(full_path)
                    if result is not None:
                        break
                    # Also try .md variant
                    md_path = Path(rd) / f"{source_path}.md"
                    if md_path.exists():
                        try:
                            content = md_path.read_text(encoding="utf-8")
                            from src.storage.base import DocumentRecord
                            record = DocumentRecord(
                                id="raw_read", title=md_path.stem,
                                source_path=md_path, file_size=len(content)
                            )
                            result = (record, content)
                        except Exception:
                            pass
                        if result is not None:
                            break
        else:
            result = None

        if result is None:
            return ToolResult.fail(
                error=f"Document not found (doc_id={doc_id}, source_path={source_path})",
                metadata={"doc_id": doc_id, "source_path": source_path, "execution_time": round(time.time() - start_time, 3)},
            )

        record, content = result

        # Strip YAML frontmatter so LLM sees clean content (no YAML noise)
        from src.converter.frontmatter import strip_frontmatter
        _, content = strip_frontmatter(content)

        # Split into lines for pagination
        lines = content.split("\n")
        total_lines = len(lines)
        file_size = record.file_size

        # Validate start_line
        if start_line < 0:
            start_line = 0
        if start_line >= total_lines:
            return ToolResult.ok(
                data="",
                metadata={
                    "doc_id": record.id,
                    "source_path": str(record.source_path),
                    "total_lines": total_lines,
                    "file_size": file_size,
                    "start_line": start_line,
                    "lines_read": 0,
                    "truncated": False,
                    "execution_time": round(time.time() - start_time, 3),
                },
            )

        # Get requested lines
        end_line = start_line + max_lines
        truncated = end_line < total_lines
        page_lines = lines[start_line:end_line]
        page_content = "\n".join(page_lines)

        # Inject TOC header on first page read for structure awareness
        if start_line == 0:
            headings = self._load_headings(doc_id=doc_id, source_path=source_path)
            toc_block = self._format_toc(headings)
            if toc_block:
                page_content = toc_block + page_content

        # P2: Record this read for duplicate detection
        if read_key:
            range_str = f"L{start_line}-L{start_line + len(page_lines)}"
            self._read_history[read_key] = range_str
            self._read_history[dup_key] = range_str

        return ToolResult.ok(
            data=page_content,
            metadata={
                "doc_id": record.id,
                "source_path": str(record.source_path),
                "title": record.title,
                "total_lines": total_lines,
                "file_size": file_size,
                "start_line": start_line,
                "lines_read": len(page_lines),
                "truncated": truncated,
                "execution_time": round(time.time() - start_time, 3),
            },
        )

    def _format_toc(self, headings: List[Dict]) -> str:
        """Format headings list as a compact TOC text block for Agent context.

        Args:
            headings: List of heading dicts with level/text/line keys.

        Returns:
            Formatted TOC string, or empty string if no headings.
        """
        if not headings:
            return ""

        lines = ["## 文档目录", ""]
        for h in headings:
            indent = "  " * (h["level"] - 1)
            lines.append(f"{indent}- {h['text']} (行 {h['line']})")
        lines.append("")
        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _load_headings(
        self,
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> List[Dict]:
        """Load headings metadata from .md.json sidecar file.

        Tries multiple resolution paths to find the .md.json file:
        1. MarkdownStore (by doc_id) → record.metadata["headings"]
        2. Raw dirs (by source_path) → {raw_dir}/{source_path}.json

        Args:
            doc_id: Document ID from search results.
            source_path: Original file path.

        Returns:
            List of heading dicts, or empty list if not found.
        """
        # Path 1: MarkdownStore by doc_id
        if self._store and doc_id:
            result = self._store.load(doc_id)
            if result is not None:
                record, _ = result
                if hasattr(record, "metadata") and isinstance(record.metadata, dict):
                    headings = record.metadata.get("headings")
                    if headings and isinstance(headings, list):
                        return headings

        # Path 2: Raw dirs by source_path → read .md.json directly
        if source_path and self._raw_dirs:
            for rd in self._raw_dirs:
                json_path = Path(rd) / f"{source_path}.json"
                if json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        headings = data.get("headings")
                        if headings and isinstance(headings, list):
                            return headings
                    except (json.JSONDecodeError, OSError):
                        pass

        return []

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
                        "start_line": {
                            "type": "integer",
                            "description": "起始行号（从0开始，默认为0）",
                            "default": 0,
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "最大读取行数（默认为500，避免加载过大文件）",
                            "default": 500,
                        },
                    },
                },
            },
        }
