"""GrepTool - Direct raw file search (DCI-Agent-Lite paradigm).

Searches raw markdown files using Python regex (cross-platform, no external deps).
This is the core primitive of Direct Corpus Interaction: the agent searches
raw text directly without any information loss through indexing or embedding.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List

from src.agent.base import Tool, ToolCache, ToolResult


class GrepTool(Tool):
    """Tool for searching raw markdown files using regex patterns.

    This implements the DCI-Agent-Lite zero-index retrieval paradigm:
    the agent searches raw text files directly, like a human using grep.

    No pre-built index needed. Searches all .md files in the configured directory.
    """

    def __init__(
        self,
        raw_dir: Path,
        max_results: int = 50,
        context_lines: int = 2,
        max_file_size: int = 5 * 1024 * 1024,
    ) -> None:
        """Initialize GrepTool.

        Args:
            raw_dir: Directory containing raw markdown files to search.
            max_results: Maximum number of match results to return.
            context_lines: Number of context lines before/after match.
            max_file_size: Skip files larger than this (bytes).
        """
        self._raw_dir = Path(raw_dir)
        self._max_results = max_results
        self._context_lines = context_lines
        self._max_file_size = max_file_size
        self._cache: ToolCache | None = None

    def set_cache(self, cache: ToolCache) -> None:
        """Attach an optional ToolCache for result caching (opt-in)."""
        self._cache = cache

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "grep"

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        return (
            "直接搜索原始文档内容，支持正则表达式。"
            "搜索原始Markdown文件，不经过索引，适合精确查找。"
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute grep search on raw markdown files.

        Args:
            pattern: Regex pattern to search for (required).
            case_sensitive: Whether search is case-sensitive (default: False).
            max_results: Override max results for this query.
            file_filter: Optional glob pattern to filter files (e.g. "*.docx.md").

        Returns:
            ToolResult with matching files, line numbers, and context.
        """
        # Opt-in cache: check before doing any work
        if self._cache is not None:
            cache_key = ToolCache.make_key(self.name, kwargs)
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Return cached result with cache_hit metadata
                hit_meta = {**cached.metadata, "cache_hit": True}
                return ToolResult(
                    success=cached.success,
                    data=cached.data,
                    error=cached.error,
                    metadata=hit_meta,
                )

        start_time = time.time()

        pattern = kwargs.get("pattern", "")
        case_sensitive = kwargs.get("case_sensitive", False)
        max_results = kwargs.get("max_results", self._max_results)
        file_filter = kwargs.get("file_filter", "*.md")

        if not pattern or not isinstance(pattern, str) or not pattern.strip():
            return ToolResult.fail(
                error="Parameter 'pattern' is required and must be a non-empty string",
                metadata={"execution_time": time.time() - start_time},
            )

        # Validate raw_dir exists
        if not self._raw_dir.is_dir():
            result = ToolResult.fail(
                error=f"Raw directory does not exist: {self._raw_dir}",
                metadata={"execution_time": time.time() - start_time},
            )
            if self._cache is not None:
                self._cache.put(cache_key, result)
            return result

        # Compile regex
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as exc:
            result = ToolResult.fail(
                error=f"Invalid regex pattern: {exc}",
                metadata={"execution_time": time.time() - start_time},
            )
            if self._cache is not None:
                self._cache.put(cache_key, result)
            return result

        # Search files
        results: List[Dict[str, Any]] = []
        files_searched = 0

        for md_file in sorted(self._raw_dir.rglob(file_filter)):
            if not md_file.is_file():
                continue

            try:
                file_size = md_file.stat().st_size
            except OSError:
                continue

            if file_size > self._max_file_size:
                continue

            files_searched += 1

            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            lines = content.split("\n")

            for line_no, line in enumerate(lines):
                if regex.search(line):
                    rel_path = md_file.relative_to(self._raw_dir)

                    start = max(0, line_no - self._context_lines)
                    ctx_before = lines[start:line_no]
                    ctx_after = lines[
                        line_no + 1 : line_no + 1 + self._context_lines
                    ]

                    results.append(
                        {
                            "file": str(rel_path),
                            "line_no": line_no,
                            "match": line.strip(),
                            "context_before": [l.strip() for l in ctx_before],
                            "context_after": [l.strip() for l in ctx_after],
                        }
                    )

                    if len(results) >= max_results:
                        break

            if len(results) >= max_results:
                break

        execution_time = time.time() - start_time

        if not results:
            # Detect multi-word patterns that may fail due to literal spaces
            hint_extra = ""
            if " " in pattern and not any(c in pattern for c in ".+*?[](){}|^$\\"):
                words = [w.strip() for w in pattern.split() if w.strip()]
                if len(words) >= 2:
                    or_pattern = "|".join(words)
                    hint_extra = (
                        f"\n4) 多词查询：尝试 OR 模式 \"{or_pattern}\""
                        f"（空格在正则中匹配字面空格，多词用 | 分隔可匹配任一词）"
                    )
            result = ToolResult.ok(
                data=(
                    "No matches found."
                    "\n\n⚠️ 正则无匹配建议："
                    "\n1) 放宽正则（去掉锚点、使用 .* 通配）"
                    "\n2) 降低 case_sensitive"
                    "\n3) 扩大 file_filter 范围（如 '*.md' 替代 '*.docx.md'）"
                    f"{hint_extra}"
                ),
                metadata={
                    "total_matches": 0,
                    "files_searched": files_searched,
                    "pattern": pattern,
                    "execution_time": execution_time,
                },
            )
            if self._cache is not None:
                self._cache.put(cache_key, result)
            return result

        # Format output as grep-like lines
        output_lines = []
        for r in results:
            output_lines.append(f"{r['file']}:{r['line_no']}: {r['match']}")
            # Append context lines
            for ctx in r["context_after"]:
                if ctx:
                    output_lines.append(f"  {r['file']}:{r['line_no']+1}- {ctx}")

        result = ToolResult.ok(
            data="\n".join(output_lines),
            metadata={
                "total_matches": len(results),
                "files_searched": files_searched,
                "pattern": pattern,
                "execution_time": execution_time,
            },
        )
        if self._cache is not None:
            self._cache.put(cache_key, result)
        return result

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format.

        Returns:
            Dict[str, Any]: Tool definition in OpenAI format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "搜索模式，支持正则表达式。"
                                "如：'年假.*规定'、'投诉处理'"
                            ),
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "是否区分大小写（默认不区分）",
                            "default": False,
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回结果数（默认50）",
                            "default": 50,
                        },
                        "file_filter": {
                            "type": "string",
                            "description": "文件过滤模式（如 '*.docx.md'）",
                            "default": "*.md",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }
