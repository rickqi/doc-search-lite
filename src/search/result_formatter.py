"""Result formatting module for search results.

This module provides the ResultFormatter class for formatting search results
in various output formats (JSON, text, Markdown) with support for
highlighting query terms and generating summaries.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    score: float
    snippet: str
    source: str | Path
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    highlighted: bool = False

    def __post_init__(self):
        """Set default timestamp if not provided, normalize source path."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        # Normalize source to Path object
        if isinstance(self.source, str):
            self.source = Path(self.source)


class ResultFormatter:
    """Formats search results in various output formats."""

    def __init__(self, highlight_pattern: str | None = None):
        """Initialize the ResultFormatter.

        Args:
            highlight_pattern: Regex pattern for terms to highlight in results.
        """
        self.highlight_pattern = highlight_pattern

    def highlight_text(self, text: str, max_length: int = 200) -> str:
        """Highlight query terms in the given text.

        Args:
            text: Text to highlight.
            max_length: Maximum length of text to return.

        Returns:
            Text with query terms wrapped in ** markers, truncated to max_length.
        """
        if not text:
            return ""

        # Truncate to max_length if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Highlight query terms if pattern provided
        if self.highlight_pattern:
            try:
                # Wrap matches in ** markers
                text = re.sub(
                    f"({self.highlight_pattern})", r"**\1**", text, flags=re.IGNORECASE
                )
            except re.error:
                # If pattern is invalid, return text without highlighting
                pass

        return text

    def format_json(
        self, results: list[SearchResult], include_summary: bool = False
    ) -> str:
        """Format search results as JSON.

        Args:
            results: List of search results to format.
            include_summary: Whether to include a summary.

        Returns:
            JSON string representation of the results.
        """
        output: dict[str, Any] = {"results": []}

        for result in results:
            formatted_result = {
                "title": result.title,
                "score": result.score,
                "snippet": self.highlight_text(result.snippet),
                "source": str(result.source),
                "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            }

            if result.metadata:
                formatted_result["metadata"] = result.metadata

            output["results"].append(formatted_result)

        if include_summary and results:
            output["summary"] = self._generate_summary(results)  # type: ignore[assignment]

        return json.dumps(output, ensure_ascii=False, indent=2)

    def format_text(
        self, results: list[SearchResult], include_summary: bool = False
    ) -> str:
        """Format search results as human-readable text.

        Args:
            results: List of search results to format.
            include_summary: Whether to include a summary.

        Returns:
            Human-readable text representation of the results.
        """
        lines = []

        if include_summary and results:
            lines.append(self._generate_summary(results))
            lines.append("")

        if not results:
            lines.append("No results found.")
            return "\n".join(lines)

        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.title} (score: {result.score:.2f})")
            lines.append(f"   {self.highlight_text(result.snippet)}")
            lines.append(f"   来源: {result.source}")
            if result.timestamp:
                lines.append(
                    f"   时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            lines.append("")

        return "\n".join(lines)

    def format_markdown(
        self, results: list[SearchResult], include_summary: bool = False
    ) -> str:
        """Format search results as Markdown.

        Args:
            results: List of search results to format.
            include_summary: Whether to include a summary.

        Returns:
            Markdown formatted representation of the results.
        """
        lines = []

        if include_summary and results:
            lines.append(f"## 摘要\n{self._generate_summary(results)}\n")

        if not results:
            lines.append("# 搜索结果")
            lines.append("\n没有找到匹配的结果。")
            return "\n".join(lines)

        lines.append("# 搜索结果\n")

        for result in results:
            lines.append(f"## {result.title}")
            lines.append(f"\n**相关性得分:** {result.score:.2f}")
            lines.append(f"\n**来源:** `{result.source.as_posix()}`")  # type: ignore[attr-defined]

            if result.timestamp:
                lines.append(
                    f"**时间:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            lines.append("\n**片段:**")
            lines.append(f"\n> {self.highlight_text(result.snippet)}")

            if result.metadata:
                lines.append("\n**元数据:**")
                for key, value in result.metadata.items():
                    lines.append(f"  - {key}: {value}")

            lines.append("\n---\n")

        return "\n".join(lines)

    def _generate_summary(self, results: list[SearchResult]) -> str:
        """Generate a summary of the search results.

        Args:
            results: List of search results.

        Returns:
            Summary string.
        """
        if not results:
            return "未找到结果。"

        count = len(results)
        avg_score = sum(r.score for r in results) / count

        summary_parts = [
            f"找到 {count} 个结果",
            f"平均相关性得分: {avg_score:.2f}",
        ]

        # Get top sources
        sources = {}
        for result in results:
            source_path = str(result.source)
            sources[source_path] = sources.get(source_path, 0) + 1

        if sources:
            top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:3]
            source_list = ", ".join(
                [f"{Path(src).name} ({count})" for src, count in top_sources]
            )
            summary_parts.append(f"主要来源: {source_list}")

        return " | ".join(summary_parts)


def format_results(
    results: list[SearchResult],
    format_type: str = "text",
    highlight_pattern: str | None = None,
    include_summary: bool = False,
) -> str:
    """Format search results in the specified format.

    Convenience function that creates a ResultFormatter and formats results.

    Args:
        results: List of search results to format.
        format_type: Output format type ('json', 'text', or 'markdown').
        highlight_pattern: Regex pattern for terms to highlight.
        include_summary: Whether to include a summary.

    Returns:
        Formatted search results.

    Raises:
        ValueError: If format_type is not 'json', 'text', or 'markdown'.
    """
    formatter = ResultFormatter(highlight_pattern=highlight_pattern)

    if format_type == "json":
        return formatter.format_json(results, include_summary=include_summary)
    elif format_type == "text":
        return formatter.format_text(results, include_summary=include_summary)
    elif format_type == "markdown":
        return formatter.format_markdown(results, include_summary=include_summary)
    else:
        raise ValueError(
            f"Unknown format_type: {format_type}. "
            "Must be 'json', 'text', or 'markdown'."
        )
