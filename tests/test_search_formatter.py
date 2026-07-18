"""Unit tests for ResultFormatter."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.search.result_formatter import SearchResult, ResultFormatter, format_results


@pytest.fixture
def sample_results():
    """Create sample search results."""
    now = datetime.now()
    return [
        SearchResult(
            title="测试文档 1",
            score=0.95,
            snippet="这是一个关于测试的文档内容，包含搜索相关的信息。",
            source=Path("/data/docs/test1.md"),
            timestamp=now,
            metadata={"author": "测试作者1", "tags": ["测试", "文档"]},
        ),
        SearchResult(
            title="另一个测试文件",
            score=0.87,
            snippet="这是另一个文件的内容，也包含了测试关键词。",
            source=Path("/data/docs/subdir/test2.md"),
            timestamp=now,
            metadata={"author": "测试作者2"},
        ),
        SearchResult(
            title="相关资料",
            score=0.75,
            snippet="这份资料与搜索查询相关。",
            source=Path("/data/docs/material.pdf"),
            timestamp=now,
        ),
    ]


@pytest.fixture
def formatter():
    """Create a ResultFormatter instance."""
    return ResultFormatter()


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """SearchResult should be created with all fields."""
        now = datetime.now()
        result = SearchResult(
            title="Test",
            score=0.5,
            snippet="Snippet",
            source=Path("/test.md"),
            timestamp=now,
        )

        assert result.title == "Test"
        assert result.score == 0.5
        assert result.snippet == "Snippet"
        assert result.source == Path("/test.md")
        assert result.timestamp == now
        assert result.highlighted is False

    def test_search_result_default_timestamp(self):
        """SearchResult should use current timestamp if not provided."""
        before = datetime.now()
        result = SearchResult(
            title="Test", score=0.5, snippet="Snippet", source=Path("/test.md")
        )
        after = datetime.now()

        assert result.timestamp is not None
        assert before <= result.timestamp <= after

    def test_search_result_default_metadata(self):
        """SearchResult should have empty dict as default metadata."""
        result = SearchResult(
            title="Test", score=0.5, snippet="Snippet", source=Path("/test.md")
        )

        assert result.metadata == {}


class TestResultFormatterInit:
    """Tests for ResultFormatter initialization."""

    def test_init_without_pattern(self):
        """Formatter should initialize without highlight pattern."""
        formatter = ResultFormatter()

        assert formatter.highlight_pattern is None

    def test_init_with_pattern(self):
        """Formatter should initialize with highlight pattern."""
        formatter = ResultFormatter(highlight_pattern="test")

        assert formatter.highlight_pattern == "test"


class TestHighlightText:
    """Tests for highlight_text method."""

    def test_highlight_text_basic(self, formatter):
        """Should highlight matching text."""
        formatter.highlight_pattern = "test"
        text = "This is a test document."

        result = formatter.highlight_text(text)

        assert "**test**" in result

    def test_highlight_text_case_insensitive(self, formatter):
        """Highlighting should be case-insensitive."""
        formatter.highlight_pattern = "test"
        text = "This is a Test document."

        result = formatter.highlight_text(text)

        assert "**Test**" in result

    def test_highlight_text_multiple_matches(self, formatter):
        """Should highlight multiple occurrences."""
        formatter.highlight_pattern = "test"
        text = "test document test again test"

        result = formatter.highlight_text(text)

        assert result.count("**test**") == 3

    def test_highlight_text_no_pattern(self, formatter):
        """Should return text as-is when no pattern provided."""
        text = "This is a test document."

        result = formatter.highlight_text(text)

        assert result == text

    def test_highlight_text_empty_string(self, formatter):
        """Should handle empty string."""
        result = formatter.highlight_text("")

        assert result == ""

    def test_highlight_text_truncation(self, formatter):
        """Should truncate text to max_length."""
        long_text = "a" * 300
        result = formatter.highlight_text(long_text, max_length=100)

        assert len(result) <= 103  # 100 chars + "..."

    def test_highlight_text_truncation_preserves_ellipsis(self, formatter):
        """Should add ellipsis when truncated."""
        long_text = "a" * 300
        result = formatter.highlight_text(long_text, max_length=50)

        assert result.endswith("...")

    def test_highlight_text_invalid_pattern(self, formatter):
        """Should handle invalid regex pattern gracefully."""
        formatter.highlight_pattern = "[invalid("
        text = "This is a test document."

        result = formatter.highlight_text(text)

        assert result == text


class TestFormatJson:
    """Tests for format_json method."""

    def test_format_json_basic(self, formatter, sample_results):
        """Should format results as valid JSON."""
        output = formatter.format_json(sample_results)

        data = json.loads(output)

        assert "results" in data
        assert len(data["results"]) == 3
        assert data["results"][0]["title"] == "测试文档 1"
        assert data["results"][0]["score"] == 0.95

    def test_format_json_includes_snippet(self, formatter, sample_results):
        """Should include highlighted snippet in JSON."""
        formatter.highlight_pattern = "测试"
        output = formatter.format_json(sample_results)

        data = json.loads(output)

        assert "snippet" in data["results"][0]
        assert "测试" in data["results"][0]["snippet"]

    def test_format_json_includes_source(self, formatter, sample_results):
        """Should include source path in JSON."""
        output = formatter.format_json(sample_results)

        data = json.loads(output)

        assert "source" in data["results"][0]
        assert str(sample_results[0].source) in data["results"][0]["source"]

    def test_format_json_includes_timestamp(self, formatter, sample_results):
        """Should include ISO format timestamp in JSON."""
        output = formatter.format_json(sample_results)

        data = json.loads(output)

        assert "timestamp" in data["results"][0]
        assert data["results"][0]["timestamp"] is not None

    def test_format_json_with_metadata(self, formatter, sample_results):
        """Should include metadata when present."""
        output = formatter.format_json(sample_results)

        data = json.loads(output)

        assert "metadata" in data["results"][0]
        assert data["results"][0]["metadata"]["author"] == "测试作者1"

    def test_format_json_no_metadata(self, formatter):
        """Should not include metadata field when absent."""
        result = SearchResult(
            title="Test", score=0.5, snippet="Snippet", source=Path("/test.md")
        )

        output = formatter.format_json([result])

        data = json.loads(output)

        assert "metadata" not in data["results"][0]

    def test_format_json_empty_results(self, formatter):
        """Should handle empty results list."""
        output = formatter.format_json([])

        data = json.loads(output)

        assert data["results"] == []
        assert "summary" not in data

    def test_format_json_with_summary(self, formatter, sample_results):
        """Should include summary when requested."""
        output = formatter.format_json(sample_results, include_summary=True)

        data = json.loads(output)

        assert "summary" in data
        assert "3" in data["summary"]  # Should mention number of results


class TestFormatText:
    """Tests for format_text method."""

    def test_format_text_basic(self, formatter, sample_results):
        """Should format results as human-readable text."""
        output = formatter.format_text(sample_results)

        assert "1. 测试文档 1" in output
        assert "score: 0.95" in output
        assert "来源:" in output

    def test_format_text_numbered(self, formatter, sample_results):
        """Should number results sequentially."""
        output = formatter.format_text(sample_results)

        assert "1." in output
        assert "2." in output
        assert "3." in output

    def test_format_text_includes_snippet(self, formatter, sample_results):
        """Should include snippet in text output."""
        output = formatter.format_text(sample_results)

        assert "搜索相关的信息" in output

    def test_format_text_with_highlight(self, formatter):
        """Should highlight query terms in text output."""
        formatter.highlight_pattern = "测试"
        result = SearchResult(
            title="测试", score=0.9, snippet="这是测试内容", source=Path("/test.md")
        )

        output = formatter.format_text([result])

        assert "**测试**" in output

    def test_format_text_empty_results(self, formatter):
        """Should handle empty results list."""
        output = formatter.format_text([])

        assert "No results found." in output

    def test_format_text_with_summary(self, formatter, sample_results):
        """Should include summary when requested."""
        output = formatter.format_text(sample_results, include_summary=True)

        assert "3 个结果" in output or "找到 3" in output

    def test_format_text_includes_timestamp(self, formatter, sample_results):
        """Should include timestamp when available."""
        output = formatter.format_text(sample_results)

        assert "时间:" in output

    def test_format_text_chinese_source_label(self, formatter, sample_results):
        """Should use Chinese label for source."""
        output = formatter.format_text(sample_results)

        assert "来源:" in output


class TestFormatMarkdown:
    """Tests for format_markdown method."""

    def test_format_markdown_basic(self, formatter, sample_results):
        """Should format results as valid Markdown."""
        output = formatter.format_markdown(sample_results)

        assert "# 搜索结果" in output
        assert "## 测试文档 1" in output
        assert "**相关性得分:** 0.95" in output

    def test_format_markdown_headers(self, formatter, sample_results):
        """Should use proper Markdown headers."""
        output = formatter.format_markdown(sample_results)

        assert output.startswith("#")
        assert "##" in output

    def test_format_markdown_code_blocks(self, formatter, sample_results):
        """Should use code blocks for source paths."""
        output = formatter.format_markdown(sample_results)

        assert "`/data/docs/test1.md`" in output

    def test_format_markdown_blockquotes(self, formatter, sample_results):
        """Should use blockquotes for snippets."""
        output = formatter.format_markdown(sample_results)

        assert "> " in output

    def test_format_markdown_with_highlight(self, formatter):
        """Should preserve highlighting in Markdown."""
        formatter.highlight_pattern = "测试"
        result = SearchResult(
            title="测试", score=0.9, snippet="这是测试内容", source=Path("/test.md")
        )

        output = formatter.format_markdown([result])

        assert "**测试**" in output

    def test_format_markdown_empty_results(self, formatter):
        """Should handle empty results list."""
        output = formatter.format_markdown([])

        assert "没有找到匹配的结果。" in output

    def test_format_markdown_with_summary(self, formatter, sample_results):
        """Should include summary header when requested."""
        output = formatter.format_markdown(sample_results, include_summary=True)

        assert "## 摘要" in output

    def test_format_markdown_horizontal_rule(self, formatter, sample_results):
        """Should separate results with horizontal rules."""
        output = formatter.format_markdown(sample_results)

        assert "---" in output

    def test_format_markdown_with_metadata(self, formatter, sample_results):
        """Should format metadata as list."""
        output = formatter.format_markdown(sample_results)

        assert "**元数据:**" in output
        assert "author:" in output

    def test_format_markdown_without_metadata(self, formatter):
        """Should not include metadata section when absent."""
        result = SearchResult(
            title="Test", score=0.5, snippet="Snippet", source=Path("/test.md")
        )

        output = formatter.format_markdown([result])

        assert "**元数据:**" not in output


class TestGenerateSummary:
    """Tests for _generate_summary method."""

    def test_generate_summary_count(self, formatter, sample_results):
        """Summary should mention number of results."""
        summary = formatter._generate_summary(sample_results)

        assert "3" in summary

    def test_generate_summary_avg_score(self, formatter, sample_results):
        """Summary should include average score."""
        summary = formatter._generate_summary(sample_results)

        # Average of 0.95, 0.87, 0.75 = 0.857
        assert "0.86" in summary or "0.85" in summary

    def test_generate_summary_sources(self, formatter, sample_results):
        """Summary should mention top sources."""
        summary = formatter._generate_summary(sample_results)

        assert "主要来源:" in summary
        assert "test1.md" in summary

    def test_generate_summary_empty(self, formatter):
        """Summary should handle empty results."""
        summary = formatter._generate_summary([])

        assert "未找到结果" in summary

    def test_generate_summary_single_result(self, formatter):
        """Summary should work with single result."""
        result = SearchResult(
            title="Test", score=0.5, snippet="Snippet", source=Path("/test.md")
        )

        summary = formatter._generate_summary([result])

        assert "1 个结果" in summary or "找到 1" in summary


class TestFormatResultsConvenience:
    """Tests for format_results convenience function."""

    def test_format_results_text(self, sample_results):
        """Should format as text by default."""
        output = format_results(sample_results, format_type="text")

        assert "score: 0.95" in output

    def test_format_results_json(self, sample_results):
        """Should format as JSON when requested."""
        output = format_results(sample_results, format_type="json")

        data = json.loads(output)
        assert "results" in data

    def test_format_results_markdown(self, sample_results):
        """Should format as Markdown when requested."""
        output = format_results(sample_results, format_type="markdown")

        assert "# 搜索结果" in output

    def test_format_results_with_highlight(self, sample_results):
        """Should highlight with provided pattern."""
        output = format_results(
            sample_results, format_type="text", highlight_pattern="测试"
        )

        assert "**测试**" in output

    def test_format_results_invalid_format(self, sample_results):
        """Should raise ValueError for invalid format."""
        with pytest.raises(ValueError, match="Unknown format_type"):
            format_results(sample_results, format_type="invalid")

    def test_format_results_with_summary(self, sample_results):
        """Should include summary when requested."""
        output = format_results(
            sample_results, format_type="text", include_summary=True
        )

        assert "3" in output or "三" in output


class TestSpecialCharacters:
    """Tests for special character handling."""

    def test_json_unicode_handling(self, formatter):
        """JSON should handle Unicode characters correctly."""
        result = SearchResult(
            title="中文标题",
            score=0.9,
            snippet="中文内容",
            source=Path("/测试.md"),
        )

        output = formatter.format_json([result])
        data = json.loads(output)

        assert data["results"][0]["title"] == "中文标题"

    def test_text_unicode_handling(self, formatter):
        """Text format should preserve Unicode."""
        result = SearchResult(
            title="中文标题", score=0.9, snippet="中文内容", source=Path("/测试.md")
        )

        output = formatter.format_text([result])

        assert "中文标题" in output
        assert "测试.md" in output

    def test_markdown_unicode_handling(self, formatter):
        """Markdown format should preserve Unicode."""
        result = SearchResult(
            title="中文标题", score=0.9, snippet="中文内容", source=Path("/测试.md")
        )

        output = formatter.format_markdown([result])

        assert "中文标题" in output
        assert "测试.md" in output

    def test_special_characters_in_snippet(self, formatter):
        """Should handle special characters in snippets."""
        result = SearchResult(
            title="Test",
            score=0.9,
            snippet='Content with "quotes", <brackets>, and {braces}',
            source=Path("/test.md"),
        )

        output = formatter.format_json([result])
        data = json.loads(output)

        assert '"quotes"' in data["results"][0]["snippet"]
