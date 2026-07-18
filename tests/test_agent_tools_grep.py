"""Unit tests for GrepTool class.

Tests cover:
- GrepTool initialization and properties
- Execute method with various patterns and options
- Context lines, file filtering, max results
- Error handling (empty pattern, invalid regex, missing dir)
- Edge cases (large files, no matches, case sensitivity)
- OpenAI tool format generation
"""

from pathlib import Path

import pytest

from src.agent.tool_types import ToolResult
from src.agent.tools.grep import GrepTool


class TestGrepToolInit:
    """Test GrepTool initialization."""

    def test_init_default_values(self, tmp_path):
        """Test initialization with default values."""
        tool = GrepTool(raw_dir=tmp_path)
        assert tool._raw_dir == tmp_path
        assert tool._max_results == 50
        assert tool._context_lines == 2
        assert tool._max_file_size == 5 * 1024 * 1024

    def test_init_custom_values(self, tmp_path):
        """Test initialization with custom values."""
        tool = GrepTool(
            raw_dir=tmp_path,
            max_results=20,
            context_lines=3,
            max_file_size=1024 * 1024,
        )
        assert tool._max_results == 20
        assert tool._context_lines == 3
        assert tool._max_file_size == 1024 * 1024

    def test_init_raw_dir_as_string(self, tmp_path):
        """Test that raw_dir accepts string and converts to Path."""
        tool = GrepTool(raw_dir=str(tmp_path))
        assert isinstance(tool._raw_dir, Path)
        assert tool._raw_dir == tmp_path

    def test_init_raw_dir_as_path(self, tmp_path):
        """Test that raw_dir accepts Path directly."""
        tool = GrepTool(raw_dir=tmp_path)
        assert isinstance(tool._raw_dir, Path)


class TestGrepToolProperties:
    """Test GrepTool properties."""

    @pytest.fixture
    def tool(self, tmp_path):
        """Create a GrepTool instance."""
        return GrepTool(raw_dir=tmp_path)

    def test_name_property(self, tool):
        """Test name property returns 'grep'."""
        assert tool.name == "grep"

    def test_description_property(self, tool):
        """Test description property returns non-empty string."""
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0


@pytest.fixture
def grep_dir(tmp_path):
    """Create temp dir with markdown files for testing."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc1.md").write_text("Line 0\n年假有5天\nLine 2\n", encoding="utf-8")
    (raw / "doc2.docx.md").write_text(
        "采购审批流程\n需要经理签字\n", encoding="utf-8"
    )
    (raw / "doc3.md").write_text(
        "第一行\n第二行内容\n第三行匹配\n第四行\n第五行\n", encoding="utf-8"
    )
    # 6MB file — should be skipped by default max_file_size=5MB
    (raw / "big.md").write_text("x" * (6 * 1024 * 1024), encoding="utf-8")
    return raw


class TestGrepToolExecuteBasic:
    """Test basic GrepTool execute functionality."""

    def test_execute_basic_search(self, grep_dir):
        """Test basic pattern search returns matches."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="年假")

        assert result.success is True
        assert "年假" in result.data
        assert result.metadata["total_matches"] >= 1

    def test_execute_regex_pattern(self, grep_dir):
        """Test regex pattern search like '年假.*天'."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="年假.*天")

        assert result.success is True
        assert result.metadata["total_matches"] >= 1
        assert "年假有5天" in result.data

    def test_execute_regex_complex(self, grep_dir):
        """Test complex regex pattern '采购.*审批'."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="采购.*审批")

        assert result.success is True
        assert result.metadata["total_matches"] >= 1

    def test_execute_case_insensitive_default(self, grep_dir):
        """Test case-insensitive search is default."""
        # Write a file with uppercase English
        (grep_dir / "english.md").write_text("Hello World\n", encoding="utf-8")
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="hello")

        assert result.success is True
        assert result.metadata["total_matches"] >= 1

    def test_execute_case_sensitive(self, grep_dir):
        """Test case-sensitive search."""
        (grep_dir / "english.md").write_text("Hello World\n", encoding="utf-8")
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="hello", case_sensitive=True)

        assert result.success is True
        # "hello" should not match "Hello" with case_sensitive=True
        assert result.metadata["total_matches"] == 0

    def test_execute_max_results_truncation(self, grep_dir):
        """Test that results are truncated at max_results."""
        # Create file with many matching lines
        content = "\n".join([f"match line {i}" for i in range(20)])
        (grep_dir / "many.md").write_text(content, encoding="utf-8")

        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="match", max_results=3)

        assert result.success is True
        assert result.metadata["total_matches"] == 3

    def test_execute_file_filter(self, grep_dir):
        """Test filtering by glob pattern."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="采购", file_filter="*.docx.md")

        assert result.success is True
        assert result.metadata["total_matches"] >= 1
        # Result should only be from doc2.docx.md
        assert "doc2.docx.md" in result.data

    def test_execute_file_filter_no_match(self, grep_dir):
        """Test file filter that excludes all files."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="年假", file_filter="*.pdf.md")

        assert result.success is True
        assert result.metadata["total_matches"] == 0

    def test_execute_context_lines(self, grep_dir):
        """Test context_before and context_after lines."""
        tool = GrepTool(raw_dir=grep_dir, context_lines=1)
        result = tool.execute(pattern="第三行匹配")

        assert result.success is True
        assert result.metadata["total_matches"] >= 1
        # Data should contain context line
        assert "第二行内容" in result.data or "第四行" in result.data


class TestGrepToolExecuteErrors:
    """Test GrepTool error handling."""

    def test_execute_empty_pattern(self, tmp_path):
        """Test empty pattern returns error."""
        tool = GrepTool(raw_dir=tmp_path)
        result = tool.execute(pattern="")

        assert result.success is False
        assert result.error is not None
        assert "pattern" in result.error.lower()

    def test_execute_whitespace_pattern(self, tmp_path):
        """Test whitespace-only pattern returns error."""
        tool = GrepTool(raw_dir=tmp_path)
        result = tool.execute(pattern="   ")

        assert result.success is False
        assert result.error is not None
        assert "pattern" in result.error.lower()

    def test_execute_none_pattern(self, tmp_path):
        """Test None pattern returns error."""
        tool = GrepTool(raw_dir=tmp_path)
        result = tool.execute()

        assert result.success is False
        assert result.error is not None

    def test_execute_invalid_regex(self, tmp_path):
        """Test invalid regex pattern returns error."""
        tool = GrepTool(raw_dir=tmp_path)
        result = tool.execute(pattern="[invalid")

        assert result.success is False
        assert result.error is not None
        assert "regex" in result.error.lower() or "pattern" in result.error.lower()

    def test_execute_missing_raw_dir(self, tmp_path):
        """Test error when raw_dir doesn't exist."""
        nonexistent = tmp_path / "nonexistent_dir"
        tool = GrepTool(raw_dir=nonexistent)
        result = tool.execute(pattern="test")

        assert result.success is False
        assert result.error is not None
        assert "does not exist" in result.error


class TestGrepToolExecuteEdgeCases:
    """Test GrepTool edge cases."""

    def test_execute_no_matches(self, grep_dir):
        """Test search with no matches returns success with 0 matches."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="nonexistent_pattern_xyz123")

        assert result.success is True
        assert "No matches" in result.data
        assert result.metadata["total_matches"] == 0

    def test_execute_large_file_skip(self, grep_dir):
        """Test that files exceeding max_file_size are skipped."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="x", max_file_size=5 * 1024 * 1024)

        assert result.success is True
        # big.md is 6MB, should be skipped
        # Only match from grep_dir files that contain 'x' — none do except big.md
        # So should have 0 matches since big.md is skipped
        assert "big.md" not in (result.data or "")

    def test_execute_empty_directory(self, tmp_path):
        """Test search in empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        tool = GrepTool(raw_dir=empty_dir)
        result = tool.execute(pattern="test")

        assert result.success is True
        assert result.metadata["total_matches"] == 0
        assert result.metadata["files_searched"] == 0

    def test_execute_metadata_has_pattern(self, grep_dir):
        """Test that metadata contains the pattern."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="年假")

        assert result.metadata["pattern"] == "年假"

    def test_execute_metadata_has_files_searched(self, grep_dir):
        """Test that metadata contains files_searched count."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="年假")

        assert "files_searched" in result.metadata
        assert result.metadata["files_searched"] >= 1

    def test_execute_metadata_has_execution_time(self, grep_dir):
        """Test that metadata contains execution_time."""
        tool = GrepTool(raw_dir=grep_dir)
        result = tool.execute(pattern="年假")

        assert "execution_time" in result.metadata
        assert result.metadata["execution_time"] >= 0

    def test_execute_subdirectory_files(self, tmp_path):
        """Test that rglob finds files in subdirectories."""
        raw = tmp_path / "raw"
        sub = raw / "subdir"
        sub.mkdir(parents=True)
        (sub / "nested.md").write_text("nested content here\n", encoding="utf-8")

        tool = GrepTool(raw_dir=raw)
        result = tool.execute(pattern="nested")

        assert result.success is True
        assert result.metadata["total_matches"] >= 1
        assert "nested" in result.data

    def test_execute_multiple_matches_in_file(self, tmp_path):
        """Test multiple matches in a single file."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "multi.md").write_text(
            "年假第一行\n其他内容\n年假第二行\n", encoding="utf-8"
        )

        tool = GrepTool(raw_dir=raw)
        result = tool.execute(pattern="年假")

        assert result.success is True
        assert result.metadata["total_matches"] == 2


class TestGrepToolOpenAIFormat:
    """Test GrepTool OpenAI format generation."""

    @pytest.fixture
    def tool(self, tmp_path):
        """Create a GrepTool instance."""
        return GrepTool(raw_dir=tmp_path)

    def test_to_openai_tool_structure(self, tool):
        """Test to_openai_tool returns correct structure."""
        openai_tool = tool.to_openai_tool()

        assert openai_tool["type"] == "function"
        assert "function" in openai_tool
        assert openai_tool["function"]["name"] == "grep"
        assert openai_tool["function"]["description"] == tool.description

    def test_to_openai_tool_parameters(self, tool):
        """Test to_openai_tool has correct parameters schema."""
        openai_tool = tool.to_openai_tool()
        params = openai_tool["function"]["parameters"]

        assert params["type"] == "object"
        assert "properties" in params

        # Check pattern parameter (required)
        assert "pattern" in params["properties"]
        assert params["properties"]["pattern"]["type"] == "string"
        assert "pattern" in params["required"]

        # Check optional parameters
        assert "case_sensitive" in params["properties"]
        assert params["properties"]["case_sensitive"]["type"] == "boolean"
        assert params["properties"]["case_sensitive"]["default"] is False

        assert "max_results" in params["properties"]
        assert params["properties"]["max_results"]["type"] == "integer"

        assert "file_filter" in params["properties"]
        assert params["properties"]["file_filter"]["type"] == "string"

    def test_to_openai_tool_parameter_descriptions(self, tool):
        """Test that parameter descriptions are present."""
        openai_tool = tool.to_openai_tool()
        props = openai_tool["function"]["parameters"]["properties"]

        assert "description" in props["pattern"]
        assert "description" in props["case_sensitive"]
        assert "description" in props["max_results"]
        assert "description" in props["file_filter"]


class TestGrepToolProtocol:
    """Test that GrepTool properly implements Tool protocol."""

    def test_implements_name_property(self, tmp_path):
        """Test that GrepTool has name property."""
        tool = GrepTool(raw_dir=tmp_path)
        assert hasattr(tool, "name")
        assert tool.name == "grep"

    def test_implements_description_property(self, tmp_path):
        """Test that GrepTool has description property."""
        tool = GrepTool(raw_dir=tmp_path)
        assert hasattr(tool, "description")
        assert isinstance(tool.description, str)

    def test_implements_execute_method(self, tmp_path):
        """Test that GrepTool has execute method."""
        tool = GrepTool(raw_dir=tmp_path)
        assert hasattr(tool, "execute")
        assert callable(tool.execute)

    def test_implements_to_openai_tool_method(self, tmp_path):
        """Test that GrepTool has to_openai_tool method."""
        tool = GrepTool(raw_dir=tmp_path)
        assert hasattr(tool, "to_openai_tool")
        assert callable(tool.to_openai_tool)

    def test_execute_returns_tool_result(self, tmp_path):
        """Test that execute returns ToolResult instance."""
        tool = GrepTool(raw_dir=tmp_path)
        result = tool.execute(pattern="test")
        assert isinstance(result, ToolResult)
