"""Unit tests for ReadTool."""

from datetime import datetime
from pathlib import Path

import pytest

from src.agent.tools.read import ReadTool
from src.storage.base import DocumentRecord
from src.storage.markdown_store import MarkdownStore


class MockMarkdownStore(MarkdownStore):
    """Mock MarkdownStore for testing."""

    def __init__(self) -> None:
        """Initialize mock store without paths."""
        self._test_content = {}
        self._test_content_by_path = {}

    def add_test_doc(
        self,
        doc_id: str,
        source_path: Path,
        title: str,
        content: str,
    ) -> None:
        """Add a test document to the mock store.

        Args:
            doc_id: Document ID
            source_path: Source file path
            title: Document title
            content: Markdown content
        """
        record = DocumentRecord(
            id=doc_id,
            source_path=source_path,
            output_path=Path("test_output.md"),
            title=title,
            content_hash="test_hash",
            file_size=len(content.encode("utf-8")),
            file_mtime=datetime.now(),
        )
        self._test_content[doc_id] = (record, content)
        self._test_content_by_path[str(source_path)] = (record, content)

    def load(self, doc_id: str) -> tuple[DocumentRecord, str] | None:
        """Load document by ID."""
        return self._test_content.get(doc_id)

    def load_by_source(self, source_path: Path) -> tuple[DocumentRecord, str] | None:
        """Load document by source path."""
        return self._test_content_by_path.get(str(source_path))


@pytest.fixture
def mock_store():
    """Create a mock MarkdownStore with test data."""
    store = MockMarkdownStore()

    # Add test documents
    store.add_test_doc(
        doc_id="test_doc_001",
        source_path=Path("docs/test1.md"),
        title="Test Document 1",
        content="""# Test Document 1

This is a test document.

## Section 1

Line 1
Line 2
Line 3

## Section 2

Line 4
Line 5
Line 6

## Section 3

Line 7
Line 8
Line 9
Line 10
""",
    )

    store.add_test_doc(
        doc_id="test_doc_002",
        source_path=Path("docs/test2.md"),
        title="Large Document",
        content="\n".join([f"Line {i}" for i in range(1, 1001)]),
    )

    store.add_test_doc(
        doc_id="test_doc_003",
        source_path=Path("docs/empty.md"),
        title="Empty Document",
        content="",
    )

    return store


@pytest.fixture
def read_tool(mock_store):
    """Create a ReadTool instance with mock store."""
    return ReadTool(mock_store)


class TestReadToolProperties:
    """Test ReadTool properties."""

    def test_name(self, read_tool):
        """Test tool name."""
        assert read_tool.name == "read"

    def test_description(self, read_tool):
        """Test tool description."""
        assert "读取" in read_tool.description
        assert "Markdown" in read_tool.description or "文档" in read_tool.description


class TestReadToolExecuteByDocId:
    """Test reading documents by doc_id."""

    def test_read_by_doc_id_success(self, read_tool):
        """Test successful read by doc_id."""
        result = read_tool.execute(doc_id="test_doc_001")

        assert result.success is True
        assert result.data is not None
        assert "Test Document 1" in result.data
        assert result.metadata["doc_id"] == "test_doc_001"
        assert result.metadata["source_path"] == str(Path("docs/test1.md"))
        assert result.metadata["title"] == "Test Document 1"
        assert result.metadata["total_lines"] > 0
        assert result.metadata["file_size"] > 0
        assert "truncated" in result.metadata

    def test_read_by_doc_id_pagination(self, read_tool):
        """Test pagination with start_line and max_lines."""
        # Read first 5 lines
        result = read_tool.execute(doc_id="test_doc_002", start_line=0, max_lines=5)
        assert result.success is True
        assert result.metadata["start_line"] == 0
        assert result.metadata["lines_read"] == 5
        assert result.metadata["truncated"] is True

        # Read lines 10-20
        result = read_tool.execute(doc_id="test_doc_002", start_line=10, max_lines=10)
        assert result.success is True
        assert result.metadata["start_line"] == 10
        assert result.metadata["lines_read"] == 10
        assert "Line 11" in result.data
        assert "Line 20" in result.data

    def test_read_by_doc_id_end_of_file(self, read_tool):
        """Test reading beyond end of file."""
        # Read beyond end
        result = read_tool.execute(doc_id="test_doc_001", start_line=1000, max_lines=10)
        assert result.success is True
        assert result.data == ""
        assert result.metadata["start_line"] == 1000
        assert result.metadata["lines_read"] == 0
        assert result.metadata["truncated"] is False

    def test_read_by_doc_id_negative_start_line(self, read_tool):
        """Test negative start_line is handled correctly."""
        result = read_tool.execute(doc_id="test_doc_001", start_line=-5, max_lines=10)
        assert result.success is True
        assert result.metadata["start_line"] == 0  # Should be clamped to 0

    def test_read_empty_document(self, read_tool):
        """Test reading empty document."""
        result = read_tool.execute(doc_id="test_doc_003")
        assert result.success is True
        assert result.data == ""
        # Empty string split by \n gives [""], which is 1 line
        assert result.metadata["total_lines"] == 1
        assert result.metadata["truncated"] is False


class TestReadToolExecuteBySourcePath:
    """Test reading documents by source_path."""

    def test_read_by_source_path_success(self, read_tool):
        """Test successful read by source_path."""
        result = read_tool.execute(source_path="docs/test1.md")

        assert result.success is True
        assert result.data is not None
        assert "Test Document 1" in result.data
        assert result.metadata["doc_id"] == "test_doc_001"
        assert result.metadata["source_path"] == str(Path("docs/test1.md"))

    def test_read_by_source_path_pagination(self, read_tool):
        """Test pagination with source_path."""
        result = read_tool.execute(
            source_path="docs/test2.md", start_line=5, max_lines=3
        )

        assert result.success is True
        assert result.metadata["start_line"] == 5
        assert result.metadata["lines_read"] == 3
        assert "Line 6" in result.data

    def test_read_by_source_path_missing(self, read_tool):
        """Test reading missing document by source_path."""
        result = read_tool.execute(source_path="docs/nonexistent.md")

        assert result.success is False
        assert "not found" in result.error.lower()
        # Path may be normalized to forward slashes
        assert "docs" in result.metadata["source_path"]
        assert "nonexistent.md" in result.metadata["source_path"]


class TestReadToolExecuteMissing:
    """Test error handling for missing documents."""

    def test_read_by_doc_id_missing(self, read_tool):
        """Test reading non-existent doc_id."""
        result = read_tool.execute(doc_id="nonexistent_doc")

        assert result.success is False
        assert "not found" in result.error.lower()
        assert result.metadata["doc_id"] == "nonexistent_doc"

    def test_read_missing_parameters(self, read_tool):
        """Test reading without any parameters."""
        result = read_tool.execute()

        assert result.success is False
        assert "doc_id or source_path" in result.error.lower()
        assert result.metadata.get("error_type") == "missing_parameter"


class TestReadToolToOpenAITool:
    """Test OpenAI function calling format conversion."""

    def test_to_openai_tool_structure(self, read_tool):
        """Test OpenAI tool structure."""
        openai_tool = read_tool.to_openai_tool()

        assert openai_tool["type"] == "function"
        assert "function" in openai_tool
        assert openai_tool["function"]["name"] == "read"
        assert "description" in openai_tool["function"]

    def test_to_openai_tool_parameters(self, read_tool):
        """Test OpenAI tool parameters."""
        openai_tool = read_tool.to_openai_tool()
        params = openai_tool["function"]["parameters"]

        assert params["type"] == "object"
        assert "properties" in params
        assert "doc_id" in params["properties"]
        assert "source_path" in params["properties"]
        assert "start_line" in params["properties"]
        assert "max_lines" in params["properties"]

        # Check parameter types
        assert params["properties"]["doc_id"]["type"] == "string"
        assert params["properties"]["source_path"]["type"] == "string"
        assert params["properties"]["start_line"]["type"] == "integer"
        assert params["properties"]["max_lines"]["type"] == "integer"


class TestReadToolTruncatedFlag:
    """Test truncated flag logic."""

    def test_truncated_true(self, read_tool):
        """Test truncated flag when content is truncated."""
        # Large doc, reading only part
        result = read_tool.execute(doc_id="test_doc_002", max_lines=100)
        assert result.success is True
        assert result.metadata["truncated"] is True

    def test_truncated_false(self, read_tool):
        """Test truncated flag when all content fits."""
        # Small doc, reading all
        result = read_tool.execute(doc_id="test_doc_001", max_lines=1000)
        assert result.success is True
        assert result.metadata["truncated"] is False

    def test_truncated_empty_doc(self, read_tool):
        """Test truncated flag for empty document."""
        result = read_tool.execute(doc_id="test_doc_003")
        assert result.success is True
        assert result.metadata["truncated"] is False
