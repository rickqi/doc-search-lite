"""Tests for the AnalyzeTool implementation."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.base import Tool
from src.agent.tools.analyze import AnalyzeTool, LLMClientProtocol
from src.storage.base import DocumentRecord
from src.storage.markdown_store import MarkdownStore


# Mock LLM Client for testing
class MockLLMClient:
    """Mock LLM client for testing purposes."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "Mock analysis response",
    ):
        """Initialize mock client.

        Args:
            responses: Dictionary mapping prompts (or substrings) to responses
            default_response: Default response if no match found
        """
        self.responses = responses or {}
        self.default_response = default_response
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate mock response."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        # Check for matching response
        for key, response in self.responses.items():
            if key in prompt:
                return response

        return self.default_response

    def count_tokens(self, text: str) -> int:
        """Mock token count (rough estimate)."""
        return len(text.split())


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_markdown_store():
    """Create a mock MarkdownStore."""
    store = MagicMock(spec=MarkdownStore)
    return store


@pytest.fixture
def sample_document_record():
    """Create a sample document record."""
    return DocumentRecord(
        id="doc123",
        source_path=Path("test/sample.pdf"),
        output_path=Path("output/test/sample.md"),
        title="Sample Document",
        content_hash="abc123",
        file_size=1000,
        file_mtime=datetime.now(),
        metadata={},
        keywords=[],
        sections=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        convert_count=1,
        last_convert_time=1.0,
        last_converter="pdf",
        status="active",
    )


@pytest.fixture
def analyze_tool(mock_markdown_store, mock_llm_client):
    """Create an AnalyzeTool with mocked dependencies."""
    return AnalyzeTool(
        markdown_store=mock_markdown_store,
        llm_client=mock_llm_client,
    )


class TestAnalyzeToolProperties:
    """Test basic tool properties."""

    def test_name(self, analyze_tool):
        """Test tool name."""
        assert analyze_tool.name == "analyze"

    def test_description(self, analyze_tool):
        """Test tool description."""
        assert "分析" in analyze_tool.description or "对比" in analyze_tool.description

    def test_is_tool_subclass(self, analyze_tool):
        """Test that AnalyzeTool is a Tool subclass."""
        assert isinstance(analyze_tool, Tool)

    def test_to_openai_tool(self, analyze_tool):
        """Test OpenAI tool format."""
        openai_tool = analyze_tool.to_openai_tool()

        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "analyze"
        assert "parameters" in openai_tool["function"]

        params = openai_tool["function"]["parameters"]["properties"]
        assert "mode" in params
        assert "doc_ids" in params
        assert "query" in params


class TestAnalyzeToolCompareMode:
    """Test compare mode functionality."""

    def test_compare_two_documents(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test comparing two documents."""
        # Setup mock
        doc1_content = "This is document 1 content about AI."
        doc2_content = "This is document 2 content about ML."

        def mock_load(doc_id):
            if doc_id == "doc1":
                return (sample_document_record, doc1_content)
            elif doc_id == "doc2":
                record2 = DocumentRecord(
                    id="doc2",
                    source_path=Path("test/sample2.pdf"),
                    output_path=Path("output/test/sample2.md"),
                    title="Document 2",
                    content_hash="def456",
                    file_size=1200,
                    file_mtime=datetime.now(),
                    metadata={},
                    keywords=[],
                    sections=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    convert_count=1,
                    last_convert_time=1.0,
                    last_converter="pdf",
                    status="active",
                )
                return (record2, doc2_content)
            return None

        mock_markdown_store.load.side_effect = mock_load
        mock_llm_client.default_response = "Comparison analysis result"

        # Execute
        result = analyze_tool.execute(
            mode="compare",
            doc_ids=["doc1", "doc2"],
        )

        # Verify
        assert result.success is True
        assert result.data == "Comparison analysis result"
        assert result.metadata["mode"] == "compare"
        assert result.metadata["doc_ids"] == ["doc1", "doc2"]
        assert "tokens_used" in result.metadata

    def test_compare_with_query(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test comparing with a specific query."""
        # Setup mock
        mock_markdown_store.load.return_value = (
            sample_document_record,
            "Document content",
        )
        mock_llm_client.default_response = "Analysis with query"

        # Execute
        result = analyze_tool.execute(
            mode="compare",
            doc_ids=["doc1", "doc2"],
            query="What are the key differences?",
        )

        # Verify
        assert result.success is True
        assert "What are the key differences?" in mock_llm_client.last_prompt

    def test_compare_requires_two_documents(self, analyze_tool, mock_llm_client):
        """Test that compare mode requires at least 2 documents."""
        # Execute with only one document
        result = analyze_tool.execute(
            mode="compare",
            doc_ids=["doc1"],
        )

        # Verify
        assert result.success is False
        assert (
            "at least 2" in result.error.lower()
            or "insufficient" in result.error.lower()
        )

    def test_compare_requires_doc_ids(self, analyze_tool, mock_llm_client):
        """Test that compare mode requires doc_ids."""
        # Execute without doc_ids
        result = analyze_tool.execute(
            mode="compare",
            query="Compare something",
        )

        # Verify
        assert result.success is False

    def test_compare_document_not_found(
        self, analyze_tool, mock_markdown_store, mock_llm_client
    ):
        """Test handling of missing document."""
        # Setup mock
        mock_markdown_store.load.return_value = None

        # Execute
        result = analyze_tool.execute(
            mode="compare",
            doc_ids=["missing1", "missing2"],
        )

        # Verify
        assert result.success is False
        assert "not found" in result.error.lower()


class TestAnalyzeToolExtractMode:
    """Test extract mode functionality."""

    def test_extract_basic(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test basic information extraction."""
        # Setup mock
        doc_content = "Name: John Doe\nAge: 30\nEmail: john@example.com"
        mock_markdown_store.load.return_value = (sample_document_record, doc_content)
        mock_llm_client.default_response = '{"name": "John Doe", "age": 30}'

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="doc123",
            query="Extract contact information",
        )

        # Verify
        assert result.success is True
        assert result.metadata["mode"] == "extract"
        assert result.metadata["doc_id"] == "doc123"

    def test_extract_with_schema(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test extraction with JSON schema."""
        # Setup mock
        mock_markdown_store.load.return_value = (
            sample_document_record,
            "Document content",
        )

        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
            },
        }

        json_response = '{"title": "Test", "summary": "A summary"}'
        mock_llm_client.default_response = json_response

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="doc123",
            query="Extract title and summary",
            schema=schema,
        )

        # Verify
        assert result.success is True
        assert result.metadata["has_schema"] is True
        # Should parse JSON
        assert isinstance(result.data, dict) or result.data == json_response

    def test_extract_from_doc_ids_single(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test extraction using doc_ids with single element."""
        # Setup mock
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Extracted info"

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_ids=["doc123"],
        )

        # Verify
        assert result.success is True
        mock_markdown_store.load.assert_called_with("doc123")

    def test_extract_missing_doc_id(self, analyze_tool, mock_llm_client):
        """Test extraction without document ID."""
        # Execute
        result = analyze_tool.execute(
            mode="extract",
            query="Extract something",
        )

        # Verify
        assert result.success is False
        assert "doc_id" in result.error.lower() or "document" in result.error.lower()

    def test_extract_document_not_found(
        self, analyze_tool, mock_markdown_store, mock_llm_client
    ):
        """Test extraction with non-existent document."""
        # Setup mock
        mock_markdown_store.load.return_value = None

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="nonexistent",
        )

        # Verify
        assert result.success is False
        assert "not found" in result.error.lower()


class TestAnalyzeToolLLMClient:
    """Test LLM client handling."""

    def test_no_llm_client(self, mock_markdown_store):
        """Test error when LLM client not set."""
        tool = AnalyzeTool(markdown_store=mock_markdown_store)

        result = tool.execute(mode="extract", doc_id="test")

        assert result.success is False
        assert "llm" in result.error.lower() or "not configured" in result.error.lower()

    def test_set_llm_client(self, mock_markdown_store, mock_llm_client):
        """Test setting LLM client after initialization."""
        tool = AnalyzeTool(markdown_store=mock_markdown_store)

        # First call should fail
        result1 = tool.execute(mode="extract", doc_id="test")
        assert result1.success is False

        # Set client
        tool.set_llm_client(mock_llm_client)

        # Now should work (if document exists)
        mock_markdown_store.load.return_value = (
            DocumentRecord(
                id="test",
                source_path=Path("test.pdf"),
                output_path=Path("test.md"),
                title="Test",
                content_hash="hash",
                file_size=100,
                file_mtime=datetime.now(),
                metadata={},
                keywords=[],
                sections=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            "Content",
        )
        result2 = tool.execute(mode="extract", doc_id="test")
        assert result2.success is True

    def test_llm_error_handling(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test handling of LLM errors."""
        # Setup mock to raise exception
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.generate = MagicMock(side_effect=Exception("LLM error"))

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="doc123",
        )

        # Verify
        assert result.success is False
        assert "llm" in result.error.lower() or "failed" in result.error.lower()


class TestAnalyzeToolInvalidMode:
    """Test invalid mode handling."""

    def test_invalid_mode(self, analyze_tool):
        """Test error for invalid mode."""
        result = analyze_tool.execute(mode="invalid_mode")

        assert result.success is False
        assert "invalid" in result.error.lower() or "mode" in result.error.lower()


class TestAnalyzeToolJSONExtraction:
    """Test JSON extraction from LLM responses."""

    def test_extract_json_from_code_block(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test extracting JSON from markdown code blocks."""
        # Setup mock
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = """Here's the extracted data:
```json
{"name": "John", "age": 30}
```
"""

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="doc123",
            schema={"type": "object"},
        )

        # Verify
        assert result.success is True
        assert result.data == {"name": "John", "age": 30}

    def test_extract_json_direct(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test extracting JSON when directly returned."""
        # Setup mock
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = '{"title": "Test", "items": [1, 2, 3]}'

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="doc123",
            schema={"type": "object"},
        )

        # Verify
        assert result.success is True
        assert result.data == {"title": "Test", "items": [1, 2, 3]}

    def test_extract_json_from_array(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test extracting JSON array."""
        # Setup mock
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = '[{"id": 1}, {"id": 2}]'

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="doc123",
            schema={"type": "array"},
        )

        # Verify
        assert result.success is True
        assert result.data == [{"id": 1}, {"id": 2}]

    def test_invalid_json_fallback(
        self, analyze_tool, mock_markdown_store, sample_document_record, mock_llm_client
    ):
        """Test fallback when JSON parsing fails."""
        # Setup mock
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "This is plain text, not JSON"

        # Execute
        result = analyze_tool.execute(
            mode="extract",
            doc_id="doc123",
            schema={"type": "object"},
        )

        # Verify - should succeed with raw text
        assert result.success is True
        assert result.data == "This is plain text, not JSON"


class TestAnalyzeToolProtocol:
    """Test LLMClientProtocol compliance."""

    def test_protocol_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        client = MockLLMClient()
        assert isinstance(client, LLMClientProtocol)

    def test_tool_accepts_protocol_compliant_client(self, mock_markdown_store):
        """Test that tool accepts any protocol-compliant client."""

        # Custom client that follows protocol
        class CustomClient:
            def generate(
                self, prompt, system_prompt=None, temperature=0.7, max_tokens=2000
            ):
                return "response"

            def count_tokens(self, text):
                return 10

        tool = AnalyzeTool(
            markdown_store=mock_markdown_store,
            llm_client=CustomClient(),
        )

        assert tool._llm is not None
