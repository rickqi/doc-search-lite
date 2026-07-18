"""Tests for the AnalysisAgent implementation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.agent.analysis_agent import AnalysisAgent, LLMClientAdapter
from src.agent.base import Agent, AgentResponse
from src.agent.llm_client import ChatMessage, ChatResponse, LLMClient
from src.storage.markdown_store import MarkdownStore
from src.storage.base import DocumentRecord


# Mock LLM Client for testing
class MockLLMClient:
    """Mock LLM client for testing purposes."""

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
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
        self.last_messages: Optional[List[ChatMessage]] = None

    def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ChatResponse:
        """Generate mock response."""
        self.call_count += 1
        self.last_messages = messages

        # Get the last user message
        user_content = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_content = msg.content
                break

        # Check for matching response
        for key, response in self.responses.items():
            if key in user_content:
                return ChatResponse(content=response, usage={"total_tokens": 100})

        return ChatResponse(
            content=self.default_response,
            usage={"total_tokens": 50},
        )


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
def analysis_agent(mock_llm_client, mock_markdown_store):
    """Create an AnalysisAgent with mocked dependencies."""
    return AnalysisAgent(
        llm_client=mock_llm_client,
        markdown_store=mock_markdown_store,
    )


class TestLLMClientAdapter:
    """Tests for LLMClientAdapter."""

    def test_adapter_generates_response(self, mock_llm_client):
        """Test that adapter generates response correctly."""
        adapter = LLMClientAdapter(mock_llm_client)

        response = adapter.generate(
            prompt="Test prompt",
            system_prompt="System instruction",
        )

        assert response == "Mock analysis response"
        assert mock_llm_client.call_count == 1
        assert len(mock_llm_client.last_messages) == 2
        assert mock_llm_client.last_messages[0].role == "system"
        assert mock_llm_client.last_messages[1].role == "user"

    def test_adapter_without_system_prompt(self, mock_llm_client):
        """Test adapter without system prompt."""
        adapter = LLMClientAdapter(mock_llm_client)

        response = adapter.generate(prompt="Test prompt")

        assert response == "Mock analysis response"
        assert len(mock_llm_client.last_messages) == 1
        assert mock_llm_client.last_messages[0].role == "user"

    def test_adapter_passes_temperature_and_max_tokens(self, mock_llm_client):
        """Test that adapter passes temperature and max_tokens."""
        adapter = LLMClientAdapter(mock_llm_client)

        adapter.generate(
            prompt="Test",
            temperature=0.5,
            max_tokens=500,
        )

        # The mock doesn't validate these, but we verify the call was made
        assert mock_llm_client.call_count == 1

    def test_count_tokens(self, mock_llm_client):
        """Test token counting approximation."""
        adapter = LLMClientAdapter(mock_llm_client)

        # Test with different text lengths
        assert adapter.count_tokens("") == 1
        assert adapter.count_tokens("abc") == 1  # 3 chars / 4 = 0.75 -> 1
        assert adapter.count_tokens("abcdefgh") == 2  # 8 chars / 4 = 2
        assert adapter.count_tokens("a" * 100) == 25  # 100 chars / 4 = 25


class TestAnalysisAgentProperties:
    """Test basic agent properties."""

    def test_name(self, analysis_agent):
        """Test agent name."""
        assert analysis_agent.name == "analysis_agent"

    def test_is_agent_subclass(self, analysis_agent):
        """Test that AnalysisAgent is an Agent subclass."""
        assert isinstance(analysis_agent, Agent)

    def test_has_analyze_tool_registered(self, analysis_agent):
        """Test that analyze tool is registered."""
        tool = analysis_agent.get_tool("analyze")
        assert tool is not None
        assert tool.name == "analyze"


class TestAnalysisAgentCompare:
    """Test compare mode functionality."""

    def test_compare_two_documents(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
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
        response = analysis_agent.compare(["doc1", "doc2"])

        # Verify
        assert response.success is True
        assert response.answer == "Comparison analysis result"
        assert response.sources == ["doc1", "doc2"]
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["tool"] == "analyze"

    def test_compare_with_aspect(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test comparing with specific aspect."""
        # Setup mock
        mock_markdown_store.load.return_value = (
            sample_document_record,
            "Document content",
        )
        mock_llm_client.default_response = "Analysis focused on pricing"

        # Execute
        response = analysis_agent.compare(
            ["doc1", "doc2"], aspect="pricing differences"
        )

        # Verify
        assert response.success is True
        # Check that aspect was passed to the tool
        tool_call = response.tool_calls[0]
        assert tool_call["arguments"]["query"] == "pricing differences"

    def test_compare_requires_two_documents(self, analysis_agent):
        """Test that compare mode requires at least 2 documents."""
        # Execute with only one document
        response = analysis_agent.compare(["doc1"])

        # Verify
        assert response.success is False
        assert "at least 2" in response.error.lower()

    def test_compare_requires_doc_ids(self, analysis_agent):
        """Test that compare mode requires doc_ids."""
        # Execute with empty list
        response = analysis_agent.compare([])

        # Verify
        assert response.success is False

    def test_compare_document_not_found(self, analysis_agent, mock_markdown_store):
        """Test handling of missing document."""
        # Setup mock
        mock_markdown_store.load.return_value = None

        # Execute
        response = analysis_agent.compare(["missing1", "missing2"])

        # Verify
        assert response.success is False
        assert "not found" in response.error.lower()

    def test_compare_includes_sources(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test that compare includes source references."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Comparison result"

        response = analysis_agent.compare(["doc1", "doc2", "doc3"])

        assert response.success is True
        assert "doc1" in response.sources
        assert "doc2" in response.sources
        assert "doc3" in response.sources


class TestAnalysisAgentExtract:
    """Test extract mode functionality."""

    def test_extract_basic(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test basic information extraction."""
        # Setup mock
        doc_content = "Name: John Doe\nAge: 30\nEmail: john@example.com"
        mock_markdown_store.load.return_value = (sample_document_record, doc_content)
        mock_llm_client.default_response = '{"name": "John Doe", "age": 30}'

        # Execute
        response = analysis_agent.extract(
            doc_id="doc123",
            query="Extract contact information",
        )

        # Verify
        assert response.success is True
        assert response.sources == ["doc123"]
        assert len(response.tool_calls) == 1

    def test_extract_with_schema(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
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
        response = analysis_agent.extract(
            doc_id="doc123",
            query="Extract title and summary",
            schema=schema,
        )

        # Verify
        assert response.success is True
        assert response.sources == ["doc123"]

    def test_extract_missing_doc_id(self, analysis_agent):
        """Test extraction without document ID."""
        # Execute
        response = analysis_agent.extract(doc_id=None, query="Extract something")

        # Verify
        assert response.success is False
        assert "doc" in response.error.lower() or "id" in response.error.lower()

    def test_extract_document_not_found(self, analysis_agent, mock_markdown_store):
        """Test extraction with non-existent document."""
        # Setup mock
        mock_markdown_store.load.return_value = None

        # Execute
        response = analysis_agent.extract(doc_id="nonexistent")

        # Verify
        assert response.success is False
        assert "not found" in response.error.lower()

    def test_extract_includes_sources(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test that extract includes source references."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Extracted info"

        response = analysis_agent.extract(doc_id="my-doc")

        assert response.success is True
        assert "my-doc" in response.sources


class TestAnalysisAgentRun:
    """Test the main run() method."""

    def test_run_auto_detect_compare_mode(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test run auto-detects compare mode from context."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Comparison result"

        response = analysis_agent.run(
            "Compare these documents",
            context={"doc_ids": ["doc1", "doc2"]},
        )

        assert response.success is True
        assert response.sources == ["doc1", "doc2"]

    def test_run_auto_detect_extract_mode(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test run auto-detects extract mode from context."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Extraction result"

        response = analysis_agent.run(
            "Extract information",
            context={"doc_id": "doc123"},
        )

        assert response.success is True
        assert response.sources == ["doc123"]

    def test_run_explicit_mode_compare(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test run with explicit compare mode."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Comparison result"

        response = analysis_agent.run(
            "Compare",
            context={"mode": "compare", "doc_ids": ["doc1", "doc2"]},
        )

        assert response.success is True

    def test_run_explicit_mode_extract(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test run with explicit extract mode."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Extraction result"

        response = analysis_agent.run(
            "Extract",
            context={"mode": "extract", "doc_id": "doc123"},
        )

        assert response.success is True

    def test_run_with_schema(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test run with schema for extraction."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = '{"key": "value"}'

        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        response = analysis_agent.run(
            "Extract data",
            context={"doc_id": "doc123", "schema": schema},
        )

        assert response.success is True

    def test_run_handles_exception(self, analysis_agent, mock_markdown_store):
        """Test run handles exceptions gracefully."""
        mock_markdown_store.load.side_effect = Exception("Database error")

        response = analysis_agent.run(
            "Analyze",
            context={"doc_id": "doc123"},
        )

        assert response.success is False
        assert "Database error" in response.error


class TestAnalysisAgentConvenienceMethods:
    """Test convenience methods."""

    def test_analyze_table(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test table extraction convenience method."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = '{"tables": []}'

        response = analysis_agent.analyze_table("doc123")

        assert response.success is True
        assert response.sources == ["doc123"]

    def test_summarize(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test summarization convenience method."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "This is a summary."

        response = analysis_agent.summarize("doc123")

        assert response.success is True
        assert response.sources == ["doc123"]

    def test_summarize_with_focus(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test summarization with focus."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Summary focusing on key points."

        response = analysis_agent.summarize("doc123", focus="key points")

        assert response.success is True


class TestAnalysisAgentResponse:
    """Test response structure."""

    def test_response_has_processing_time(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test response includes processing time."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Result"

        response = analysis_agent.extract(doc_id="doc123")

        assert response.processing_time >= 0

    def test_response_has_tokens_used(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test response includes tokens used."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Result"

        response = analysis_agent.extract(doc_id="doc123")

        assert response.tokens_used >= 0

    def test_response_has_reasoning(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test response includes reasoning."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Result"

        response = analysis_agent.extract(doc_id="doc123", query="Extract info")

        assert response.reasoning != ""
        assert (
            "Extract" in response.reasoning or "extract" in response.reasoning.lower()
        )

    def test_response_tool_calls_structure(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test tool calls have correct structure."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.default_response = "Result"

        response = analysis_agent.compare(["doc1", "doc2"])

        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]
        assert "tool" in tool_call
        assert "arguments" in tool_call
        assert "result" in tool_call


class TestAnalysisAgentConfidence:
    """Test confidence scoring."""

    def test_confidence_calculation(self, analysis_agent):
        """Test confidence is calculated."""
        # Low token usage
        confidence_low = analysis_agent._calculate_confidence(100, 1)
        assert 0.0 <= confidence_low <= 1.0

        # High token usage
        confidence_high = analysis_agent._calculate_confidence(1500, 1)
        assert confidence_high > confidence_low

    def test_confidence_with_multiple_docs(self, analysis_agent):
        """Test confidence increases with more documents."""
        confidence_single = analysis_agent._calculate_confidence(500, 1)
        confidence_multiple = analysis_agent._calculate_confidence(500, 3)

        assert confidence_multiple >= confidence_single

    def test_confidence_bounded(self, analysis_agent):
        """Test confidence is bounded between 0 and 1."""
        # Very high values
        confidence = analysis_agent._calculate_confidence(10000, 10)
        assert confidence <= 1.0


class TestAnalysisAgentErrorHandling:
    """Test error handling."""

    def test_compare_handles_tool_error(self, analysis_agent, mock_markdown_store):
        """Test compare handles tool execution errors."""
        mock_markdown_store.load.return_value = None

        response = analysis_agent.compare(["missing1", "missing2"])

        assert response.success is False
        assert response.error is not None

    def test_extract_handles_tool_error(self, analysis_agent, mock_markdown_store):
        """Test extract handles tool execution errors."""
        mock_markdown_store.load.return_value = None

        response = analysis_agent.extract(doc_id="missing")

        assert response.success is False
        assert response.error is not None

    def test_llm_exception_handled(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test LLM exceptions are handled gracefully."""
        mock_markdown_store.load.return_value = (sample_document_record, "Content")
        mock_llm_client.chat = MagicMock(side_effect=Exception("LLM API error"))

        response = analysis_agent.extract(doc_id="doc123")

        assert response.success is False
        assert "LLM API error" in response.error


class TestAnalysisAgentIntegration:
    """Integration tests for AnalysisAgent."""

    def test_full_compare_workflow(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test complete compare workflow."""
        # Setup
        doc1 = DocumentRecord(
            id="report-q1",
            source_path=Path("reports/q1.pdf"),
            output_path=Path("output/q1.md"),
            title="Q1 Report",
            content_hash="hash1",
            file_size=2000,
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
        doc2 = DocumentRecord(
            id="report-q2",
            source_path=Path("reports/q2.pdf"),
            output_path=Path("output/q2.md"),
            title="Q2 Report",
            content_hash="hash2",
            file_size=2500,
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

        def mock_load(doc_id):
            if doc_id == "report-q1":
                return (doc1, "Q1 revenue: $1M, expenses: $800K")
            elif doc_id == "report-q2":
                return (doc2, "Q2 revenue: $1.2M, expenses: $900K")
            return None

        mock_markdown_store.load.side_effect = mock_load
        mock_llm_client.default_response = "Q2 shows 20% revenue growth compared to Q1"

        # Execute
        response = analysis_agent.run(
            "Compare quarterly reports",
            context={
                "doc_ids": ["report-q1", "report-q2"],
                "aspect": "financial performance",
            },
        )

        # Verify
        assert response.success is True
        assert "report-q1" in response.sources
        assert "report-q2" in response.sources
        assert response.processing_time >= 0
        assert len(response.tool_calls) == 1

    def test_full_extract_workflow(
        self,
        analysis_agent,
        mock_markdown_store,
        sample_document_record,
        mock_llm_client,
    ):
        """Test complete extract workflow."""
        # Setup
        mock_markdown_store.load.return_value = (
            sample_document_record,
            "Product: Widget Pro\nPrice: $99.99\nStock: 500 units",
        )
        mock_llm_client.default_response = json.dumps(
            {"product": "Widget Pro", "price": "$99.99", "stock": 500}
        )

        schema = {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "price": {"type": "string"},
                "stock": {"type": "integer"},
            },
        }

        # Execute
        response = analysis_agent.run(
            "Extract product information",
            context={
                "doc_id": "doc123",
                "schema": schema,
            },
        )

        # Verify
        assert response.success is True
        assert response.sources == ["doc123"]
        assert (
            "product" in response.answer.lower() or "widget" in response.answer.lower()
        )
