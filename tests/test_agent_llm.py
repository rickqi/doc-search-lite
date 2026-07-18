"""Tests for LLMClient with mocked LiteLLM responses.

This module tests the LLMClient class including:
- Chat completions
- Streaming completions
- Tool calling
- Error handling and retry logic
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.base import Tool
from src.agent.llm_client import (
    AuthenticationError,
    BudgetExceededError,
    ChatMessage,
    ChatResponse,
    LLMClient,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ToolCall,
    ToolResult,
)
from src.stats.budget_guard import BudgetAlert, BudgetCheckResult
from src.utils.config import Config


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> Config:
    """Create a mock config for testing."""
    return Config(
        glm_api_key="test-api-key-12345",
        glm_base_url="https://api.test.com/v1",
        llm_model="glm-4",
        llm_temperature=0.7,
        llm_max_tokens=1000,
    )


@pytest.fixture
def llm_client(mock_config: Config) -> LLMClient:
    """Create an LLMClient instance for testing."""
    return LLMClient(
        config=mock_config,
        max_retries=2,
        retry_delay=0.1,
        timeout=30.0,
    )


@pytest.fixture
def mock_tool() -> Tool:
    """Create a mock tool for testing."""

    class MockTool(Tool):
        @property
        def name(self) -> str:
            return "get_weather"

        @property
        def description(self) -> str:
            return "Get the current weather"

        def execute(self, **kwargs: Any) -> Any:
            location = kwargs.get("location", "unknown")
            return f"Weather in {location}: sunny, 25°C"

        def to_openai_tool(self) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }

    return MockTool()


def create_mock_response(
    content: str = "Hello, I'm an AI assistant.",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Create a mock LiteLLM response.

    Args:
        content: Response content text
        tool_calls: Optional list of tool calls
        finish_reason: Completion finish reason
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        MagicMock: Mocked response object
    """
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "zai/glm-4"

    return mock_response


def create_mock_tool_call(
    call_id: str = "call_123",
    name: str = "get_weather",
    arguments: str = '{"location": "Beijing"}',
) -> MagicMock:
    """Create a mock tool call object.

    Args:
        call_id: Tool call ID
        name: Function name
        arguments: Arguments as JSON string

    Returns:
        MagicMock: Mocked tool call object
    """
    mock_function = MagicMock()
    mock_function.name = name
    mock_function.arguments = arguments

    mock_tool_call = MagicMock()
    mock_tool_call.id = call_id
    mock_tool_call.function = mock_function

    return mock_tool_call


def create_mock_stream_chunk(content: str = "") -> MagicMock:
    """Create a mock streaming chunk.

    Args:
        content: Chunk content

    Returns:
        MagicMock: Mocked streaming chunk
    """
    mock_delta = MagicMock()
    mock_delta.content = content

    mock_choice = MagicMock()
    mock_choice.delta = mock_delta

    mock_chunk = MagicMock()
    mock_chunk.choices = [mock_choice]

    return mock_chunk


# =============================================================================
# Test ChatMessage
# =============================================================================


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_to_dict_basic(self) -> None:
        """Test basic to_dict conversion."""
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "Hello"
        assert "name" not in result
        assert "tool_call_id" not in result

    def test_to_dict_with_tool_info(self) -> None:
        """Test to_dict with tool call information."""
        msg = ChatMessage(
            role="tool",
            content="Result",
            name="get_weather",
            tool_call_id="call_123",
        )
        result = msg.to_dict()

        assert result["role"] == "tool"
        assert result["content"] == "Result"
        assert result["name"] == "get_weather"
        assert result["tool_call_id"] == "call_123"

    def test_to_dict_with_tool_calls(self) -> None:
        """Test to_dict with assistant tool calls."""
        msg = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "function": {"name": "test"}}],
        )
        result = msg.to_dict()

        assert result["role"] == "assistant"
        assert "tool_calls" in result


# =============================================================================
# Test ToolCall
# =============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_from_litellm_valid_json(self) -> None:
        """Test parsing tool call with valid JSON arguments."""
        mock_tc = create_mock_tool_call(
            call_id="call_456",
            name="search",
            arguments='{"query": "test"}',
        )

        result = ToolCall.from_litellm(mock_tc)

        assert result.id == "call_456"
        assert result.name == "search"
        assert result.arguments == {"query": "test"}
        assert result.raw_arguments == '{"query": "test"}'

    def test_from_litellm_invalid_json(self) -> None:
        """Test parsing tool call with invalid JSON arguments."""
        mock_tc = create_mock_tool_call(
            arguments="not valid json",
        )

        result = ToolCall.from_litellm(mock_tc)

        assert result.arguments == {}
        assert result.raw_arguments == "not valid json"


# =============================================================================
# Test ChatResponse
# =============================================================================


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_has_tool_calls_true(self) -> None:
        """Test has_tool_calls returns True when tool calls exist."""
        response = ChatResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="test", arguments={})],
        )
        assert response.has_tool_calls is True

    def test_has_tool_calls_false(self) -> None:
        """Test has_tool_calls returns False when no tool calls."""
        response = ChatResponse(content="Hello")
        assert response.has_tool_calls is False


# =============================================================================
# Test ToolResult
# =============================================================================


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_to_message_success(self) -> None:
        """Test converting successful result to message."""
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            content="Sunny, 25°C",
            success=True,
        )
        msg = result.to_message()

        assert msg.role == "tool"
        assert msg.content == "Sunny, 25°C"
        assert msg.name == "get_weather"
        assert msg.tool_call_id == "call_123"

    def test_to_message_failure(self) -> None:
        """Test converting failed result to message."""
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            content="",
            success=False,
            error="API timeout",
        )
        msg = result.to_message()

        assert msg.role == "tool"
        assert msg.content == "Error: API timeout"


# =============================================================================
# Test LLMClient Initialization
# =============================================================================


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_init_with_config(self, mock_config: Config) -> None:
        """Test initialization with config object."""
        client = LLMClient(mock_config)

        assert client.api_key == "test-api-key-12345"
        assert client.base_url == "https://api.test.com/v1"
        assert client.model == "zai/glm-4"
        assert client.temperature == 0.7
        assert client.max_tokens == 1000

    def test_init_adds_zai_prefix(self, mock_config: Config) -> None:
        """Test that zai/ prefix is added if missing."""
        mock_config.llm_model = "glm-4-flash"
        client = LLMClient(mock_config)

        assert client.model == "zai/glm-4-flash"

    def test_init_preserves_zai_prefix(self, mock_config: Config) -> None:
        """Test that zai/ prefix is preserved if present."""
        mock_config.llm_model = "zai/glm-4-plus"
        client = LLMClient(mock_config)

        assert client.model == "zai/glm-4-plus"

    def test_init_custom_retry_settings(self, mock_config: Config) -> None:
        """Test custom retry settings."""
        client = LLMClient(
            config=mock_config,
            max_retries=5,
            retry_delay=2.0,
            timeout=120.0,
        )

        assert client.max_retries == 5
        assert client.retry_delay == 2.0
        assert client.timeout == 120.0


# =============================================================================
# Test LLMClient.chat()
# =============================================================================


class TestLLMClientChat:
    """Tests for LLMClient.chat() method."""

    @patch("src.agent.llm_client.completion")
    def test_chat_basic(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test basic chat completion."""
        mock_completion.return_value = create_mock_response(
            content="Hello! How can I help you?"
        )

        messages = [{"role": "user", "content": "Hi"}]
        response = llm_client.chat(messages)

        assert response.content == "Hello! How can I help you?"
        assert response.finish_reason == "stop"
        assert response.has_tool_calls is False
        mock_completion.assert_called_once()

    @patch("src.agent.llm_client.completion")
    def test_chat_with_chat_message(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test chat with ChatMessage objects."""
        mock_completion.return_value = create_mock_response(content="Response")

        messages = [ChatMessage(role="user", content="Hello")]
        response = llm_client.chat(messages)

        assert response.content == "Response"
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "user"

    @patch("src.agent.llm_client.completion")
    def test_chat_with_tools(
        self, mock_completion: MagicMock, llm_client: LLMClient, mock_tool: Tool
    ) -> None:
        """Test chat with tools available."""
        mock_completion.return_value = create_mock_response(
            content="Let me check that."
        )

        messages = [{"role": "user", "content": "What's the weather?"}]
        response = llm_client.chat(messages, tools=[mock_tool])

        assert response.content == "Let me check that."
        call_kwargs = mock_completion.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    @patch("src.agent.llm_client.completion")
    def test_chat_with_tool_calls(
        self, mock_completion: MagicMock, llm_client: LLMClient, mock_tool: Tool
    ) -> None:
        """Test chat response with tool calls."""
        mock_tool_call = create_mock_tool_call()
        mock_completion.return_value = create_mock_response(
            content="",
            tool_calls=[mock_tool_call],
            finish_reason="tool_calls",
        )

        messages = [{"role": "user", "content": "Weather in Beijing?"}]
        response = llm_client.chat(messages, tools=[mock_tool])

        assert response.has_tool_calls is True
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Beijing"}

    @patch("src.agent.llm_client.completion")
    def test_chat_with_custom_params(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test chat with custom temperature and max_tokens."""
        mock_completion.return_value = create_mock_response()

        messages = [{"role": "user", "content": "Hello"}]
        llm_client.chat(messages, temperature=0.5, max_tokens=500)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500

    @patch("src.agent.llm_client.completion")
    def test_chat_includes_api_key(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test that API key is included in request."""
        mock_completion.return_value = create_mock_response()

        messages = [{"role": "user", "content": "Hello"}]
        llm_client.chat(messages)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_key"] == "test-api-key-12345"

    @patch("src.agent.llm_client.completion")
    def test_chat_includes_base_url(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test that base URL is included in request."""
        mock_completion.return_value = create_mock_response()

        messages = [{"role": "user", "content": "Hello"}]
        llm_client.chat(messages)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "https://api.test.com/v1"


# =============================================================================
# Test LLMClient.stream_chat()
# =============================================================================


class TestLLMClientStreamChat:
    """Tests for LLMClient.stream_chat() method."""

    @pytest.mark.anyio
    @patch("src.agent.llm_client.acompletion")
    async def test_stream_chat_basic(
        self, mock_acompletion: AsyncMock, llm_client: LLMClient
    ) -> None:
        """Test basic streaming chat."""

        # Create mock async generator
        async def mock_stream():
            for content in ["Hello", " there", "!"]:
                yield create_mock_stream_chunk(content)

        mock_acompletion.return_value = mock_stream()

        messages = [{"role": "user", "content": "Hi"}]
        chunks = []
        async for chunk in llm_client.stream_chat(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " there", "!"]

    @pytest.mark.anyio
    @patch("src.agent.llm_client.acompletion")
    async def test_stream_chat_with_tools(
        self, mock_acompletion: AsyncMock, llm_client: LLMClient, mock_tool: Tool
    ) -> None:
        """Test streaming chat with tools."""

        async def mock_stream():
            yield create_mock_stream_chunk("Response")

        mock_acompletion.return_value = mock_stream()

        messages = [{"role": "user", "content": "Hi"}]
        chunks = []
        async for chunk in llm_client.stream_chat(messages, tools=[mock_tool]):
            chunks.append(chunk)

        assert len(chunks) == 1
        call_kwargs = mock_acompletion.call_args[1]
        assert "tools" in call_kwargs


# =============================================================================
# Test LLMClient.execute_tool_calls()
# =============================================================================


class TestLLMClientExecuteToolCalls:
    """Tests for LLMClient.execute_tool_calls() method."""

    def test_execute_single_tool(self, llm_client: LLMClient, mock_tool: Tool) -> None:
        """Test executing a single tool call."""
        tool_calls = [
            ToolCall(
                id="call_1",
                name="get_weather",
                arguments={"location": "Shanghai"},
                raw_arguments='{"location": "Shanghai"}',
            )
        ]

        results = llm_client.execute_tool_calls(tool_calls, [mock_tool])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].name == "get_weather"
        assert "Shanghai" in results[0].content

    def test_execute_multiple_tools(self, llm_client: LLMClient) -> None:
        """Test executing multiple tool calls."""

        class Tool1(Tool):
            @property
            def name(self) -> str:
                return "tool1"

            @property
            def description(self) -> str:
                return "Tool 1"

            def execute(self, **kwargs: Any) -> Any:
                return "result1"

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        class Tool2(Tool):
            @property
            def name(self) -> str:
                return "tool2"

            @property
            def description(self) -> str:
                return "Tool 2"

            def execute(self, **kwargs: Any) -> Any:
                return {"data": "result2"}

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        tool_calls = [
            ToolCall(id="call_1", name="tool1", arguments={}),
            ToolCall(id="call_2", name="tool2", arguments={}),
        ]

        results = llm_client.execute_tool_calls(tool_calls, [Tool1(), Tool2()])

        assert len(results) == 2
        assert results[0].content == "result1"
        assert "result2" in results[1].content

    def test_execute_unknown_tool(self, llm_client: LLMClient, mock_tool: Tool) -> None:
        """Test executing an unknown tool."""
        tool_calls = [ToolCall(id="call_1", name="unknown_tool", arguments={})]

        results = llm_client.execute_tool_calls(tool_calls, [mock_tool])

        assert len(results) == 1
        assert results[0].success is False
        assert "Unknown tool" in results[0].error

    def test_execute_tool_with_error(self, llm_client: LLMClient) -> None:
        """Test executing a tool that raises an error."""

        class ErrorTool(Tool):
            @property
            def name(self) -> str:
                return "error_tool"

            @property
            def description(self) -> str:
                return "Tool that errors"

            def execute(self, **kwargs: Any) -> Any:
                raise ValueError("Something went wrong")

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        tool_calls = [ToolCall(id="call_1", name="error_tool", arguments={})]

        results = llm_client.execute_tool_calls(tool_calls, [ErrorTool()])

        assert len(results) == 1
        assert results[0].success is False
        assert "Something went wrong" in results[0].error

    def test_execute_tool_returns_dict(self, llm_client: LLMClient) -> None:
        """Test executing a tool that returns a dictionary."""

        class DictTool(Tool):
            @property
            def name(self) -> str:
                return "dict_tool"

            @property
            def description(self) -> str:
                return "Tool that returns dict"

            def execute(self, **kwargs: Any) -> Any:
                return {"temperature": 25, "humidity": 60}

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        tool_calls = [ToolCall(id="call_1", name="dict_tool", arguments={})]

        results = llm_client.execute_tool_calls(tool_calls, [DictTool()])

        assert results[0].success is True
        assert "temperature" in results[0].content


# =============================================================================
# Test LLMClient.chat_with_tools()
# =============================================================================


class TestLLMClientChatWithTools:
    """Tests for LLMClient.chat_with_tools() method."""

    @patch("src.agent.llm_client.completion")
    def test_chat_with_tools_no_tools_needed(
        self, mock_completion: MagicMock, llm_client: LLMClient, mock_tool: Tool
    ) -> None:
        """Test chat_with_tools when no tools are needed."""
        mock_completion.return_value = create_mock_response(
            content="Hello! How can I help?",
            finish_reason="stop",
        )

        messages = [{"role": "user", "content": "Hi"}]
        response = llm_client.chat_with_tools(messages, [mock_tool])

        assert response.content == "Hello! How can I help?"
        assert response.has_tool_calls is False
        assert mock_completion.call_count == 1

    @patch("src.agent.llm_client.completion")
    def test_chat_with_tools_executes_and_continues(
        self, mock_completion: MagicMock, llm_client: LLMClient, mock_tool: Tool
    ) -> None:
        """Test chat_with_tools executes tools and continues conversation."""
        # First call: returns tool call
        mock_tool_call = create_mock_tool_call()
        first_response = create_mock_response(
            content="",
            tool_calls=[mock_tool_call],
            finish_reason="tool_calls",
        )
        # Second call: returns final response
        second_response = create_mock_response(
            content="The weather in Beijing is sunny, 25°C.",
            finish_reason="stop",
        )

        mock_completion.side_effect = [first_response, second_response]

        messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
        response = llm_client.chat_with_tools(messages, [mock_tool])

        assert response.content == "The weather in Beijing is sunny, 25°C."
        assert mock_completion.call_count == 2

    @patch("src.agent.llm_client.completion")
    def test_chat_with_tools_max_iterations(
        self, mock_completion: MagicMock, llm_client: LLMClient, mock_tool: Tool
    ) -> None:
        """Test chat_with_tools respects max_iterations."""
        # Always return tool calls
        mock_tool_call = create_mock_tool_call()
        tool_response = create_mock_response(
            content="",
            tool_calls=[mock_tool_call],
            finish_reason="tool_calls",
        )

        mock_completion.return_value = tool_response

        messages = [{"role": "user", "content": "Weather?"}]
        response = llm_client.chat_with_tools(messages, [mock_tool], max_iterations=2)

        # Should stop after 2 tool iterations + 1 final answer request = 3 calls
        assert mock_completion.call_count == 3


# =============================================================================
# Test Error Handling
# =============================================================================


class TestLLMClientErrorHandling:
    """Tests for LLMClient error handling."""

    @patch("src.agent.llm_client.completion")
    @patch("src.agent.llm_client.time.sleep")
    def test_retry_on_rate_limit(
        self, mock_sleep: MagicMock, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test retry on rate limit error."""
        # First call: rate limit error
        mock_completion.side_effect = [
            Exception("Rate limit exceeded (429)"),
            create_mock_response(content="Success"),
        ]

        messages = [{"role": "user", "content": "Hello"}]
        response = llm_client.chat(messages)

        assert response.content == "Success"
        assert mock_completion.call_count == 2
        mock_sleep.assert_called_once()

    @patch("src.agent.llm_client.completion")
    @patch("src.agent.llm_client.time.sleep")
    def test_retry_on_timeout(
        self, mock_sleep: MagicMock, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test retry on timeout error."""
        mock_completion.side_effect = [
            Exception("Request timeout"),
            create_mock_response(content="Success"),
        ]

        messages = [{"role": "user", "content": "Hello"}]
        response = llm_client.chat(messages)

        assert response.content == "Success"
        assert mock_completion.call_count == 2

    @patch("src.agent.llm_client.completion")
    def test_raises_authentication_error(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test authentication error is raised properly."""
        mock_completion.side_effect = Exception("401 Unauthorized")

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(AuthenticationError):
            llm_client.chat(messages)

    @patch("src.agent.llm_client.completion")
    def test_raises_network_error(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test network error is raised properly."""
        mock_completion.side_effect = Exception("Connection refused")

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(NetworkError):
            llm_client.chat(messages)

    @patch("src.agent.llm_client.completion")
    @patch("src.agent.llm_client.time.sleep")
    def test_max_retries_exceeded(
        self, mock_sleep: MagicMock, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test that max retries are respected."""
        mock_completion.side_effect = Exception("Rate limit (429)")

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RateLimitError):
            llm_client.chat(messages)

        # Initial attempt + max_retries (2) = 3 total calls
        assert mock_completion.call_count == 2

    @patch("src.agent.llm_client.completion")
    @patch("src.agent.llm_client.time.sleep")
    def test_exponential_backoff(
        self, mock_sleep: MagicMock, mock_completion: MagicMock, mock_config: Config
    ) -> None:
        """Test exponential backoff timing."""
        client = LLMClient(mock_config, max_retries=3, retry_delay=1.0)

        mock_completion.side_effect = [
            Exception("Rate limit"),
            Exception("Rate limit"),
            create_mock_response(content="Success"),
        ]

        messages = [{"role": "user", "content": "Hello"}]
        client.chat(messages)

        # Check exponential backoff: 1.0, 2.0
        assert mock_sleep.call_count == 2
        calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert calls[0] == 1.0  # 1.0 * 2^0
        assert calls[1] == 2.0  # 1.0 * 2^1


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestLLMClientEdgeCases:
    """Tests for edge cases and special scenarios."""

    @patch("src.agent.llm_client.completion")
    def test_empty_content(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test handling of empty content."""
        mock_completion.return_value = create_mock_response(content="")

        messages = [{"role": "user", "content": "Hello"}]
        response = llm_client.chat(messages)

        assert response.content == ""

    @patch("src.agent.llm_client.completion")
    def test_system_message(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test chat with system message."""
        mock_completion.return_value = create_mock_response()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        llm_client.chat(messages)

        call_kwargs = mock_completion.call_args[1]
        assert len(call_kwargs["messages"]) == 2

    def test_convert_empty_tools(self, llm_client: LLMClient) -> None:
        """Test converting empty tools list returns empty list."""
        result = llm_client._convert_tools([])
        # Empty list should return empty list (not None)
        assert result == []

    def test_convert_none_tools(self, llm_client: LLMClient) -> None:
        """Test converting None tools."""
        result = llm_client._convert_tools(None)
        assert result is None

    @patch("src.agent.llm_client.completion")
    def test_no_usage_in_response(
        self, mock_completion: MagicMock, llm_client: LLMClient
    ) -> None:
        """Test handling response without usage info."""
        mock_response = create_mock_response()
        mock_response.usage = None
        mock_completion.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = llm_client.chat(messages)

        assert response.usage == {}

    @pytest.mark.anyio
    @patch("src.agent.llm_client.acompletion")
    async def test_stream_empty_chunks(
        self, mock_acompletion: AsyncMock, llm_client: LLMClient
    ) -> None:
        """Test streaming with empty chunks - empty chunks are skipped."""

        async def mock_stream():
            yield create_mock_stream_chunk("Hello")
            yield create_mock_stream_chunk("")  # Empty chunk - should be skipped
            yield create_mock_stream_chunk(" world")

        mock_acompletion.return_value = mock_stream()

        messages = [{"role": "user", "content": "Hi"}]
        chunks = []
        async for chunk in llm_client.stream_chat(messages):
            chunks.append(chunk)

        # Empty chunks are skipped, so only 2 chunks should be received
        assert len(chunks) == 2
        assert chunks == ["Hello", " world"]


# =============================================================================
# Test Tool Call Deduplication
# =============================================================================


class TestToolCallDedup:
    """Tests for _ToolCallCache and tool call deduplication in execute_tool_calls()."""

    def test_cache_hit_skips_execute(self, llm_client: LLMClient) -> None:
        """Repeated tool+args within same session returns cached result without re-executing."""

        call_count = 0

        class CountingTool(Tool):
            @property
            def name(self) -> str:
                return "counter"

            @property
            def description(self) -> str:
                return "counting tool"

            def execute(self, **kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                return f"invocation_{call_count}"

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        tool = CountingTool()
        tool_calls = [
            ToolCall(id="c1", name="counter", arguments={"x": 1}),
            ToolCall(id="c2", name="counter", arguments={"x": 1}),
        ]

        # Set up a fresh cache (simulates chat_with_tools init)
        from src.agent.llm_client import _ToolCallCache

        llm_client._tool_call_cache = _ToolCallCache()

        results = llm_client.execute_tool_calls(tool_calls, [tool])

        assert call_count == 1  # Only first call actually executed
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True
        assert results[0].content == "invocation_1"
        assert results[1].content == "invocation_1"  # Cached

    def test_cache_miss_different_args(self, llm_client: LLMClient) -> None:
        """Different arguments produce separate cache entries."""

        call_count = 0

        class CountingTool(Tool):
            @property
            def name(self) -> str:
                return "counter"

            @property
            def description(self) -> str:
                return "counting tool"

            def execute(self, **kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                return f"invocation_{call_count}"

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        tool = CountingTool()
        tool_calls = [
            ToolCall(id="c1", name="counter", arguments={"x": 1}),
            ToolCall(id="c2", name="counter", arguments={"x": 2}),
        ]

        from src.agent.llm_client import _ToolCallCache

        llm_client._tool_call_cache = _ToolCallCache()
        results = llm_client.execute_tool_calls(tool_calls, [tool])

        assert call_count == 2  # Both executed — different args
        assert results[0].content == "invocation_1"
        assert results[1].content == "invocation_2"

    def test_cache_ttl_expiry_re_executes(self, llm_client: LLMClient) -> None:
        """Expired cache entries re-execute the tool."""

        call_count = 0

        class CountingTool(Tool):
            @property
            def name(self) -> str:
                return "counter"

            @property
            def description(self) -> str:
                return "counting tool"

            def execute(self, **kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                return f"invocation_{call_count}"

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        tool = CountingTool()

        from src.agent.llm_client import _ToolCallCache

        # Use a very short TTL
        cache = _ToolCallCache(max_age=0.01)
        llm_client._tool_call_cache = cache

        # First call populates cache
        tc1 = ToolCall(id="c1", name="counter", arguments={"x": 1})
        r1 = llm_client.execute_tool_calls([tc1], [tool])
        assert call_count == 1

        # Wait for TTL to expire
        import time

        time.sleep(0.05)

        # Same args — cache expired, should re-execute
        tc2 = ToolCall(id="c2", name="counter", arguments={"x": 1})
        r2 = llm_client.execute_tool_calls([tc2], [tool])
        assert call_count == 2

    def test_cache_max_size_eviction(self) -> None:
        """Cache evicts oldest entries when max_size is exceeded."""
        from src.agent.llm_client import _ToolCallCache

        cache = _ToolCallCache(max_age=60.0, max_size=2)

        cache.put("tool_a", {"x": 1}, "result_a")
        cache.put("tool_b", {"x": 2}, "result_b")
        assert cache.get("tool_a", {"x": 1}) == "result_a"
        assert cache.get("tool_b", {"x": 2}) == "result_b"

        # This should evict tool_a (oldest)
        cache.put("tool_c", {"x": 3}, "result_c")
        assert cache.get("tool_a", {"x": 1}) is None  # Evicted
        assert cache.get("tool_b", {"x": 2}) == "result_b"
        assert cache.get("tool_c", {"x": 3}) == "result_c"

    def test_cache_clear(self) -> None:
        """Cache clear removes all entries."""
        from src.agent.llm_client import _ToolCallCache

        cache = _ToolCallCache()
        cache.put("tool_a", {"x": 1}, "result_a")
        cache.put("tool_b", {"x": 2}, "result_b")

        cache.clear()

        assert cache.get("tool_a", {"x": 1}) is None
        assert cache.get("tool_b", {"x": 2}) is None

    def test_no_cache_without_chat_with_tools(self, llm_client: LLMClient) -> None:
        """execute_tool_calls works without cache (no _tool_call_cache attribute)."""
        # Ensure no cache attribute exists
        if hasattr(llm_client, "_tool_call_cache"):
            del llm_client._tool_call_cache

        class SimpleTool(Tool):
            @property
            def name(self) -> str:
                return "simple"

            @property
            def description(self) -> str:
                return "simple tool"

            def execute(self, **kwargs: Any) -> Any:
                return "ok"

            def to_openai_tool(self) -> Dict[str, Any]:
                return {"type": "function", "function": {"name": self.name}}

        tc = ToolCall(id="c1", name="simple", arguments={})
        results = llm_client.execute_tool_calls([tc, tc], [SimpleTool()])

        # Both executed (no dedup without cache)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_cache_lru_promotion(self) -> None:
        """Accessing a cached entry promotes it (LRU), preventing eviction."""
        from src.agent.llm_client import _ToolCallCache

        cache = _ToolCallCache(max_age=60.0, max_size=3)

        cache.put("a", {}, "ra")
        cache.put("b", {}, "rb")
        cache.put("c", {}, "rc")

        # Access "a" — promotes it to newest
        assert cache.get("a", {}) == "ra"

        # Insert "d" — should evict "b" (now oldest), not "a"
        cache.put("d", {}, "rd")
        assert cache.get("a", {}) == "ra"  # Still present
        assert cache.get("b", {}) is None  # Evicted
        assert cache.get("c", {}) == "rc"
        assert cache.get("d", {}) == "rd"


# =============================================================================
# Test Budget Enforcement
# =============================================================================


class TestBudgetEnforcement:
    """Tests for budget_guard hard enforcement in chat()."""

    def _make_blocking_guard(self) -> MagicMock:
        """Build a mock BudgetGuard whose check_budget() reports an exceeded,
        blocking budget."""
        guard = MagicMock()
        alert = BudgetAlert(
            budget_name="monthly",
            period="monthly",
            current_spend_cents=12000,
            limit_cents=10000,
            usage_percent=120.0,
            is_exceeded=True,
            should_block=True,
        )
        result = BudgetCheckResult(is_within_budget=False, alerts=[alert])
        guard.check_budget.return_value = result
        return guard

    def _make_ok_guard(self) -> MagicMock:
        """Build a mock BudgetGuard whose check_budget() reports all clear."""
        guard = MagicMock()
        result = BudgetCheckResult(is_within_budget=True, alerts=[])
        guard.check_budget.return_value = result
        return guard

    @patch("src.agent.llm_client.completion")
    def test_budget_exceeded_raises(
        self, mock_completion: MagicMock, mock_config: Config
    ) -> None:
        """chat() raises BudgetExceededError when a blocking budget is exceeded."""
        guard = self._make_blocking_guard()
        client = LLMClient(mock_config, budget_guard=guard)

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(BudgetExceededError):
            client.chat(messages)

        mock_completion.assert_not_called()

    @patch("src.agent.llm_client.completion")
    def test_budget_ok_allows_chat(
        self, mock_completion: MagicMock, mock_config: Config
    ) -> None:
        """chat() proceeds normally when budget is within limits."""
        guard = self._make_ok_guard()
        client = LLMClient(mock_config, budget_guard=guard)

        mock_completion.return_value = create_mock_response(content="ok")

        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response.content == "ok"
        guard.check_budget.assert_called_once()
        mock_completion.assert_called_once()

    @patch("src.agent.llm_client.completion")
    def test_no_budget_guard_skips_check(
        self, mock_completion: MagicMock, mock_config: Config
    ) -> None:
        """chat() works normally when no budget_guard is provided."""
        client = LLMClient(mock_config)

        mock_completion.return_value = create_mock_response(content="ok")

        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response.content == "ok"
        mock_completion.assert_called_once()

    @patch("src.agent.llm_client.completion")
    def test_exceeded_but_no_block_allows_chat(
        self, mock_completion: MagicMock, mock_config: Config
    ) -> None:
        """chat() proceeds when budget is exceeded but should_block is False."""
        guard = MagicMock()
        alert = BudgetAlert(
            budget_name="advisory",
            period="monthly",
            current_spend_cents=12000,
            limit_cents=10000,
            usage_percent=120.0,
            is_exceeded=True,
            should_block=False,
        )
        result = BudgetCheckResult(is_within_budget=False, alerts=[alert])
        guard.check_budget.return_value = result

        client = LLMClient(mock_config, budget_guard=guard)
        mock_completion.return_value = create_mock_response(content="ok")

        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat(messages)

        assert response.content == "ok"
        mock_completion.assert_called_once()

    def test_budget_exceeded_error_is_llm_client_error(self) -> None:
        """BudgetExceededError is a subclass of LLMClientError."""
        assert issubclass(BudgetExceededError, LLMClientError)

    @patch("src.agent.llm_client.completion")
    def test_cached_result_skips_budget_check(
        self, mock_completion: MagicMock, mock_config: Config
    ) -> None:
        """Cached results (zero tokens) bypass budget enforcement."""
        guard = self._make_blocking_guard()
        client = LLMClient(mock_config, budget_guard=guard)

        messages = [{"role": "user", "content": "Hello"}]
        client._llm_cache.put(
            [{"role": "user", "content": "Hello"}],
            client.model,
            client.temperature,
            client.max_tokens,
            "cached-content",
        )

        response = client.chat(messages)

        assert response.cache_hit is True
        assert response.content == "cached-content"
        guard.check_budget.assert_not_called()
        mock_completion.assert_not_called()
