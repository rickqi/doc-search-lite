"""Unit tests for agent base classes.

Tests cover:
- Tool ABC and protocol
- AgentResponse dataclass
- Agent ABC with tool registry
- ToolResult helper class
"""

import pytest

from src.agent.base import Agent, AgentResponse, Tool, ToolResult


# Concrete implementations for testing
class MockTool(Tool):
    """Mock tool implementation for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        execute_result: any = "mock_result",
    ) -> None:
        self._name = name
        self._description = description
        self._execute_result = execute_result
        self.execute_call_count = 0
        self.last_kwargs = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs) -> any:
        self.execute_call_count += 1
        self.last_kwargs = kwargs
        return self._execute_result

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }


class MockAgent(Agent):
    """Mock agent implementation for testing."""

    def __init__(self, name: str = "mock_agent") -> None:
        super().__init__()
        self._name = name
        self.run_call_count = 0
        self.last_query = None
        self.last_context = None

    @property
    def name(self) -> str:
        return self._name

    def run(self, query: str, context: dict | None = None) -> AgentResponse:
        self.run_call_count += 1
        self.last_query = query
        self.last_context = context

        # Execute any registered tools if context says to
        if context and "execute_tool" in context:
            tool_name = context["execute_tool"]
            tool_args = context.get("tool_args", {})
            result = self.execute_tool(tool_name, **tool_args)
            tool_call = self._record_tool_call(tool_name, tool_args, result)
            return AgentResponse(
                success=True,
                answer=f"Executed {tool_name}: {result}",
                tool_calls=[tool_call],
            )

        return AgentResponse(
            success=True,
            answer=f"Processed: {query}",
        )


class TestTool:
    """Tests for Tool ABC."""

    def test_tool_name_property(self) -> None:
        """Test tool name property."""
        tool = MockTool(name="test_tool")
        assert tool.name == "test_tool"

    def test_tool_description_property(self) -> None:
        """Test tool description property."""
        tool = MockTool(description="Test description")
        assert tool.description == "Test description"

    def test_tool_execute(self) -> None:
        """Test tool execute method."""
        tool = MockTool(execute_result="test_result")
        result = tool.execute(query="test", limit=10)
        assert result == "test_result"
        assert tool.execute_call_count == 1
        assert tool.last_kwargs == {"query": "test", "limit": 10}

    def test_tool_execute_multiple_times(self) -> None:
        """Test tool can be executed multiple times."""
        tool = MockTool()
        tool.execute(a=1)
        tool.execute(b=2)
        assert tool.execute_call_count == 2

    def test_tool_to_openai_tool_format(self) -> None:
        """Test to_openai_tool returns correct format."""
        tool = MockTool(name="search", description="Search documents")
        openai_tool = tool.to_openai_tool()

        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "search"
        assert openai_tool["function"]["description"] == "Search documents"
        assert "parameters" in openai_tool["function"]
        assert openai_tool["function"]["parameters"]["type"] == "object"

    def test_tool_cannot_be_instantiated_directly(self) -> None:
        """Test Tool ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Tool()  # type: ignore


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_default_values(self) -> None:
        """Test AgentResponse default values."""
        response = AgentResponse(success=True, answer="Test answer")
        assert response.success is True
        assert response.answer == "Test answer"
        assert response.sources == []
        assert response.tool_calls == []
        assert response.reasoning == ""
        assert response.tokens_used == 0
        assert response.processing_time == 0.0
        assert response.error is None

    def test_all_fields(self) -> None:
        """Test AgentResponse with all fields specified."""
        response = AgentResponse(
            success=True,
            answer="Full answer",
            sources=["doc1.pdf", "doc2.pdf"],
            tool_calls=[{"tool": "search", "args": {"q": "test"}}],
            reasoning="Step by step",
            tokens_used=100,
            processing_time=1.5,
            error=None,
        )
        assert response.success is True
        assert response.answer == "Full answer"
        assert len(response.sources) == 2
        assert len(response.tool_calls) == 1
        assert response.reasoning == "Step by step"
        assert response.tokens_used == 100
        assert response.processing_time == 1.5
        assert response.error is None

    def test_error_response_factory(self) -> None:
        """Test error_response factory method."""
        response = AgentResponse.error_response(
            "Something went wrong", processing_time=0.5
        )
        assert response.success is False
        assert response.answer == ""
        assert response.error == "Something went wrong"
        assert response.processing_time == 0.5

    def test_error_response_default_processing_time(self) -> None:
        """Test error_response with default processing time."""
        response = AgentResponse.error_response("Error")
        assert response.processing_time == 0.0

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        response = AgentResponse(
            success=True,
            answer="Test",
            sources=["s1"],
            tokens_used=50,
        )
        result = response.to_dict()
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["answer"] == "Test"
        assert result["sources"] == ["s1"]
        assert result["tokens_used"] == 50
        assert result["error"] is None


class TestAgent:
    """Tests for Agent ABC."""

    def test_agent_name_property(self) -> None:
        """Test agent name property."""
        agent = MockAgent(name="test_agent")
        assert agent.name == "test_agent"

    def test_agent_tools_initially_empty(self) -> None:
        """Test agent starts with no tools."""
        agent = MockAgent()
        assert agent.tools == []

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        agent = MockAgent()
        tool = MockTool(name="search")
        agent.register_tool(tool)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "search"

    def test_register_multiple_tools(self) -> None:
        """Test registering multiple tools."""
        agent = MockAgent()
        agent.register_tool(MockTool(name="search"))
        agent.register_tool(MockTool(name="retrieve"))
        assert len(agent.tools) == 2
        names = [t.name for t in agent.tools]
        assert "search" in names
        assert "retrieve" in names

    def test_register_duplicate_tool_raises_error(self) -> None:
        """Test registering duplicate tool raises ValueError."""
        agent = MockAgent()
        agent.register_tool(MockTool(name="search"))
        with pytest.raises(ValueError, match="already registered"):
            agent.register_tool(MockTool(name="search"))

    def test_unregister_tool(self) -> None:
        """Test unregistering a tool."""
        agent = MockAgent()
        agent.register_tool(MockTool(name="search"))
        result = agent.unregister_tool("search")
        assert result is True
        assert len(agent.tools) == 0

    def test_unregister_nonexistent_tool(self) -> None:
        """Test unregistering non-existent tool returns False."""
        agent = MockAgent()
        result = agent.unregister_tool("nonexistent")
        assert result is False

    def test_get_tool(self) -> None:
        """Test getting a tool by name."""
        agent = MockAgent()
        tool = MockTool(name="search")
        agent.register_tool(tool)
        retrieved = agent.get_tool("search")
        assert retrieved is tool

    def test_get_nonexistent_tool(self) -> None:
        """Test getting non-existent tool returns None."""
        agent = MockAgent()
        retrieved = agent.get_tool("nonexistent")
        assert retrieved is None

    def test_execute_tool(self) -> None:
        """Test executing a tool by name."""
        agent = MockAgent()
        tool = MockTool(name="search", execute_result="search results")
        agent.register_tool(tool)
        result = agent.execute_tool("search", query="test")
        assert result == "search results"
        assert tool.execute_call_count == 1

    def test_execute_nonexistent_tool_raises_error(self) -> None:
        """Test executing non-existent tool raises KeyError."""
        agent = MockAgent()
        with pytest.raises(KeyError, match="No tool registered"):
            agent.execute_tool("nonexistent")

    def test_get_openai_tools(self) -> None:
        """Test getting all tools in OpenAI format."""
        agent = MockAgent()
        agent.register_tool(MockTool(name="search"))
        agent.register_tool(MockTool(name="retrieve"))
        openai_tools = agent.get_openai_tools()
        assert len(openai_tools) == 2
        assert all(t["type"] == "function" for t in openai_tools)

    def test_run_abstract_method(self) -> None:
        """Test run method implementation."""
        agent = MockAgent()
        response = agent.run("test query")
        assert response.success is True
        assert "test query" in response.answer
        assert agent.run_call_count == 1

    def test_run_with_context(self) -> None:
        """Test run method with context."""
        agent = MockAgent()
        context = {"user": "test_user", "session_id": "123"}
        agent.run("query", context=context)
        assert agent.last_context == context

    def test_run_with_tool_execution(self) -> None:
        """Test run method executing a tool."""
        agent = MockAgent()
        tool = MockTool(name="search", execute_result="found 3 docs")
        agent.register_tool(tool)
        context = {"execute_tool": "search", "tool_args": {"query": "python"}}
        response = agent.run("find python docs", context=context)
        assert response.success is True
        assert "search" in response.answer
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["tool"] == "search"

    def test_record_tool_call(self) -> None:
        """Test _record_tool_call helper method."""
        agent = MockAgent()
        record = agent._record_tool_call(
            "search",
            {"query": "test"},
            "results",
        )
        assert record["tool"] == "search"
        assert record["arguments"] == {"query": "test"}
        assert record["result"] == "results"

    def test_agent_cannot_be_instantiated_directly(self) -> None:
        """Test Agent ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Agent()  # type: ignore


class TestToolResult:
    """Tests for ToolResult helper class."""

    def test_ok_factory(self) -> None:
        """Test ok factory method."""
        result = ToolResult.ok(data="success data")
        assert result.success is True
        assert result.data == "success data"
        assert result.error is None

    def test_ok_with_metadata(self) -> None:
        """Test ok factory with metadata."""
        result = ToolResult.ok(data="data", metadata={"count": 5})
        assert result.metadata == {"count": 5}

    def test_fail_factory(self) -> None:
        """Test fail factory method."""
        result = ToolResult.fail("something went wrong")
        assert result.success is False
        assert result.error == "something went wrong"
        assert result.data is None

    def test_fail_with_metadata(self) -> None:
        """Test fail factory with metadata."""
        result = ToolResult.fail("error", metadata={"code": 500})
        assert result.metadata == {"code": 500}

    def test_to_dict_success(self) -> None:
        """Test to_dict for successful result."""
        result = ToolResult.ok(data="test", metadata={"key": "value"})
        d = result.to_dict()
        assert d["success"] is True
        assert d["data"] == "test"
        assert d["metadata"] == {"key": "value"}
        assert "error" not in d

    def test_to_dict_failure(self) -> None:
        """Test to_dict for failed result."""
        result = ToolResult.fail("error message", metadata={"code": 404})
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "error message"
        assert d["metadata"] == {"code": 404}

    def test_default_metadata(self) -> None:
        """Test default empty metadata."""
        result = ToolResult.ok()
        assert result.metadata == {}


class TestIntegration:
    """Integration tests for agent framework."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: create agent, register tools, run query."""
        agent = MockAgent(name="doc_assistant")

        # Register tools
        search_tool = MockTool(name="search", execute_result="Found 5 documents")
        retrieve_tool = MockTool(
            name="retrieve", execute_result="Document content here"
        )
        agent.register_tool(search_tool)
        agent.register_tool(retrieve_tool)

        # Verify tools registered
        assert len(agent.tools) == 2

        # Run query with tool execution
        context = {"execute_tool": "search", "tool_args": {"query": "python"}}
        response = agent.run("Find python docs", context=context)

        # Verify response
        assert response.success is True
        assert len(response.tool_calls) == 1
        assert "search" in response.answer

    def test_multiple_tool_executions(self) -> None:
        """Test agent executing multiple tools sequentially."""
        agent = MockAgent()

        # Register tools
        agent.register_tool(MockTool(name="search", execute_result=["doc1", "doc2"]))
        agent.register_tool(MockTool(name="retrieve", execute_result="Full text"))

        # Execute tools directly
        search_result = agent.execute_tool("search", query="test")
        retrieve_result = agent.execute_tool("retrieve", doc_id="doc1")

        assert search_result == ["doc1", "doc2"]
        assert retrieve_result == "Full text"

    def test_agent_response_error_handling(self) -> None:
        """Test agent response with error."""
        response = AgentResponse.error_response("Processing failed")
        assert response.success is False
        assert response.error == "Processing failed"
        assert response.answer == ""
