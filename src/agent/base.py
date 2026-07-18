"""Agent base classes and protocols for doc-search application.

This module defines the core abstractions for the agent system:
- Tool: Protocol/ABC for agent tools
- AgentResponse: Dataclass for agent responses
- Agent: Abstract base class for agents
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


class Tool(ABC):
    """Abstract base class for agent tools.

    Tools are executable functions that agents can use to perform
    specific tasks like searching, document retrieval, etc.

    Subclasses must implement:
    - name: str property - unique tool identifier
    - description: str property - what the tool does
    - execute(**kwargs) -> Any - the actual tool logic
    - to_openai_tool() -> Dict - convert to OpenAI function calling format
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool.

        Returns:
            str: The tool's unique name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does.

        Returns:
            str: Description of the tool's functionality
        """
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for tool execution

        Returns:
            Any: The result of tool execution
        """
        pass

    @abstractmethod
    def to_openai_tool(self) -> dict[str, Any]:
        """Convert tool to OpenAI function calling format.

        Returns:
            Dict[str, Any]: Tool definition in OpenAI format:
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "Tool description",
                        "parameters": {
                            "type": "object",
                            "properties": {...},
                            "required": [...]
                        }
                    }
                }
        """
        pass


@dataclass
class AgentResponse:
    """Dataclass representing an agent's response.

    Attributes:
        success: Whether the agent successfully completed the task
        answer: The main response text/answer
        sources: List of source document paths used (backward compat)
        search_hits: List of structured search hit details for display
        tool_calls: List of tool calls made during execution
        reasoning: The agent's reasoning process (optional)
        confidence: Agent's confidence score (0.0-1.0)
        tokens_used: Total tokens consumed during processing
        processing_time: Time taken to process the request in seconds
        error: Error message if success is False (optional)
    """

    success: bool
    answer: str
    sources: list[str] = field(default_factory=list)
    search_hits: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    tokens_used: int = 0
    processing_time: float = 0.0
    error: str | None = None

    # Diagnostics fields (populated by DiagnosticsCollector)
    step_timings: dict[str, float] = field(default_factory=dict)
    llm_call_count: int = 0
    llm_latency_total: float = 0.0
    tool_execution_total: float = 0.0
    cache_hits: int = 0
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the response
        """
        return {
            "success": self.success,
            "answer": self.answer,
            "sources": self.sources,
            "search_hits": self.search_hits,
            "tool_calls": self.tool_calls,
            "reasoning": self.reasoning,
            "tokens_used": self.tokens_used,
            "processing_time": self.processing_time,
            "error": self.error,
            "step_timings": self.step_timings,
            "llm_call_count": self.llm_call_count,
            "llm_latency_total": self.llm_latency_total,
            "tool_execution_total": self.tool_execution_total,
            "cache_hits": self.cache_hits,
            "retry_count": self.retry_count,
        }

    @classmethod
    def error_response(
        cls, error: str, processing_time: float = 0.0
    ) -> "AgentResponse":
        """Create an error response.

        Args:
            error: The error message
            processing_time: Time taken before error occurred

        Returns:
            AgentResponse: A response indicating failure
        """
        return cls(
            success=False,
            answer="",
            error=error,
            processing_time=processing_time,
        )


class Agent(ABC):
    """Abstract base class for agents.

    Agents are intelligent entities that can use tools to accomplish tasks.
    They maintain a registry of available tools and can execute them on demand.

    Subclasses must implement:
    - name: str property - agent identifier
    - run(query, context) -> AgentResponse - main execution logic
    """

    def __init__(self) -> None:
        """Initialize agent with empty tool registry."""
        self._tools: dict[str, Tool] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the agent.

        Returns:
            str: The agent's unique name
        """
        pass

    @property
    def tools(self) -> list[Tool]:
        """List of registered tools.

        Returns:
            List[Tool]: All registered tools
        """
        return list(self._tools.values())

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the agent.

        Args:
            tool: The tool to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: The name of the tool to unregister

        Returns:
            bool: True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Tool | None:
        """Get a registered tool by name.

        Args:
            name: The name of the tool

        Returns:
            Optional[Tool]: The tool if found, None otherwise
        """
        return self._tools.get(name)

    def execute_tool(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool by name.

        Args:
            name: The name of the tool to execute
            **kwargs: Arguments to pass to the tool's execute method

        Returns:
            Any: The result of tool execution

        Raises:
            KeyError: If no tool with the given name is registered
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        return self._tools[name].execute(**kwargs)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all registered tools in OpenAI function calling format.

        Returns:
            List[Dict[str, Any]]: List of tool definitions in OpenAI format
        """
        return [tool.to_openai_tool() for tool in self.tools]

    @abstractmethod
    def run(
        self, query: str, context: dict[str, Any] | None = None
    ) -> AgentResponse:
        """Execute the agent's main logic.

        Args:
            query: The user's query or task
            context: Optional context information for the query

        Returns:
            AgentResponse: The agent's response
        """
        pass

    def _record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        """Record a tool call for the response.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            result: Result returned by the tool

        Returns:
            Dict[str, Any]: Tool call record
        """
        return {
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
        }


class ToolResult:
    """Helper class to wrap tool execution results.

    Provides a standardized way to return results from tools,
    including success/failure status and optional metadata.
    """

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize tool result.

        Args:
            success: Whether the tool execution was successful
            data: The result data (if successful)
            error: Error message (if failed)
            metadata: Optional additional metadata
        """
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        result = {"success": self.success, "data": self.data}
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def ok(
        cls, data: Any = None, metadata: dict[str, Any] | None = None
    ) -> "ToolResult":
        """Create a successful result.

        Args:
            data: The result data
            metadata: Optional metadata

        Returns:
            ToolResult: A successful result
        """
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(
        cls, error: str, metadata: dict[str, Any] | None = None
    ) -> "ToolResult":
        """Create a failed result.

        Args:
            error: The error message
            metadata: Optional metadata

        Returns:
            ToolResult: A failed result
        """
        return cls(success=False, error=error, metadata=metadata)


class ToolCache:
    """TTL-based LRU cache for tool execution results.

    Stores ToolResult objects keyed by a deterministic hash of tool name
    and parameters. Supports automatic expiry (TTL) and bounded size with
    LRU eviction.

    Args:
        ttl: Time-to-live in seconds for cached entries (default 300).
        max_size: Maximum number of entries (default 128).
    """

    def __init__(self, ttl: float = 300, max_size: int = 128) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[float, ToolResult]] = OrderedDict()

    # -- public API -------------------------------------------------------

    def get(self, key: str) -> Optional["ToolResult"]:
        """Return cached ToolResult if present and not expired, else None.

        A successful get promotes the entry to the most-recently-used position.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, result = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        # promote (LRU by access time)
        self._store.move_to_end(key)
        return result

    def put(self, key: str, result: "ToolResult") -> None:
        """Store a ToolResult under *key*, evicting LRU entries if needed."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.time(), result)
        # evict oldest entries when over capacity
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    @staticmethod
    def make_key(tool_name: str, kwargs: dict[str, Any]) -> str:
        """Return a deterministic cache key from tool name and parameters.

        The key is the MD5 hex-digest of ``"<tool_name>:<sorted-json>"``.
        """
        raw = f"{tool_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        return hashlib.md5(raw.encode()).hexdigest()
