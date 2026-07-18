"""LiteLLM-based LLM client with Zhipu GLM integration.

This module provides a unified LLM client that integrates with LiteLLM
for Zhipu GLM models, supporting chat completions, streaming, and tool calling.
"""

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Union

import litellm
from litellm import acompletion, completion

from src.agent.base import Tool
from src.utils.config import Config

logger = logging.getLogger(__name__)


class _ToolCallCache:
    """LRU cache for tool call deduplication within a single chat_with_tools session."""

    def __init__(self, max_age: float = 60.0, max_size: int = 50) -> None:
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_age = max_age
        self._max_size = max_size

    def _make_key(self, tool_name: str, arguments: dict) -> str:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"

    def get(self, tool_name: str, arguments: dict) -> Any | None:
        key = self._make_key(tool_name, arguments)
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self._max_age:
                # Promote (LRU)
                self._cache.move_to_end(key)
                return result
            del self._cache[key]
        return None

    def put(self, tool_name: str, arguments: dict, result: Any) -> None:
        key = self._make_key(tool_name, arguments)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Evict oldest
        self._cache[key] = (time.time(), result)

    def clear(self) -> None:
        self._cache.clear()


class LLMResultCache:
    """In-memory LRU cache for deterministic LLM call results.

    Caches expansion, decomposition, sufficiency, verification results
    keyed on (prompt_hash, model, temperature).  TTL-based expiry with
    LRU eviction.  Inspired by RAGFlow's xxhash LLM cache pattern.

    Thread-safe via OrderedDict (single-threaded agent).
    """

    def __init__(self, ttl: float = 3600.0, max_size: int = 128) -> None:
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._ttl = ttl
        self._max_size = max_size

    @staticmethod
    def _hash_key(messages: Any, model: str, temperature: float, max_tokens: int) -> str:
        """Create a stable cache key from call parameters."""
        import hashlib
        raw = f"{json.dumps(messages, sort_keys=True, default=str)}|{model}|{temperature}|{max_tokens}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, messages: Any, model: str, temperature: float, max_tokens: int) -> str | None:
        key = self._hash_key(messages, model, temperature, max_tokens)
        if key in self._cache:
            ts, content = self._cache[key]
            if time.time() - ts < self._ttl:
                self._cache.move_to_end(key)
                return content
            del self._cache[key]
        return None

    def put(self, messages: Any, model: str, temperature: float, max_tokens: int, content: str) -> None:
        key = self._hash_key(messages, model, temperature, max_tokens)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), content)

    def clear(self) -> None:
        self._cache.clear()


@dataclass
class ChatMessage:
    """Represents a chat message.

    Attributes:
        role: The role of the message sender (system, user, assistant, tool)
        content: The content of the message
        name: Optional name for tool messages
        tool_call_id: Optional tool call ID for tool response messages
        tool_calls: Optional list of tool calls from assistant
    """

    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LiteLLM.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        result: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


@dataclass
class ToolCall:
    """Represents a tool call from the LLM.

    Attributes:
        id: Unique identifier for the tool call
        name: Name of the function to call
        arguments: Arguments as a dictionary
        raw_arguments: Raw arguments string from LLM
    """

    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""

    @classmethod
    def from_litellm(cls, tool_call: Any) -> "ToolCall":
        """Create ToolCall from LiteLLM response tool call.

        Args:
            tool_call: Tool call object from LiteLLM response

        Returns:
            ToolCall: Parsed tool call
        """
        raw_args = tool_call.function.arguments
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
            logger.warning(f"Failed to parse tool arguments: {raw_args}")

        return cls(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=args,
            raw_arguments=raw_args,
        )


@dataclass
class ChatResponse:
    """Represents a chat completion response.

    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls if any
        finish_reason: Reason for completion (stop, tool_calls, etc.)
        usage: Token usage information
        model: Model used for completion
        raw_response: Raw response from LiteLLM
    """

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw_response: Optional[Any] = None
    latency_ms: float = 0.0
    cache_hit: bool = False

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls.

        Returns:
            bool: True if there are tool calls
        """
        return len(self.tool_calls) > 0


@dataclass
class ToolResult:
    """Represents the result of a tool execution.

    Attributes:
        tool_call_id: ID of the tool call this result is for
        name: Name of the tool that was executed
        content: The result content as a string
        success: Whether the tool execution was successful
        error: Error message if execution failed
    """

    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_message(self) -> ChatMessage:
        """Convert to a ChatMessage for the conversation.

        Returns:
            ChatMessage: Message containing the tool result
        """
        return ChatMessage(
            role="tool",
            content=self.content if self.success else f"Error: {self.error}",
            name=self.name,
            tool_call_id=self.tool_call_id,
        )


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class RateLimitError(LLMClientError):
    """Raised when API rate limit is exceeded."""

    pass


class AuthenticationError(LLMClientError):
    """Raised when API authentication fails."""

    pass


class NetworkError(LLMClientError):
    """Raised when network-related errors occur."""

    pass


class BudgetExceededError(LLMClientError):
    """Raised when a budget with block_exceed=True is exceeded."""

    pass


class LLMClient:
    """LiteLLM-based LLM client with Zhipu GLM integration.

    This client provides a unified interface for chat completions,
    streaming, and tool calling using the LiteLLM library.

    Attributes:
        config: Configuration object containing API settings
        model: Model name in LiteLLM format (e.g., "zai/glm-4")
        api_key: API key for authentication
        base_url: Base URL for API (optional)
        temperature: Temperature for completions
        max_tokens: Maximum tokens for responses
        max_retries: Maximum number of retries on failure
        retry_delay: Initial delay between retries (exponential backoff)
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        config: Config,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        timeout: float = 60.0,
        usage_tracker=None,
        diagnostics: Any = None,
        model_tier: Optional[str] = None,
        budget_guard=None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            config: Configuration object with API settings
            max_retries: Maximum number of retries on failure
            retry_delay: Initial delay between retries in seconds
            timeout: Request timeout in seconds
            usage_tracker: Optional UsageTracker for recording API usage
            diagnostics: Optional diagnostics collector
            model_tier: Optional model tier ("fast", "power", or None for default).
                When set, chat() calls will use the corresponding tier model
                unless overridden per-call.
        """
        self.config = config
        self.api_key = config.active_api_key
        self.base_url = config.active_base_url
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens

        # Build model name in LiteLLM format via config property
        # e.g. "zai/glm-4" for GLM, "deepseek/deepseek-chat" for DeepSeek
        self.model = config.litellm_model

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._usage_tracker = usage_tracker
        self._diagnostics = diagnostics
        self._budget_guard = budget_guard

        # Optional PII desensitizer
        self._desensitizer = None
        _desensitize_enabled = os.environ.get("DESENSITIZE_ENABLED", "true").lower() in ("1", "true")
        if _desensitize_enabled:
            try:
                from src.security.desensitizer import Desensitizer
                self._desensitizer = Desensitizer()
                logger.debug("LLMClient: PII desensitizer enabled")
            except Exception as e:
                logger.debug("LLMClient: desensitizer init skipped: %s", e)

        # Model tier for tiered routing (None = use default model)
        self._model_tier = model_tier

        # Track provider failures to auto-disable tiers (e.g., expired API keys)
        self._disabled_tiers: Dict[str, float] = {}  # tier -> timestamp of last failure

        # Optional litellm Router for connection pooling and failover
        # Skip Router when running under pytest (test mocks patch direct completion())
        self._router = None
        _in_pytest = "PYTEST_CURRENT_TEST" in os.environ
        if not _in_pytest and os.getenv("LITELLM_ROUTER_ENABLED", "").lower() in ("1", "true"):
            try:
                from litellm import Router

                def _api_key_for_model(model: str) -> str:
                    """Get the correct API key for a model's provider."""
                    if model.startswith("deepseek/"):
                        return config.deepseek_api_key or config.glm_api_key
                    return self.api_key  # Default provider key

                def _base_url_for_model(model: str) -> str:
                    """Get the correct base URL for a model's provider."""
                    if model.startswith("deepseek/"):
                        return config.deepseek_base_url or "https://api.deepseek.com"
                    return self.base_url or ""

                router_model_list = [
                    {
                        "model_name": "default",
                        "litellm_params": {"model": self.model, "api_key": self.api_key},
                    },
                ]
                # Only add fast tier if it differs from default and has credentials
                fast_model = config.fast_model
                if fast_model != self.model:
                    fast_params: Dict[str, Any] = {"model": fast_model, "api_key": _api_key_for_model(fast_model)}
                    fast_base = _base_url_for_model(fast_model)
                    if fast_base:
                        fast_params["api_base"] = fast_base
                    router_model_list.append({"model_name": "fast", "litellm_params": fast_params})
                # Only add power tier if it differs from default and has credentials
                power_model = config.power_model
                if power_model != self.model:
                    power_params: Dict[str, Any] = {"model": power_model, "api_key": _api_key_for_model(power_model)}
                    power_base = _base_url_for_model(power_model)
                    if power_base:
                        power_params["api_base"] = power_base
                    router_model_list.append({"model_name": "power", "litellm_params": power_params})
                if self.base_url:
                    for entry in router_model_list:
                        if entry["model_name"] == "default":
                            entry["litellm_params"]["api_base"] = self.base_url

                self._router = Router(
                    model_list=router_model_list,
                    num_retries=2,
                    timeout=60,
                    allowed_fails=3,
                    cooldown_time=60,
                )
                logger.info("litellm Router enabled with %d model(s)", len(router_model_list))
            except Exception as exc:
                logger.warning("Failed to initialize litellm Router, falling back to direct calls: %s", exc)
                self._router = None

        # Configure LiteLLM global settings
        litellm.request_timeout = timeout
        litellm.num_retries = 0  # We handle retries manually

        # LLM result cache for deterministic calls (expand, sufficiency, verify)
        self._llm_cache = LLMResultCache(ttl=3600.0, max_size=128)

        logger.info(f"Initialized LLMClient with model: {self.model}")

    def _resolve_model(self, override_tier: Optional[str] = None) -> str:
        """Resolve which model to use based on tier config.

        Fallback chain:
            fast  → config.fast_model → glm-4-flash → default model
            power → config.power_model → default model

        Auto-disables tiers on authentication errors for 5 minutes.

        Args:
            override_tier: Per-call tier override ("fast", "power", or None).

        Returns:
            Model string in litellm format.
        """
        tier = override_tier or self._model_tier

        # Check if this tier was recently disabled (auth failure)
        if tier and tier in self._disabled_tiers:
            if time.time() - self._disabled_tiers[tier] < 300:  # 5 min cooldown
                logger.debug("Tier '%s' disabled due to auth failure, using default model", tier)
                return self.model
            del self._disabled_tiers[tier]  # Cooldown expired

        if tier == "fast":
            fast = self.config.fast_model
            if fast.startswith("deepseek/") and not self.config.deepseek_api_key:
                model_result = "zai/glm-4-flash" if self.config.glm_api_key else self.model
            elif fast.startswith("zai/") and not self.config.glm_api_key:
                model_result = self.model
            else:
                model_result = fast
        elif tier == "power":
            power = self.config.power_model
            if power.startswith("deepseek/") and not self.config.deepseek_api_key:
                model_result = self.model
            elif power.startswith("zai/") and not self.config.glm_api_key:
                model_result = self.model
            else:
                model_result = power
        else:
            model_result = self.model  # Default

        # P1: Validate model name format — warn early on misconfiguration
        if model_result and not any(
            model_result.startswith(prefix)
            for prefix in ("zai/", "deepseek/", "openai/")
        ):
            logger.warning(
                "Model '%s' has unrecognized provider prefix — "
                "may cause API errors. Expected format: 'zai/' or 'deepseek/' prefix",
                model_result,
            )

        return model_result

    def validate_tiered_models(self) -> list[str]:
        """Validate tiered model configuration. Returns warning messages.

        Call this after constructing LLMClient to catch misconfiguration early.
        """
        warnings = []
        for tier_name, model_attr in [("fast", "fast_model"), ("power", "power_model")]:
            model_val = getattr(self.config, model_attr, "")
            if model_val and not any(
                model_val.startswith(prefix)
                for prefix in ("zai/", "deepseek/", "openai/")
            ):
                warnings.append(
                    f"Tier '{tier_name}' model '{model_val}' lacks provider prefix "
                    f"(expected 'zai/' or 'deepseek/')"
                )
        return warnings

    def _mark_tier_failed(self, tier: Optional[str]) -> None:
        """Mark a model tier as failed (e.g., auth error) to disable it temporarily."""
        if tier:
            self._disabled_tiers[tier] = time.time()
            logger.warning("Tier '%s' marked as failed, falling back to default model for 5 minutes", tier)

    def _resolve_router_model_name(self, override_tier: Optional[str] = None) -> str:
        """Resolve the Router model_name for the given tier.

        Falls back to "default" if the tier model matches the default model.

        Args:
            override_tier: Per-call tier override ("fast", "power", or None).

        Returns:
            Router model_name string (e.g. "default", "fast", "power").
        """
        tier = override_tier or self._model_tier
        resolved_model = self._resolve_model(override_tier)
        # If the resolved model is the same as default, use "default" model_name
        if resolved_model == self.model:
            return "default"
        return tier or "default"

    def _convert_tools(
        self, tools: Optional[List[Tool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Tool objects to LiteLLM format.

        Args:
            tools: List of Tool objects to convert

        Returns:
            Optional[List[Dict[str, Any]]]: Tools in LiteLLM format or None
        """
        if tools is None:
            return None
        if len(tools) == 0:
            return []
        return [tool.to_openai_tool() for tool in tools]

    def _parse_response(self, response: Any) -> ChatResponse:
        """Parse LiteLLM response into ChatResponse.

        Args:
            response: Raw response from LiteLLM

        Returns:
            ChatResponse: Parsed response
        """
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [ToolCall.from_litellm(tc) for tc in message.tool_calls]

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=usage,
            model=response.model or self.model,
            raw_response=response,
        )

    def _handle_error(self, error: Exception, attempt: int) -> None:
        """Handle errors with appropriate exception mapping.

        Args:
            error: The original exception
            attempt: Current retry attempt number

        Raises:
            LLMClientError: Mapped exception
        """
        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str:
            raise RateLimitError(f"Rate limit exceeded: {error}")
        elif "auth" in error_str or "401" in error_str or "403" in error_str:
            raise AuthenticationError(f"Authentication failed: {error}")
        elif (
            "timeout" in error_str
            or "connection" in error_str
            or "network" in error_str
        ):
            raise NetworkError(f"Network error: {error}")
        else:
            raise LLMClientError(f"LLM API error: {error}")

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry.

        Args:
            error: The exception to check

        Returns:
            bool: True if should retry
        """
        error_str = str(error).lower()
        retryable_errors = [
            "rate limit",
            "429",
            "timeout",
            "connection",
            "network",
            "503",
            "502",
            "500",
        ]
        return any(err in error_str for err in retryable_errors)

    def chat(
        self,
        messages: Sequence[Union[ChatMessage, Dict[str, Any]]],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_tier: Optional[str] = None,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of chat messages
            tools: Optional list of tools available to the model
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model_tier: Optional per-call model tier override ("fast", "power").
                When set, uses the corresponding tier model for this call only.

        Returns:
            ChatResponse: The completion response

        Raises:
            LLMClientError: If the request fails after all retries
        """
        # ── PII Desensitization ──
        _des_mappings: List[Dict[str, str]] = []
        if self._desensitizer:
            _msgs = []
            for msg in messages:
                if isinstance(msg, ChatMessage) and msg.role in ("user", "system"):
                    dr = self._desensitizer.desensitize(msg.content)
                    _msgs.append(ChatMessage(role=msg.role, content=dr.masked_text))
                    _des_mappings.append(dr.mapping)
                else:
                    _msgs.append(msg)
            messages = _msgs

        # Convert messages to dict format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append(msg.to_dict())
            else:
                formatted_messages.append(msg)

        # Resolve model based on tier
        resolved_model = self._resolve_model(model_tier)

        # LLM result cache: only for deterministic (no-tools) calls
        # Caches expansion, sufficiency, verification results keyed on prompt+model+params
        effective_temp = temperature if temperature is not None else self.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if tools is None:
            cached = self._llm_cache.get(
                formatted_messages, resolved_model, effective_temp, effective_max_tokens
            )
            if cached is not None:
                logger.debug("LLM cache hit for deterministic call")
                return ChatResponse(
                    content=cached,
                    finish_reason="stop",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    latency_ms=0.0,
                    cache_hit=True,
                )

        if self._budget_guard:
            result = self._budget_guard.check_budget()
            if not result.is_within_budget:
                for alert in result.alerts:
                    if alert.should_block:
                        raise BudgetExceededError(
                            f"Budget '{alert.budget_name}' exceeded: "
                            f"{alert.current_spend_cents}/{alert.limit_cents} cents ({alert.period})"
                        )

        # Pick correct API key + base URL for the resolved model's provider
        resolved_api_key = self.api_key
        resolved_base_url = self.base_url
        if resolved_model.startswith("deepseek/"):
            if self.config.deepseek_api_key:
                resolved_api_key = self.config.deepseek_api_key
            if self.config.deepseek_base_url:
                resolved_base_url = self.config.deepseek_base_url
            elif not resolved_base_url:
                resolved_base_url = "https://api.deepseek.com"

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": formatted_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "api_key": resolved_api_key,
        }

        # Add base URL if configured
        if resolved_base_url:
            kwargs["api_base"] = resolved_base_url

        # Add tools if provided
        litellm_tools = self._convert_tools(tools)
        if litellm_tools:
            kwargs["tools"] = litellm_tools
            kwargs["tool_choice"] = "auto"

        # Retry logic with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Sending chat request (attempt {attempt + 1}/{self.max_retries})"
                )
                call_start = time.time()

                # Use Router if available, else direct completion
                if self._router is not None:
                    router_name = self._resolve_router_model_name(model_tier)
                    router_kwargs: Dict[str, Any] = {
                        "model": router_name,
                        "messages": formatted_messages,
                        "temperature": kwargs["temperature"],
                        "max_tokens": kwargs["max_tokens"],
                    }
                    # Do NOT override api_base here — Router's model_list
                    # already has the correct api_base per provider (set at
                    # __init__).  Overriding with self.base_url would send
                    # DeepSeek requests to the GLM endpoint (401).
                    if litellm_tools:
                        router_kwargs["tools"] = litellm_tools
                        router_kwargs["tool_choice"] = "auto"
                    response = self._router.completion(**router_kwargs)
                else:
                    response = completion(**kwargs)

                latency_ms = (time.time() - call_start) * 1000
                parsed = self._parse_response(response)
                parsed.latency_ms = latency_ms
                # Track usage via UsageTracker if available
                if self._usage_tracker and parsed.usage:
                    self._usage_tracker.record_llm(
                        model=parsed.model or resolved_model,
                        input_tokens=parsed.usage.get("prompt_tokens", 0),
                        output_tokens=parsed.usage.get("completion_tokens", 0),
                        total_tokens=parsed.usage.get("total_tokens", 0),
                        request_meta={"temperature": kwargs.get("temperature")},
                    )
                # Cache deterministic (no-tools) results for reuse
                if tools is None and parsed.content:
                    self._llm_cache.put(
                        formatted_messages, resolved_model,
                        effective_temp, effective_max_tokens,
                        parsed.content,
                    )
                # ── PII Desensitization: restore ──
                if _des_mappings and parsed.content:
                    for mapping in reversed(_des_mappings):
                        if mapping:
                            parsed.content = self._desensitizer.restore(
                                parsed.content, mapping
                            )
                return parsed

            except Exception as e:
                last_error = e
                logger.warning(f"Chat request failed (attempt {attempt + 1}): {e}")

                # Auto-disable tier on auth errors (expired key, invalid credentials)
                if isinstance(e, litellm.AuthenticationError):
                    self._mark_tier_failed(model_tier)

                if not self._should_retry(e) or attempt == self.max_retries - 1:
                    self._handle_error(e, attempt)

                # Exponential backoff
                delay = self.retry_delay * (2**attempt)
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

        # Should not reach here, but handle it
        raise LLMClientError(f"Max retries exceeded: {last_error}")

    async def stream_chat(
        self,
        messages: Sequence[Union[ChatMessage, Dict[str, Any]]],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        This is an async generator that yields response chunks.

        Args:
            messages: List of chat messages
            tools: Optional list of tools available to the model
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Yields:
            str: Response content chunks

        Raises:
            LLMClientError: If the request fails after all retries
        """
        # Convert messages to dict format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append(msg.to_dict())
            else:
                formatted_messages.append(msg)

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "api_key": self.api_key,
            "stream": True,
        }

        # Add base URL if configured
        if self.base_url:
            kwargs["api_base"] = self.base_url

        # Add tools if provided
        litellm_tools = self._convert_tools(tools)
        if litellm_tools:
            kwargs["tools"] = litellm_tools
            kwargs["tool_choice"] = "auto"

        # Retry logic with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Sending stream chat request (attempt {attempt + 1}/{self.max_retries})"
                )
                response = await acompletion(**kwargs)

                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

                return  # Successfully completed streaming

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Stream chat request failed (attempt {attempt + 1}): {e}"
                )

                if not self._should_retry(e) or attempt == self.max_retries - 1:
                    self._handle_error(e, attempt)

                # Exponential backoff
                delay = self.retry_delay * (2**attempt)
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        # Should not reach here, but handle it
        raise LLMClientError(f"Max retries exceeded: {last_error}")

    def execute_tool_calls(
        self,
        tool_calls: List[ToolCall],
        tools: List[Tool],
    ) -> List[ToolResult]:
        """Execute tool calls from LLM response.

        Args:
            tool_calls: List of tool calls to execute
            tools: List of available tools

        Returns:
            List[ToolResult]: Results of tool executions
        """
        results = []
        tool_map = {tool.name: tool for tool in tools}

        for tc in tool_calls:
            if tc.name not in tool_map:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content="",
                        success=False,
                        error=f"Unknown tool: {tc.name}",
                    )
                )
                continue

            tool = tool_map[tc.name]

            try:
                logger.info(f"Executing tool: {tc.name} with args: {tc.arguments}")

                # Check dedup cache
                cached_content: Any | None = None
                if hasattr(self, "_tool_call_cache") and self._tool_call_cache is not None:
                    cached_content = self._tool_call_cache.get(tc.name, tc.arguments)
                if cached_content is not None:
                    results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=cached_content,
                            success=True,
                        )
                    )
                    logger.info(f"Tool {tc.name} deduplicated (cache hit)")
                    if self._diagnostics and self._diagnostics.is_active:
                        self._diagnostics.record_tool_call(cache_hit=True)
                    continue

                result = tool.execute(**tc.arguments)

                # Convert result to string for LLM consumption
                # Handle base.ToolResult objects which have .data attribute
                if isinstance(result, str):
                    content = result
                elif isinstance(result, dict):
                    content = json.dumps(result, ensure_ascii=False)
                elif hasattr(result, "data") and result.data is not None:
                    # base.ToolResult — extract .data which is the actual content
                    if isinstance(result.data, str):
                        content = result.data
                    elif isinstance(result.data, dict):
                        content = json.dumps(result.data, ensure_ascii=False)
                    else:
                        content = str(result.data)
                else:
                    content = str(result)

                # Store in dedup cache
                if hasattr(self, "_tool_call_cache") and self._tool_call_cache is not None:
                    self._tool_call_cache.put(tc.name, tc.arguments, content)

                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=content,
                        success=True,
                        metadata=getattr(result, "metadata", None),
                    )
                )
                logger.info(f"Tool {tc.name} executed successfully")
                if self._diagnostics and self._diagnostics.is_active:
                    self._diagnostics.record_tool_call(cache_hit=False)

            except Exception as e:
                logger.error(f"Tool {tc.name} execution failed: {e}")
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content="",
                        success=False,
                        error=str(e),
                    )
                )

        return results

    def _summarize_history(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize old conversation messages using LLM.

        Args:
            messages: List of conversation messages to summarize

        Returns:
            str: A brief summary of the conversation history
        """
        history_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")[:300]
            if role in ("user", "assistant"):
                history_text += f"{role}: {content}\n"
            elif role == "tool":
                history_text += f"tool_result: {content[:100]}...\n"

        if not history_text:
            return ""

        try:
            summary_messages: List[Union[ChatMessage, Dict[str, Any]]] = [
                ChatMessage(
                    role="system",
                    content="用一句话总结以下对话历史的关键信息。",
                ),
                ChatMessage(role="user", content=history_text[:2000]),
            ]
            resp = self.chat(
                messages=summary_messages, temperature=0.1, max_tokens=200
            )
            return resp.content.strip()
        except Exception as exc:
            logger.warning(f"Failed to summarize history: {exc}")
            return history_text[:500]

    def _compress_conversation(
        self,
        conversation: List[Dict[str, Any]],
        level: str = "level3",
    ) -> List[Dict[str, Any]]:
        """Apply context management to keep conversation within limits.

        Levels (matching DCI-Agent-Lite):
        - "level0": No compression (baseline)
        - "level1": Truncate large tool results (>2000 chars)
        - "level2": Truncate + compact old tool results (keep last 3)
        - "level3": Truncate + compact + summarize old results (recommended)
        - "level4": level3 + summarize conversation history older than last 10 turns
        - "level5": Most aggressive — truncate everything to 500 chars, summarize all

        Args:
            conversation: The conversation message list (mutated in-place)
            level: Compression level string

        Returns:
            List[Dict[str, Any]]: The (possibly modified) conversation
        """
        if level == "level0":
            return conversation

        # --- Level 1+: Truncate large tool results ---
        truncation_threshold = 2000
        truncation_limit = 1000
        if level == "level5":
            truncation_threshold = 500
            truncation_limit = 250

        for msg in conversation:
            if msg.get("role") == "tool" and len(msg.get("content", "")) > truncation_threshold:
                msg["content"] = (
                    msg["content"][:truncation_limit]
                    + "\n...[truncated, use read tool for full content]"
                )

        if level == "level1":
            return conversation

        # --- Level 2+: Compact old tool results ---
        keep_last = 3 if level != "level5" else 2
        tool_result_indices = [
            i for i, m in enumerate(conversation) if m.get("role") == "tool"
        ]

        if level == "level2":
            for idx in tool_result_indices[:-keep_last]:
                orig_len = len(conversation[idx].get("content", ""))
                conversation[idx]["content"] = (
                    f"[tool result cleared - {orig_len} chars, use tools to re-fetch]"
                )
            return conversation

        # --- Level 3+: Smart summary of compacted results ---
        for idx in tool_result_indices[:-keep_last]:
            content = conversation[idx].get("content", "")
            summary = content[:200].replace("\n", " ").strip()
            conversation[idx]["content"] = f"[历史工具结果摘要: {summary}...]"

        if level == "level3":
            return conversation

        # --- Level 4+: Summarize old conversation turns ---
        if level in ("level4", "level5") and len(conversation) > 20:
            keep_recent = 10 if level == "level4" else 5
            old_msgs = conversation[:-keep_recent]
            recent_msgs = conversation[-keep_recent:]
            summary_text = self._summarize_history(old_msgs)
            conversation[:] = [
                {
                    "role": "system",
                    "content": f"Earlier conversation summary: {summary_text}",
                },
            ] + recent_msgs

        return conversation

    def chat_with_tools(
        self,
        messages: Sequence[Union[ChatMessage, Dict[str, Any]]],
        tools: List[Tool],
        max_iterations: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        context_management: str = "level3",
        on_tool_call: Optional[Callable] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> ChatResponse:
        """Send a chat request with automatic tool execution.

        This method automatically executes tool calls and continues
        the conversation until a final response is received or
        max_iterations is reached.

        Args:
            messages: List of chat messages
            tools: List of available tools
            max_iterations: Maximum tool call iterations
            temperature: Override default temperature
            max_tokens: Override default max tokens
            max_total_tokens: Optional cumulative token budget across all
                iterations.  When exceeded a forced-answer prompt is injected.
            context_management: Compression level for context management
                ("level0"–"level5"). Defaults to "level3".
            on_tool_call: Optional callback invoked as
                ``on_tool_call(tool_call, tool_result)`` after each tool
                execution for observability / logging.
            should_stop: Optional callable checked after each tool execution.
                Return True to break the loop and force a final answer.

        Returns:
            ChatResponse: The final response
        """
        # Convert messages to dict format for mutation
        conversation: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                conversation.append(msg.to_dict())
            else:
                conversation.append(msg)

        # Initialize response to satisfy type checker
        response = ChatResponse(content="", finish_reason="stop")
        total_tokens_used = 0

        # Fresh dedup cache for this session
        self._tool_call_cache = _ToolCallCache()

        for iteration in range(max_iterations):
            response = self.chat(
                messages=conversation,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Track cumulative token usage
            total_tokens_used += response.usage.get("total_tokens", 0)

            # Check token budget
            if max_total_tokens and total_tokens_used >= max_total_tokens:
                logger.warning(
                    f"Token budget exceeded: {total_tokens_used} >= {max_total_tokens}"
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": "Token预算即将用完，请基于已收集的信息立即回答。",
                    }
                )
                response = self.chat(
                    messages=conversation,
                    tools=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response

            if not response.has_tool_calls:
                return response

            # Add assistant message with tool calls
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments,
                        },
                    }
                    for tc in response.tool_calls
                ]
            conversation.append(assistant_message)

            # Execute tools and add results
            tool_results = self.execute_tool_calls(response.tool_calls, tools)

            # Invoke observability callback for each tool call / result pair
            if on_tool_call is not None:
                for tc, result in zip(response.tool_calls, tool_results):
                    try:
                        on_tool_call(tc, result)
                    except Exception as cb_exc:
                        logger.warning(f"on_tool_call callback raised: {cb_exc}")

            for result in tool_results:
                conversation.append(result.to_message().to_dict())

            # Apply context management (graduated compression)
            self._compress_conversation(conversation, level=context_management)

            # P0 ReAct compliance check: if LLM made tool calls without reasoning, nudge it
            if response.content and len(response.content.strip()) < 20:
                logger.info(
                    f"ReAct nudge: iteration {iteration + 1} tool call with thin reasoning "
                    f"({len(response.content.strip())} chars)"
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "请在下一步操作前简要说明：已知什么信息、还缺什么、计划怎么查。"
                        ),
                    }
                )
            elif not response.content:
                conversation.append(
                    {
                        "role": "user",
                        "content": "请在下一步操作前简要说明你的推理过程。",
                    }
                )

            logger.info(f"Completed tool iteration {iteration + 1}/{max_iterations}")

            # Early-stop: check if convergence or external signal says stop
            if should_stop is not None:
                try:
                    if should_stop():
                        logger.info(
                            f"Early-stop triggered after iteration {iteration + 1}, "
                            "forcing final answer"
                        )
                        conversation.append(
                            {
                                "role": "user",
                                "content": "搜索结果已收敛，请基于以上收集的信息立即回答用户的问题。",
                            }
                        )
                        response = self.chat(
                            messages=conversation,
                            tools=None,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        return response
                except Exception as stop_exc:
                    logger.warning(f"should_stop callback raised: {stop_exc}")

        # Max iterations reached — if last response was a tool call, force a final answer
        if response.has_tool_calls:
            logger.info(
                f"Max iterations ({max_iterations}) reached with pending tool calls, "
                "requesting final answer from LLM"
            )
            # Add the pending assistant message + tool results so context is complete
            assistant_message = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments,
                        },
                    }
                    for tc in response.tool_calls
                ]
            conversation.append(assistant_message)

            tool_results = self.execute_tool_calls(response.tool_calls, tools)
            if on_tool_call is not None:
                for tc, result in zip(response.tool_calls, tool_results):
                    try:
                        on_tool_call(tc, result)
                    except Exception as cb_exc:
                        logger.warning(f"on_tool_call callback raised: {cb_exc}")
            for result in tool_results:
                conversation.append(result.to_message().to_dict())

            conversation.append(
                {
                    "role": "user",
                    "content": "已达到最大搜索轮次，请基于以上收集的信息立即回答用户的问题。",
                }
            )
            response = self.chat(
                messages=conversation,
                tools=None,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return response


def _smart_truncate(text: str, limit: int) -> str:
    """Truncate text at the nearest sentence boundary before *limit*.

    Falls back to a hard cut when no boundary is found within the
    acceptable window (70%-100% of limit).
    """
    if len(text) <= limit:
        return text

    window = text[:limit]
    # Prefer the last sentence-ending punctuation / newline in the window
    best = -1
    for sep in ("。", "！", "？", "；", ".", "!", "?", ";", "\n"):
        pos = window.rfind(sep)
        if pos > best:
            best = pos

    # Only use the boundary if it doesn't waste too much space
    threshold = int(limit * 0.7)
    if best >= threshold:
        return text[: best + 1]

    # No good boundary — hard cut
    return window
