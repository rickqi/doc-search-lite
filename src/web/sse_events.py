"""SSE event types and serialization for Agent event streaming.

Defines the standard event types that flow from the SearchAgent to the
browser via Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any


class AgentEventType(str, Enum):
    """Standardized Agent lifecycle events for SSE streaming."""

    SESSION_START = "session_start"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SEARCH_RESULT = "search_result"
    DOCUMENT_READ = "document_read"
    ANSWER_CHUNK = "answer_chunk"
    ANSWER_COMPLETE = "answer_complete"
    ERROR = "error"
    ABORTED = "aborted"
    STRATEGY_INFO = "strategy_info"
    SUFFICIENCY_CHECK = "sufficiency_check"
    HEARTBEAT = "heartbeat"


def sse_encode(event_type: AgentEventType, data: dict[str, Any]) -> str:
    """Encode a single SSE event frame.

    Format (per SSE spec):
        event: <event_type>\\n
        data: <json_data>\\n\\n

    Args:
        event_type: The event type from AgentEventType enum.
        data: Dictionary payload to serialize as JSON.

    Returns:
        SSE-formatted string ready to write to the response stream.
    """
    json_str = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_type.value}\ndata: {json_str}\n\n"


def heartbeat() -> str:
    """Generate an SSE heartbeat comment frame.

    Sent every 30s to prevent proxies/load balancers from closing
    idle connections.

    Returns:
        SSE comment frame string.
    """
    return ":\n\n"


def make_event(
    event_type: AgentEventType,
    **kwargs: Any,
) -> str:
    """Convenience helper to create an SSE frame with timestamp.

    Args:
        event_type: Event type.
        **kwargs: Data fields to include in the payload.

    Returns:
        SSE-formatted string.
    """
    data: dict[str, Any] = dict(kwargs)
    data.setdefault("timestamp", time.time())
    return sse_encode(event_type, data)
