"""doc-search Web module — browser-based search interface.

Provides:
- SessionManager: session lifecycle (create/list/delete)
- SSE events: real-time Agent event streaming
- Static files: HTML/CSS/JS frontend
- AuthMiddleware: token-based API authentication
"""

from src.web.auth import AuthMiddleware, TokenStore, get_auth_mode, get_web_api_key
from src.web.session_manager import SessionContext, SessionManager
from src.web.sse_events import AgentEventType, sse_encode

__all__ = [
    "SessionManager",
    "SessionContext",
    "AgentEventType",
    "sse_encode",
    "AuthMiddleware",
    "TokenStore",
    "get_web_api_key",
    "get_auth_mode",
]
