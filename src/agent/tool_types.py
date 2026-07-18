"""Shared tool types — ToolResult and ToolCache.

Moved from agent/base.py to break the search→agent circular dependency.
Used by both agent tools (agent/tools/) and search module (search/hybrid.py).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, Optional


class ToolResult:
    """Helper class to wrap tool execution results.

    Provides a standardized way to return results from tools,
    including success/failure status and optional metadata.
    """

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success, "data": self.data}
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def ok(cls, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> "ToolResult":
        return cls(success=False, error=error, metadata=metadata)


class ToolCache:
    """TTL-based LRU cache for tool execution results.

    Caches ToolResult objects keyed by a deterministic hash of
    tool name + arguments. Entries expire after TTL seconds.
    Uses OrderedDict for O(1) LRU eviction.
    """

    def __init__(self, ttl: float = 60.0, max_size: int = 50) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[float, "ToolResult"]] = OrderedDict()

    def get(self, key: str) -> Optional["ToolResult"]:
        if key not in self._store:
            return None
        expires, result = self._store[key]
        if time.time() > expires:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return result

    def put(self, key: str, result: "ToolResult") -> None:
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = (time.time() + self._ttl, result)
        self._store.move_to_end(key)

    @staticmethod
    def make_key(tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Generate a deterministic cache key from tool name + arguments."""
        import hashlib, json
        raw = json.dumps({"name": tool_name, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def clear(self) -> None:
        self._store.clear()
