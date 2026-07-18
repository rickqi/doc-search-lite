"""Session Manager for doc-search web interface.

Manages the lifecycle of search sessions: creation, listing, deletion,
and idle timeout cleanup. Each session wraps a SearchAgent instance
and an asyncio queue for SSE event streaming.

Uses SQLite-backed SessionStore for cross-process persistence.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from src.web.session_store import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Active search session context.

    Holds the agent instance, SSE event queue, and metadata for one
    browser session.
    """

    session_id: str
    index_path: Path
    raw_dir: Optional[Path] = None
    model: str = "deepseek-v4-pro"

    # Session state
    prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # SSE event queue (populated by agent hooks, consumed by /events endpoint)
    event_queue: Optional[asyncio.Queue] = None

    # Abort signal
    abort_event: Optional[asyncio.Event] = None

    def touch(self) -> None:
        """Update last-active timestamp."""
        self.last_active = time.time()

    def add_message(self, role: str, content: str, srch_id: str = "") -> None:
        """Append a message to the history.

        Args:
            role: 'user' or 'assistant'
            content: Message text
            srch_id: SearchLogger session ID for cross-reference (optional)
        """
        msg = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if srch_id:
            msg["srch_session_id"] = srch_id
        self.messages.append(msg)
        self.touch()


class SessionManager:
    """Global session manager.

    Maintains an in-memory registry of active sessions with automatic
    idle timeout cleanup. Sessions are persisted via SQLite (SessionStore)
    for cross-process sharing.

    Thread-safe: uses a Lock for the session registry.
    """

    IDLE_TIMEOUT = 1800  # 30 minutes
    MAX_SESSIONS = 20    # Max concurrent sessions

    def __init__(self, storage_dir: Optional[Path] = None):
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = Lock()
        _dir = storage_dir or Path("sessions")
        _dir.mkdir(parents=True, exist_ok=True)
        self._store = SessionStore(_dir / "sessions.db")
        self._cleanup_task: Optional[asyncio.Task] = None

    # ── CRUD ──────────────────────────────────────────────

    def create(
        self,
        index_path: Path,
        raw_dir: Optional[Path] = None,
        model: str = "deepseek-v4-pro",
    ) -> SessionContext:
        """Create a new session.

        Args:
            index_path: Path to the search index.
            raw_dir: Optional raw markdown directory.
            model: Model name to use for this session.

        Returns:
            New SessionContext.

        Raises:
            RuntimeError: If max concurrent sessions reached.
        """
        with self._lock:
            if len(self._sessions) >= self.MAX_SESSIONS:
                self._evict_oldest()

            session_id = f"ses_{uuid.uuid4().hex[:12]}"
            ctx = SessionContext(
                session_id=session_id,
                index_path=index_path,
                raw_dir=raw_dir,
                model=model,
            )
            self._sessions[session_id] = ctx
            logger.info("Created session: %s (index=%s)", session_id, index_path)
            return ctx

    def get(self, session_id: str) -> Optional[SessionContext]:
        """Get an active session by ID."""
        with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx:
                ctx.touch()
            return ctx

    def get_or_create(
        self,
        session_id: str,
        index_path: Path,
        raw_dir: Optional[Path] = None,
        model: str = "deepseek-v4-pro",
    ) -> SessionContext:
        """Get an existing session or create a new one with the given ID.

        For API callers that provide their own session_id. Unlike create(),
        this does NOT auto-generate an ID — the caller controls the ID.
        """
        with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx is not None:
                ctx.touch()
                return ctx

            if len(self._sessions) >= self.MAX_SESSIONS:
                self._evict_oldest()

            ctx = SessionContext(
                session_id=session_id,
                index_path=index_path,
                raw_dir=raw_dir,
                model=model,
            )
            self._sessions[session_id] = ctx
            logger.info("Created API session: %s (index=%s)", session_id, index_path)
            return ctx

    def delete(self, session_id: str) -> bool:
        """Delete a session (abort if running).

        Returns:
            True if the session existed and was deleted.
        """
        with self._lock:
            ctx = self._sessions.pop(session_id, None)
            if ctx is None:
                return False

        # Signal abort if running
        if ctx.abort_event:
            ctx.abort_event.set()
        logger.info("Deleted session: %s", session_id)
        return True

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions (active + persisted from SQLite store).

        Active in-memory sessions take precedence over DB entries.
        """
        # Merge: active sessions in memory + persisted sessions from DB
        seen: set = set()
        result: List[Dict[str, Any]] = []

        with self._lock:
            for s in self._sessions.values():
                seen.add(s.session_id)
                result.append({
                    "id": s.session_id,
                    "prompt": s.prompt[:100] if s.prompt else "",
                    "messages_count": len(s.messages),
                    "sources_count": len(s.sources),
                    "model": s.model,
                    "created": s.created,
                    "last_active": s.last_active,
                })

        # Add persisted sessions not currently in memory
        for row in self._store.list_all():
            sid = row["session_id"]
            if sid not in seen:
                result.append({
                    "id": sid,
                    "prompt": (row.get("prompt") or "")[:100],
                    "messages_count": len(row.get("messages") or []),
                    "sources_count": len(row.get("sources") or []),
                    "model": row.get("model", ""),
                    "created": row.get("created", 0),
                    "last_active": row.get("last_active", 0),
                })

        return result

    def touch(self, session_id: str) -> bool:
        """Update last-active time for a session."""
        ctx = self.get(session_id)
        if ctx:
            ctx.touch()
            return True
        return False

    # ── Persistence ──────────────────────────────────────

    def save(self, ctx: SessionContext) -> None:
        """Persist session to SQLite store."""
        self._store.save(ctx.session_id, {
            "session_id": ctx.session_id,
            "index_path": str(ctx.index_path),
            "raw_dir": str(ctx.raw_dir) if ctx.raw_dir else None,
            "model": ctx.model,
            "prompt": ctx.prompt,
            "messages": ctx.messages,
            "sources": ctx.sources,
            "created": ctx.created,
            "last_active": ctx.last_active,
        })

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load persisted session data from SQLite store."""
        return self._store.load(session_id)

    def touch_persisted(self, session_id: str) -> bool:
        """Update last_active in the store (for sessions loaded from DB)."""
        return self._store.touch(session_id)

    # ── Cleanup ──────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Evict the least-recently-active session to make room."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_active)
        ctx = self._sessions.pop(oldest_id)
        if ctx.abort_event:
            ctx.abort_event.set()
        logger.info("Evicted oldest session: %s", oldest_id)

    def cleanup_expired(self) -> int:
        """Remove sessions idle longer than IDLE_TIMEOUT.

        Cleans both in-memory registry (for active sessions) and
        SQLite store (for persisted sessions).

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        total_removed = 0

        # Clean in-memory active sessions
        with self._lock:
            expired = [
                sid
                for sid, ctx in self._sessions.items()
                if now - ctx.last_active > self.IDLE_TIMEOUT
            ]
            for sid in expired:
                ctx = self._sessions.pop(sid)
                if ctx.abort_event:
                    ctx.abort_event.set()
                logger.info("Cleaned up expired session: %s", sid)
            total_removed += len(expired)

        # Clean persisted sessions in SQLite store
        db_removed = self._store.cleanup_expired(self.IDLE_TIMEOUT)
        total_removed += db_removed

        return total_removed

    async def start_cleanup_loop(self, interval: float = 300.0) -> None:
        """Start a background task that periodically cleans up expired sessions.

        Args:
            interval: Seconds between cleanup checks (default: 5 min).
        """
        async def _loop():
            while True:
                await asyncio.sleep(interval)
                removed = self.cleanup_expired()
                if removed:
                    logger.info("Cleanup removed %d expired session(s)", removed)

        self._cleanup_task = asyncio.create_task(_loop())

    async def stop_cleanup_loop(self) -> None:
        """Stop the cleanup background task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    @property
    def active_count(self) -> int:
        """Number of currently active sessions."""
        with self._lock:
            return len(self._sessions)


# ── Global singleton ─────────────────────────────────────

_default_manager: Optional[SessionManager] = None


def get_session_manager(storage_dir: Optional[Path] = None) -> SessionManager:
    """Get or create the global SessionManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SessionManager(storage_dir=storage_dir)
    return _default_manager
