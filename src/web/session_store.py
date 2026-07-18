"""
SQLite-backed session store for cross-process session persistence.

Replaces JSON file persistence in SessionManager with a SQLite database,
enabling multiple doc-search processes to share session state.
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    index_path   TEXT NOT NULL,
    raw_dir      TEXT,
    model        TEXT DEFAULT 'deepseek-v4-pro',
    prompt       TEXT DEFAULT '',
    messages     TEXT DEFAULT '[]',
    sources      TEXT DEFAULT '[]',
    created      REAL NOT NULL,
    last_active  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_last_active
    ON sessions(last_active);
"""


class SessionStore:
    """Thread-safe SQLite store for session persistence.

    Usage:
        store = SessionStore(Path("sessions.db"))
        store.save(session_id, data)
        sessions = store.list_all()
    """

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path("sessions.db")
        self._db_path = Path(db_path).resolve()
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Get a thread-safe SQLite connection (WAL mode)."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── CRUD ──────────────────────────────────────────────

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Insert or replace a session record."""
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                       (session_id, index_path, raw_dir, model, prompt,
                        messages, sources, created, last_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    data["index_path"],
                    data.get("raw_dir"),
                    data.get("model", "deepseek-v4-pro"),
                    data.get("prompt", ""),
                    json.dumps(data.get("messages", []), ensure_ascii=False),
                    json.dumps(data.get("sources", []), ensure_ascii=False),
                    data.get("created", time.time()),
                    data.get("last_active", time.time()),
                ),
            )
            conn.commit()

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a single session by ID."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def delete(self, session_id: str) -> bool:
        """Delete a session record. Returns True if it existed."""
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_all(self) -> list[dict[str, Any]]:
        """List all sessions ordered by last_active descending."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY last_active DESC"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        """Count total sessions."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            return row[0]

    def cleanup_expired(self, idle_timeout: float) -> int:
        """Remove sessions idle longer than idle_timeout seconds.

        Returns number of removed sessions.
        """
        cutoff = time.time() - idle_timeout
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE last_active < ?",
                (cutoff,),
            )
            conn.commit()
            removed = cursor.rowcount
            if removed:
                logger.info("SessionStore: cleaned up %d expired session(s)", removed)
            return removed

    def touch(self, session_id: str) -> bool:
        """Update last_active timestamp. Returns True if session exists."""
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def vacuum(self) -> None:
        """Reclaim space from deleted sessions."""
        with self._lock, self._get_conn() as conn:
            conn.execute("VACUUM")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a SQLite Row to a dict with parsed JSON fields."""
        return {
            "session_id": row["session_id"],
            "index_path": row["index_path"],
            "raw_dir": row["raw_dir"],
            "model": row["model"],
            "prompt": row["prompt"],
            "messages": json.loads(row["messages"]) if row["messages"] else [],
            "sources": json.loads(row["sources"]) if row["sources"] else [],
            "created": row["created"],
            "last_active": row["last_active"],
        }
