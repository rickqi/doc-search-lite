"""
API Key & Token authentication for doc-search Web API.

Supports three modes (auto-detected):
  1. Legacy mode:    WEB_API_KEY in .env → single shared key (Bearer / X-API-Key)
  2. Token mode:     tokens.json exists → multi-token with scopes/expiry
  3. Open mode:      neither configured → no auth (development only)

Token format:
  {
    "tokens": {
      "sk-xxx": {
        "id": "sk-xxx",
        "name": "admin",
        "key": "sk-xxx-hash",
        "scopes": ["*"],
        "created": "2026-06-16T00:00:00",
        "expires_at": null
      }
    }
  }

Scopes:
  "*"      — all endpoints
  "search" — /query, /api/search/*
  "agent"  — /query/agent, /api/sessions/*
  "admin"  — /api/admin/*
  "read"   — /document/*
"""

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)

DEFAULT_TOKEN_FILE = Path(__file__).resolve().parent.parent.parent / "tokens.json"

_auth_log_db: Optional[Any] = None
_auth_log_db_path: Optional[str] = None


def set_auth_log_db_path(path: Optional[str]) -> None:
    global _auth_log_db_path, _auth_log_db
    _auth_log_db_path = path
    _auth_log_db = None


def _get_auth_log_db() -> Optional[Any]:
    global _auth_log_db
    if _auth_log_db is not None:
        return _auth_log_db
    db_path = os.environ.get("AUTH_LOG_DB_PATH", "") or (_auth_log_db_path or "")
    if not db_path:
        return None
    try:
        from src.storage.convert_db import ConvertDB
        p = Path(db_path)
        if not p.exists():
            return None
        _auth_log_db = ConvertDB(p).open()
        return _auth_log_db
    except Exception:
        logger.debug("Failed to open auth log DB", exc_info=True)
        return None


def _log_auth_request(
    endpoint: str,
    method: str,
    token_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    status_code: int = 200,
) -> None:
    db = _get_auth_log_db()
    if db is None:
        return
    db.record_auth_log(
        endpoint=endpoint,
        method=method,
        token_id=token_id,
        client_ip=client_ip,
        status_code=status_code,
    )


# ═══════════════════════════════════════════════════════════════
# Token storage
# ═══════════════════════════════════════════════════════════════

@dataclass
class Token:
    id: str
    name: str
    key: str
    scopes: list[str] = field(default_factory=lambda: ["*"])
    created: str = ""
    expires_at: Optional[str] = None

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.now(timezone.utc)

    def has_scope(self, required: str) -> bool:
        if "*" in self.scopes:
            return True
        return required in self.scopes

    def to_dict(self, include_key: bool = False) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "scopes": self.scopes,
            "created": self.created,
            "expires_at": self.expires_at,
        }
        if include_key:
            d["key"] = self.key
        return d


class TokenStore:
    """JSON file-backed token storage."""

    def __init__(self, path: Path = DEFAULT_TOKEN_FILE):
        self._path = path
        self._tokens: dict[str, Token] = {}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for tid, t in data.get("tokens", {}).items():
                self._tokens[tid] = Token(
                    id=t["id"], name=t["name"], key=t["key"],
                    scopes=t.get("scopes", ["*"]),
                    created=t.get("created", ""),
                    expires_at=t.get("expires_at"),
                )
        except Exception as e:
            logger.warning("Failed to load tokens.json: %s", e)

    def _save(self):
        data = {
            "tokens": {tid: t.to_dict(include_key=True) for tid, t in self._tokens.items()},
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def validate(self, key: str) -> Optional[Token]:
        for token in self._tokens.values():
            if token.key == key:
                if token.is_expired():
                    logger.warning("Token %s is expired", token.id)
                    return None
                return token
        return None

    def create(self, name: str, scopes: list[str] = None, expires_days: int = 0) -> Token:
        tid = f"sk-{secrets.token_hex(12)}"
        key = f"sk-{secrets.token_hex(24)}"
        expires = None
        if expires_days > 0:
            expires = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()
        token = Token(
            id=tid, name=name, key=key,
            scopes=scopes or ["*"],
            created=datetime.now(timezone.utc).isoformat(),
            expires_at=expires,
        )
        self._tokens[tid] = token
        self._save()
        return token

    def list(self) -> list[dict]:
        return [t.to_dict(include_key=False) for t in self._tokens.values()]

    def revoke(self, tid: str) -> bool:
        if tid in self._tokens:
            del self._tokens[tid]
            self._save()
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._tokens)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

def get_web_api_key() -> str:
    return os.environ.get("WEB_API_KEY", "").strip()

def get_dify_api_key() -> str:
    return os.environ.get("DIFY_API_KEY", "").strip()

def get_auth_mode() -> str:
    """Returns 'token', 'legacy', or 'open'."""
    if os.environ.get("PI_FORCE_AUTH") == "0":
        return "open"
    if DEFAULT_TOKEN_FILE.exists():
        store = TokenStore()
        if store.count > 0:
            return "token"
    if get_web_api_key():
        return "legacy"
    return "open"


# ═══════════════════════════════════════════════════════════════
# Middleware
# ═══════════════════════════════════════════════════════════════

class AuthMiddleware(BaseHTTPMiddleware):
    """Unified auth middleware supporting legacy key + multi-token."""

    def __init__(self, app, legacy_key: str = "", token_store: TokenStore = None):
        super().__init__(app)
        self._legacy_key = legacy_key
        if os.environ.get("PI_FORCE_AUTH") == "0":
            token_store = TokenStore()
            token_store._tokens = {}
        self._token_store = token_store or TokenStore()
        self._mode = "token" if self._token_store.count > 0 else ("legacy" if legacy_key else "open")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        if self._mode == "open":
            response = await call_next(request)
            self._try_log_auth(request, response.status_code)
            return response

        key = self._extract_key(request)
        if not key:
            return self._unauthorized("Missing API key")

        if self._mode == "legacy":
            if key == self._legacy_key:
                request.state.token_name = "legacy-admin"
                request.state.token_scopes = ["*"]
                response = await call_next(request)
                self._try_log_auth(request, response.status_code)
                return response
            return self._unauthorized("Invalid API key")

        if self._mode == "token":
            token = self._token_store.validate(key)
            if token:
                request.state.token_name = token.name
                request.state.token_scopes = token.scopes
                request.state.token_id = token.id

                scope = self._required_scope(path)
                if scope and not token.has_scope(scope):
                    return self._forbidden(f"Token lacks '{scope}' scope")
                response = await call_next(request)
                self._try_log_auth(request, response.status_code)
                return response
            return self._unauthorized("Invalid or expired token")

        response = await call_next(request)
        self._try_log_auth(request, response.status_code)
        return response

    def _try_log_auth(self, request: Request, status_code: int) -> None:
        try:
            _log_auth_request(
                endpoint=request.url.path,
                method=request.method,
                token_id=getattr(request.state, "token_id", None),
                client_ip=request.client.host if request.client else None,
                status_code=status_code,
            )
        except Exception:
            logger.debug("Auth log write failed", exc_info=True)

    def _extract_key(self, request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        key = request.headers.get("X-API-Key", "")
        if key:
            return key
        # SSE EventSource can't set headers — allow token via query param
        if request.url.path.endswith("/events"):
            return request.query_params.get("token", "")
        return ""

    def _required_scope(self, path: str) -> Optional[str]:
        if path.startswith("/api/admin/"):
            return "admin"
        if path.startswith("/query/agent") or path.startswith("/api/sessions"):
            return "agent"
        if path.startswith("/query") or path.startswith("/api/search"):
            return "search"
        if path.startswith("/document"):
            return "read"
        return None

    def _unauthorized(self, msg: str) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": msg})

    def _forbidden(self, msg: str) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": msg})


# ═══════════════════════════════════════════════════════════════
# FastAPI dependency for route-level scope checks
# ═══════════════════════════════════════════════════════════════

def require_scope(scope: str):
    """FastAPI dependency: require a specific scope on the request token."""
    def checker(request: Request):
        scopes = getattr(request.state, "token_scopes", ["*"])
        if "*" not in scopes and scope not in scopes:
            raise HTTPException(status_code=403, detail=f"Token requires '{scope}' scope")
    return checker
