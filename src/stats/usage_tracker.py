"""Unified API usage tracker for doc-search."""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BudgetStatus:
    """Budget check result."""

    name: str
    period: str
    limit_cents: int
    used_cents: int
    usage_ratio: float
    is_exceeded: bool
    is_alert: bool


class UsageTracker:
    """Unified API usage tracker.

    Records OCR, LLM Chat, and Rerank API calls to SQLite via ConvertDB.
    Calculates costs based on the pricing table.

    Usage is OPTIONAL — if no tracker is provided, components work normally.
    """

    def __init__(self, db: Any, source_dir: str = None):
        """Initialize tracker.

        Args:
            db: ConvertDB instance (must be open)
            source_dir: Source directory name for grouping
        """
        self._db = db
        self._source_dir = source_dir
        self._session_id: str | None = None

    def start_session(self, query: str = None, mode: str = None) -> str:
        """Start a new tracking session. Returns session_id."""
        self._session_id = str(uuid.uuid4())[:8]
        return self._session_id

    @property
    def session_id(self) -> str | None:
        """Current session ID, or None if no session started."""
        return self._session_id

    def record(
        self,
        call_type: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        file_id: int = None,
        request_meta: dict = None,
    ) -> int:
        """Record a single API call with automatic cost calculation."""
        cost = self._db.calculate_cost(model, input_tokens, output_tokens)
        meta_json = (
            json.dumps(request_meta, ensure_ascii=False) if request_meta else None
        )

        return self._db.add_token_usage_extended(
            call_type=call_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_millicents=cost,
            file_id=file_id,
            source_dir=self._source_dir,
            session_id=self._session_id,
            request_meta=meta_json,
        )

    def record_llm(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        request_meta: dict = None,
    ) -> int:
        """Record an LLM chat call."""
        return self.record(
            call_type="llm_chat",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            request_meta=request_meta,
        )

    def record_ocr(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        file_id: int = None,
    ) -> int:
        """Record an OCR call."""
        return self.record(
            call_type="ocr",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            file_id=file_id,
        )

    def record_rerank(
        self,
        model: str = "rerank",
        input_tokens: int = 0,
        total_tokens: int = 0,
    ) -> int:
        """Record a Rerank API call."""
        return self.record(
            call_type="rerank",
            model=model,
            input_tokens=input_tokens,
            total_tokens=total_tokens,
        )

    def get_summary(self, days: int = None) -> dict:
        """Get usage summary."""
        return self._db.get_token_usage_summary(
            source_dir=self._source_dir, days=days
        )

    def get_daily(self, days: int = 30) -> list:
        """Get daily usage trend."""
        return self._db.get_token_usage_daily(days=days, source_dir=self._source_dir)

    def get_by_model(self, days: int = None) -> list:
        """Get usage by model."""
        return self._db.get_token_usage_by_model(
            source_dir=self._source_dir, days=days
        )
