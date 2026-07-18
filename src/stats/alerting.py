"""Real-time alerting via webhook notifications.

Sends alerts to a configurable webhook when error rates spike,
budgets are breached, or indexes become unhealthy.
All HTTP calls are fire-and-forget in daemon threads.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import httpx

from src.stats.budget_guard import BudgetCheckResult

logger = logging.getLogger(__name__)

_RATE_LIMIT_SECONDS = 300


class AlertManager:
    """Webhook-based alert manager with rate limiting."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        error_threshold: int = 3,
        budget_warning_threshold: float = 0.8,
    ):
        self.webhook_url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL", "")
        self.error_threshold = error_threshold
        self.budget_warning_threshold = budget_warning_threshold
        self._last_sent: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _is_rate_limited(self, category: str) -> bool:
        now = time.time()
        last = self._last_sent.get(category, 0)
        return (now - last) < _RATE_LIMIT_SECONDS

    def _mark_sent(self, category: str) -> None:
        self._last_sent[category] = time.time()

    def _send_webhook(self, payload: dict) -> None:
        if not self.webhook_url:
            return

        def _post():
            try:
                httpx.post(self.webhook_url, json=payload, timeout=5.0)
            except Exception:
                logger.debug("Alert webhook failed silently", exc_info=True)

        thread = threading.Thread(target=_post, daemon=True)
        thread.start()

    def _emit(
        self, category: str, level: str, message: str, details: dict
    ) -> bool:
        with self._lock:
            if self._is_rate_limited(category):
                return False
            self._mark_sent(category)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "category": category,
            "message": message,
            "details": details,
        }
        self._send_webhook(payload)
        return True

    def check_llm_errors(self, recent_errors: int) -> bool:
        if recent_errors < self.error_threshold:
            return False
        return self._emit(
            "llm_errors",
            "critical",
            f"LLM error spike: {recent_errors} recent errors "
            f"(threshold={self.error_threshold})",
            {"recent_errors": recent_errors, "threshold": self.error_threshold},
        )

    def check_budget(self, budget_result: BudgetCheckResult) -> bool:
        critical = None
        warning = None
        for alert in budget_result.alerts:
            if alert.is_exceeded:
                if critical is None:
                    critical = alert
            elif alert.usage_percent >= self.budget_warning_threshold * 100:
                if warning is None:
                    warning = alert

        target = critical or warning
        if target is None:
            return False

        level = "critical" if target.is_exceeded else "warning"
        verb = "exceeded" if target.is_exceeded else "warning"
        return self._emit(
            "budget",
            level,
            f"Budget '{target.budget_name}' {verb}: "
            f"{target.usage_percent:.1f}% used",
            {
                "budget_name": target.budget_name,
                "period": target.period,
                "usage_percent": target.usage_percent,
                "current_spend_cents": target.current_spend_cents,
                "limit_cents": target.limit_cents,
                "is_exceeded": target.is_exceeded,
            },
        )

    def check_index_health(self, index_path: str, doc_count: int) -> bool:
        if doc_count > 0:
            return False
        return self._emit(
            "index_health",
            "critical",
            f"Index '{index_path}' is unhealthy: 0 documents",
            {"index_path": index_path, "doc_count": doc_count},
        )
