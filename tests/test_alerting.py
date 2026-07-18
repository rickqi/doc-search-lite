"""Tests for AlertManager — real-time webhook alerting."""

import time
from unittest.mock import patch

from src.stats.alerting import AlertManager
from src.stats.budget_guard import BudgetAlert, BudgetCheckResult


class TestAlertManagerInit:
    def test_webhook_url_from_argument(self):
        am = AlertManager(webhook_url="https://hooks.example.com/alert")
        assert am.webhook_url == "https://hooks.example.com/alert"

    def test_webhook_url_from_env(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.env.test/notify")
        am = AlertManager()
        assert am.webhook_url == "https://hooks.env.test/notify"

    def test_webhook_url_empty_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        am = AlertManager()
        assert am.webhook_url == ""

    def test_default_thresholds(self):
        am = AlertManager(webhook_url="https://h.test")
        assert am.error_threshold == 3
        assert am.budget_warning_threshold == 0.8


class TestCheckLLMErrors:
    @patch("src.stats.alerting.httpx.post")
    def test_sends_alert_when_threshold_exceeded(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=3)
        result = am.check_llm_errors(5)
        assert result is True
        assert mock_post.call_count == 1
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["category"] == "llm_errors"
        assert payload["level"] == "critical"
        assert payload["details"]["recent_errors"] == 5

    @patch("src.stats.alerting.httpx.post")
    def test_sends_alert_at_exact_threshold(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=3)
        assert am.check_llm_errors(3) is True
        assert mock_post.call_count == 1

    @patch("src.stats.alerting.httpx.post")
    def test_no_alert_below_threshold(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=3)
        assert am.check_llm_errors(2) is False
        mock_post.assert_not_called()

    @patch("src.stats.alerting.httpx.post")
    def test_no_alert_when_no_webhook(self, mock_post):
        am = AlertManager(webhook_url="", error_threshold=1)
        assert am.check_llm_errors(10) is True
        mock_post.assert_not_called()


class TestCheckBudget:
    def _make_alert(self, **overrides) -> BudgetAlert:
        defaults = dict(
            budget_name="monthly",
            period="monthly",
            current_spend_cents=0,
            limit_cents=10000,
            usage_percent=0.0,
            is_exceeded=False,
            should_block=False,
        )
        defaults.update(overrides)
        return BudgetAlert(**defaults)

    @patch("src.stats.alerting.httpx.post")
    def test_sends_critical_on_budget_exceeded(self, mock_post):
        alert = self._make_alert(
            current_spend_cents=10000,
            usage_percent=100.0,
            is_exceeded=True,
        )
        result_budget = BudgetCheckResult(
            is_within_budget=False, alerts=[alert], total_spend_cents=10000
        )
        am = AlertManager(webhook_url="https://h.test")
        assert am.check_budget(result_budget) is True
        assert mock_post.call_count == 1
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["category"] == "budget"
        assert payload["level"] == "critical"
        assert payload["details"]["is_exceeded"] is True

    @patch("src.stats.alerting.httpx.post")
    def test_sends_warning_at_80_percent(self, mock_post):
        alert = self._make_alert(
            current_spend_cents=8000,
            usage_percent=80.0,
            is_exceeded=False,
        )
        result_budget = BudgetCheckResult(
            is_within_budget=True, alerts=[alert], total_spend_cents=8000
        )
        am = AlertManager(webhook_url="https://h.test", budget_warning_threshold=0.8)
        assert am.check_budget(result_budget) is True
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["level"] == "warning"
        assert payload["details"]["is_exceeded"] is False

    @patch("src.stats.alerting.httpx.post")
    def test_no_alert_below_warning_threshold(self, mock_post):
        alert = self._make_alert(
            current_spend_cents=5000,
            usage_percent=50.0,
            is_exceeded=False,
        )
        result_budget = BudgetCheckResult(
            is_within_budget=True, alerts=[alert], total_spend_cents=5000
        )
        am = AlertManager(webhook_url="https://h.test")
        assert am.check_budget(result_budget) is False
        mock_post.assert_not_called()

    @patch("src.stats.alerting.httpx.post")
    def test_multiple_alerts_one_critical_takes_precedence(self, mock_post):
        warning_alert = self._make_alert(
            budget_name="daily",
            current_spend_cents=800,
            limit_cents=1000,
            usage_percent=80.0,
        )
        critical_alert = self._make_alert(
            budget_name="monthly",
            current_spend_cents=10000,
            usage_percent=100.0,
            is_exceeded=True,
        )
        result_budget = BudgetCheckResult(
            is_within_budget=False,
            alerts=[warning_alert, critical_alert],
        )
        am = AlertManager(webhook_url="https://h.test")
        assert am.check_budget(result_budget) is True
        assert mock_post.call_count == 1
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["level"] == "critical"

    @patch("src.stats.alerting.httpx.post")
    def test_no_alert_when_no_alerts(self, mock_post):
        result_budget = BudgetCheckResult(
            is_within_budget=True, alerts=[], total_spend_cents=0
        )
        am = AlertManager(webhook_url="https://h.test")
        assert am.check_budget(result_budget) is False
        mock_post.assert_not_called()


class TestCheckIndexHealth:
    @patch("src.stats.alerting.httpx.post")
    def test_sends_alert_when_doc_count_zero(self, mock_post):
        am = AlertManager(webhook_url="https://h.test")
        assert am.check_index_health("/idx/path", 0) is True
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["category"] == "index_health"
        assert payload["level"] == "critical"
        assert payload["details"]["index_path"] == "/idx/path"

    @patch("src.stats.alerting.httpx.post")
    def test_no_alert_when_docs_exist(self, mock_post):
        am = AlertManager(webhook_url="https://h.test")
        assert am.check_index_health("/idx/path", 42) is False
        mock_post.assert_not_called()


class TestRateLimiting:
    @patch("src.stats.alerting.httpx.post")
    def test_same_category_not_sent_twice_within_5_min(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=3)
        assert am.check_llm_errors(5) is True
        assert am.check_llm_errors(10) is False
        assert mock_post.call_count == 1

    @patch("src.stats.alerting.httpx.post")
    def test_different_categories_both_sent(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=3)
        assert am.check_llm_errors(5) is True
        assert am.check_index_health("/idx", 0) is True
        assert mock_post.call_count == 2

    @patch("src.stats.alerting.httpx.post")
    def test_sent_again_after_rate_limit_window(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=3)
        assert am.check_llm_errors(5) is True
        with patch.object(am, "_last_sent", {"llm_errors": time.time() - 301}):
            assert am.check_llm_errors(5) is True
        assert mock_post.call_count == 2


class TestSendWebhookRobustness:
    @patch("src.stats.alerting.httpx.post", side_effect=Exception("network down"))
    def test_catches_exception_silently(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=1)
        am._send_webhook({"level": "critical", "message": "test"})

    @patch("src.stats.alerting.httpx.post", side_effect=ConnectionError("refused"))
    def test_catches_connection_error_silently(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=1)
        am._send_webhook({"level": "critical", "message": "test"})

    @patch("src.stats.alerting.httpx.post")
    def test_uses_correct_url_and_timeout(self, mock_post):
        am = AlertManager(webhook_url="https://hooks.test/incoming")
        am._send_webhook({"level": "warning"})
        args, kwargs = mock_post.call_args
        assert args[0] == "https://hooks.test/incoming"
        assert kwargs["timeout"] == 5.0


class TestPayloadFormat:
    @patch("src.stats.alerting.httpx.post")
    def test_payload_has_required_fields(self, mock_post):
        am = AlertManager(webhook_url="https://h.test", error_threshold=1)
        am.check_llm_errors(5)
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        for key in ("timestamp", "level", "category", "message", "details"):
            assert key in payload
        assert isinstance(payload["details"], dict)
        assert "T" in payload["timestamp"]
