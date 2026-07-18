"""Tests for BudgetGuard — budget monitoring and enforcement."""

from pathlib import Path

from src.stats.budget_guard import BudgetCheckResult, BudgetGuard
from src.storage.convert_db import ConvertDB

# ── Helpers ──────────────────────────────────────────────────────────


def _open_db(tmp_path: Path) -> ConvertDB:
    """Create and open a fresh ConvertDB in tmp_path."""
    db = ConvertDB(tmp_path / "test.db")
    db.open()
    return db


# ════════════════════════════════════════════════════════════════════════
# BudgetGuard Tests
# ════════════════════════════════════════════════════════════════════════


class TestBudgetGuard:
    """Test BudgetGuard class."""

    def _make_guard(self, tmp_path):
        """Create a BudgetGuard with a fresh DB."""
        db = _open_db(tmp_path)
        guard = BudgetGuard(db)
        return guard, db

    def test_set_budget(self, tmp_path):
        """set_budget should create a new budget and return its ID."""
        guard, db = self._make_guard(tmp_path)
        budget_id = guard.set_budget(
            name="monthly-limit",
            limit_cents=10000,
            period="monthly",
            alert_threshold=0.8,
            block_exceed=False,
        )
        assert budget_id > 0

        budgets = guard.get_budgets()
        assert len(budgets) == 1
        assert budgets[0]["name"] == "monthly-limit"
        assert budgets[0]["limit_cents"] == 10000
        assert budgets[0]["period"] == "monthly"
        db.close()

    def test_set_budget_updates_existing(self, tmp_path):
        """set_budget with same name should update the existing budget."""
        guard, db = self._make_guard(tmp_path)
        id1 = guard.set_budget(name="test", limit_cents=5000)
        id2 = guard.set_budget(name="test", limit_cents=10000)
        assert id1 == id2

        budgets = guard.get_budgets()
        assert len(budgets) == 1
        assert budgets[0]["limit_cents"] == 10000
        db.close()

    def test_check_budget_within_limit(self, tmp_path):
        """check_budget should return is_within_budget=True when spend is below limit."""
        guard, db = self._make_guard(tmp_path)

        # Set a budget with 10000 cents = 100 yuan
        guard.set_budget(name="monthly", limit_cents=10000, period="monthly")

        # Add a small amount of usage (350 millicents = 0.35 cents)
        db.add_token_usage_extended(
            "ocr", "glm-ocr", input_tokens=100, cost_millicents=350,
        )

        result = guard.check_budget()
        assert isinstance(result, BudgetCheckResult)
        assert result.is_within_budget is True
        assert len(result.alerts) == 1
        assert result.alerts[0].is_exceeded is False
        assert result.alerts[0].usage_percent < 100
        db.close()

    def test_check_budget_exceeded(self, tmp_path):
        """check_budget should detect when spend exceeds the limit."""
        guard, db = self._make_guard(tmp_path)

        # Set a very low budget: 1 cent = 1000 millicents
        guard.set_budget(name="tiny", limit_cents=1, period="monthly")

        # Add usage that exceeds: 2000 millicents = 2 cents
        db.add_token_usage_extended(
            "ocr", "glm-ocr", input_tokens=1000, cost_millicents=2000,
        )

        result = guard.check_budget()
        assert result.is_within_budget is False
        assert len(result.alerts) == 1
        assert result.alerts[0].is_exceeded is True
        assert result.alerts[0].budget_name == "tiny"
        db.close()

    def test_check_budget_alert_threshold(self, tmp_path):
        """check_budget should show high usage percent when approaching threshold."""
        guard, db = self._make_guard(tmp_path)

        # Budget: 100 cents with 80% alert threshold
        guard.set_budget(
            name="alert-test", limit_cents=100,
            period="monthly", alert_threshold=0.8,
        )

        # Spend 85 cents = 85000 millicents (85% usage)
        db.add_token_usage_extended(
            "ocr", "glm-ocr", input_tokens=10000, cost_millicents=85000,
        )

        result = guard.check_budget()
        alert = result.alerts[0]
        assert alert.usage_percent >= 80.0
        assert alert.is_exceeded is False  # Not exceeded yet, but above threshold
        db.close()

    def test_get_budgets(self, tmp_path):
        """get_budgets should return all configured budgets."""
        guard, db = self._make_guard(tmp_path)
        guard.set_budget(name="budget-a", limit_cents=1000)
        guard.set_budget(name="budget-b", limit_cents=2000, period="daily")

        budgets = guard.get_budgets()
        assert len(budgets) == 2
        names = {b["name"] for b in budgets}
        assert "budget-a" in names
        assert "budget-b" in names
        db.close()

    def test_remove_budget(self, tmp_path):
        """remove_budget should delete the specified budget."""
        guard, db = self._make_guard(tmp_path)
        bid = guard.set_budget(name="to-remove", limit_cents=5000)

        assert len(guard.get_budgets()) == 1
        assert guard.remove_budget(bid) is True
        assert len(guard.get_budgets()) == 0
        db.close()

    def test_remove_budget_nonexistent(self, tmp_path):
        """remove_budget should return False for nonexistent ID."""
        guard, db = self._make_guard(tmp_path)
        assert guard.remove_budget(9999) is False
        db.close()

    def test_block_exceed_flag(self, tmp_path):
        """check_budget should set should_block when block_exceed=True and budget exceeded."""
        guard, db = self._make_guard(tmp_path)

        guard.set_budget(
            name="blocker", limit_cents=1, period="monthly", block_exceed=True,
        )

        # Exceed the budget
        db.add_token_usage_extended(
            "llm_chat", "zai/glm-4", input_tokens=500, cost_millicents=5000,
        )

        result = guard.check_budget()
        assert len(result.alerts) == 1
        assert result.alerts[0].is_exceeded is True
        assert result.alerts[0].should_block is True
        db.close()

    def test_block_not_set_when_not_exceeded(self, tmp_path):
        """should_block should be False when budget is not exceeded even with block_exceed=True."""
        guard, db = self._make_guard(tmp_path)

        guard.set_budget(
            name="blocker-safe", limit_cents=10000,
            period="monthly", block_exceed=True,
        )

        # Small usage
        db.add_token_usage_extended(
            "ocr", "glm-ocr", input_tokens=50, cost_millicents=25,
        )

        result = guard.check_budget()
        assert result.alerts[0].is_exceeded is False
        assert result.alerts[0].should_block is False
        db.close()

    def test_check_budget_no_budgets(self, tmp_path):
        """check_budget with no budgets should return within_budget=True."""
        guard, db = self._make_guard(tmp_path)
        result = guard.check_budget()
        assert result.is_within_budget is True
        assert len(result.alerts) == 0
        assert result.total_spend_cents == 0
        db.close()

    def test_check_budget_multiple_periods(self, tmp_path):
        """check_budget should handle multiple budgets with different periods."""
        guard, db = self._make_guard(tmp_path)

        guard.set_budget(name="daily-budget", limit_cents=100, period="daily")
        guard.set_budget(name="monthly-budget", limit_cents=10000, period="monthly")
        guard.set_budget(name="total-budget", limit_cents=100000, period="total")

        db.add_token_usage_extended(
            "ocr", "glm-ocr", input_tokens=500, cost_millicents=5000,
        )

        result = guard.check_budget()
        assert len(result.alerts) == 3
        periods = {a.period for a in result.alerts}
        assert periods == {"daily", "monthly", "total"}
        db.close()

    def test_check_budget_with_source_dir(self, tmp_path):
        """check_budget with source_dir should filter spending."""
        guard, db = self._make_guard(tmp_path)

        guard.set_budget(name="dir-budget", limit_cents=100, period="total")

        # Add usage for two different source dirs
        db.add_token_usage_extended(
            "ocr", "glm-ocr", input_tokens=100, cost_millicents=5000,
            source_dir="project-a",
        )
        db.add_token_usage_extended(
            "ocr", "glm-ocr", input_tokens=100, cost_millicents=1000,
            source_dir="project-b",
        )

        # Filter by project-a — should see only 5000 millicents = 5 cents
        result = guard.check_budget(source_dir="project-a")
        assert len(result.alerts) == 1
        # 5000 millicents // 1000 = 5 cents
        assert result.alerts[0].current_spend_cents == 5
        assert result.alerts[0].is_exceeded is False
        db.close()
