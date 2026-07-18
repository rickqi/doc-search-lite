"""Budget monitoring and enforcement for API usage."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BudgetAlert:
    """Alert details for a single budget check."""

    budget_name: str
    period: str
    current_spend_cents: int
    limit_cents: int
    usage_percent: float
    is_exceeded: bool
    should_block: bool


@dataclass
class BudgetCheckResult:
    """Result of checking all active budgets."""

    is_within_budget: bool
    alerts: list[BudgetAlert] = field(default_factory=list)
    total_spend_cents: int = 0


class BudgetGuard:
    """Budget monitoring and enforcement.

    Checks current spending against configured budget limits
    in the budget table. Supports monthly, daily, and total periods.
    """

    def __init__(self, db: Any):
        """Initialize BudgetGuard.

        Args:
            db: ConvertDB instance (must be open)
        """
        self._db = db

    def set_budget(
        self,
        name: str,
        limit_cents: int,
        period: str = "monthly",
        alert_threshold: float = 0.8,
        block_exceed: bool = False,
    ) -> int:
        """Create or update a budget limit.

        If a budget with the same name exists, it is updated.
        Otherwise, a new budget is created.

        Args:
            name: Budget name (unique identifier)
            limit_cents: Budget limit in cents (1 cent = 1/100 yuan)
            period: Budget period - "monthly", "daily", or "total"
            alert_threshold: Fraction (0-1) at which to trigger alerts
            block_exceed: If True, budget exceedance should block further calls

        Returns:
            Budget row ID
        """
        existing = self._db.conn.execute(
            "SELECT id FROM budget WHERE name = ?", (name,)
        ).fetchone()

        if existing:
            self._db.conn.execute(
                """UPDATE budget
                   SET limit_cents = ?, period = ?, alert_threshold = ?,
                       block_exceed = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                (limit_cents, period, alert_threshold, int(block_exceed), existing["id"]),
            )
            self._db.conn.commit()
            return existing["id"]
        else:
            cursor = self._db.conn.execute(
                """INSERT INTO budget (name, limit_cents, period, alert_threshold, block_exceed)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, limit_cents, period, alert_threshold, int(block_exceed)),
            )
            self._db.conn.commit()
            return cursor.lastrowid

    def check_budget(self, source_dir: str = None) -> BudgetCheckResult:
        """Check current spending against all active budgets.

        Args:
            source_dir: Optional source directory filter for spending

        Returns:
            BudgetCheckResult with alerts for each budget
        """
        budgets = self._db.conn.execute(
            "SELECT * FROM budget"
        ).fetchall()

        if not budgets:
            return BudgetCheckResult(is_within_budget=True, total_spend_cents=0)

        # Get total spending in millicents
        conditions: list[str] = []
        params: list = []
        if source_dir is not None:
            conditions.append("source_dir = ?")
            params.append(source_dir)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        total_row = self._db.conn.execute(
            f"SELECT COALESCE(SUM(cost_millicents), 0) as total FROM token_usage {where}",
            params,
        ).fetchone()
        total_spend_millicents = total_row["total"] if total_row else 0
        total_spend_cents = total_spend_millicents // 1000

        # For period-specific queries
        period_conditions: list[str] = list(conditions)
        period_params: list = list(params)

        alerts: list[BudgetAlert] = []
        is_within = True

        for budget in budgets:
            b = dict(budget)
            period = b["period"]

            # Calculate spend for this budget's period
            if period == "daily":
                p_conditions = list(period_conditions) + [
                    "DATE(created_at) = DATE('now')"
                ]
                p_params = list(period_params)
            elif period == "monthly":
                p_conditions = list(period_conditions) + [
                    "created_at >= datetime('now', '-30 days')"
                ]
                p_params = list(period_params)
            else:  # total
                p_conditions = list(period_conditions)
                p_params = list(period_params)

            p_where = ""
            if p_conditions:
                p_where = "WHERE " + " AND ".join(p_conditions)

            row = self._db.conn.execute(
                f"SELECT COALESCE(SUM(cost_millicents), 0) as spend FROM token_usage {p_where}",
                p_params,
            ).fetchone()
            spend_millicents = row["spend"] if row else 0
            spend_cents = spend_millicents // 1000

            limit_cents = b["limit_cents"]
            threshold = b["alert_threshold"]
            block = bool(b["block_exceed"])

            usage_pct = (spend_cents / limit_cents * 100) if limit_cents > 0 else 0.0
            exceeded = spend_cents >= limit_cents if limit_cents > 0 else False

            if exceeded:
                is_within = False

            alert = BudgetAlert(
                budget_name=b["name"],
                period=period,
                current_spend_cents=spend_cents,
                limit_cents=limit_cents,
                usage_percent=round(usage_pct, 1),
                is_exceeded=exceeded,
                should_block=block and exceeded,
            )
            alerts.append(alert)

        return BudgetCheckResult(
            is_within_budget=is_within,
            alerts=alerts,
            total_spend_cents=total_spend_cents,
        )

    def get_budgets(self) -> list[dict]:
        """List all configured budgets.

        Returns:
            List of budget dicts with id, name, limit_cents, period,
            alert_threshold, block_exceed
        """
        rows = self._db.conn.execute(
            "SELECT * FROM budget ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def remove_budget(self, budget_id: int) -> bool:
        """Remove a budget by ID.

        Args:
            budget_id: Budget row ID

        Returns:
            True if a budget was removed, False if not found
        """
        cursor = self._db.conn.execute(
            "DELETE FROM budget WHERE id = ?", (budget_id,)
        )
        self._db.conn.commit()
        return cursor.rowcount > 0
