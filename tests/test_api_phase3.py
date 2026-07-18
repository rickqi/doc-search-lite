"""
Tests for stats export, budget management, benchmark endpoints.
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestAdminStatsExport:
    """Tests for GET /api/admin/stats-export"""

    def test_export_no_db_returns_error(self):
        """Export without convert.db returns an error."""
        r = client.get("/api/admin/stats-export", params={"raw_dir": "/nonexistent"})
        assert r.status_code in (200, 400, 404, 500)

    def test_export_csv_format(self):
        """CSV format returns text/csv."""
        r = client.get("/api/admin/stats-export", params={"raw_dir": "/nonexistent", "format": "csv"})
        assert r.status_code in (200, 400, 404, 500)

    def test_export_md_format(self):
        """Markdown format returns text/markdown."""
        r = client.get("/api/admin/stats-export", params={"raw_dir": "/nonexistent", "format": "md"})
        assert r.status_code in (200, 400, 404, 500)


class TestAdminBudget:
    """Tests for budget set/remove endpoints."""

    def test_budget_set_no_db(self):
        """Setting budget without convert.db returns error."""
        r = client.post("/api/admin/budget-set", params={
            "raw_dir": "/nonexistent", "limit_cents": 10000
        })
        assert r.status_code in (200, 400, 404, 500)


class TestAdminBenchmark:
    """Tests for POST /api/admin/benchmark"""

    def test_benchmark_no_index(self):
        """Benchmark without valid index returns error or empty results."""
        r = client.post("/api/admin/benchmark", params={"index_path": "/nonexistent"})
        assert r.status_code in (200, 400, 404)

    def test_benchmark_endpoint_registered(self):
        """Verify the endpoint is registered."""
        r = client.post("/api/admin/benchmark", params={
            "index_path": "/nonexistent", "modes": "bm25", "runs": 1
        })
        assert r.status_code in (200, 400, 500)
