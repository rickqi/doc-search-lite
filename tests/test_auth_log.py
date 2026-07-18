"""
Tests for authentication audit logging:
ConvertDB.record_auth_log / get_auth_log and GET /api/admin/auth-log.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.storage.convert_db import ConvertDB
from src.web import auth as auth_module

client = TestClient(app)


# ── ConvertDB unit tests ──────────────────────────────────────


class TestConvertDBAuthLog:
    """Direct ConvertDB record_auth_log / get_auth_log tests."""

    def test_record_and_retrieve(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            rid = db.record_auth_log(
                endpoint="/api/search",
                method="GET",
                token_id="sk-abc",
                client_ip="10.0.0.1",
                status_code=200,
            )
            assert isinstance(rid, int) and rid > 0

            records = db.get_auth_log(days=7)
            assert len(records) == 1
            r = records[0]
            assert r["endpoint"] == "/api/search"
            assert r["method"] == "GET"
            assert r["token_id"] == "sk-abc"
            assert r["client_ip"] == "10.0.0.1"
            assert r["status_code"] == 200
        finally:
            db.close()

    def test_record_minimal(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            rid = db.record_auth_log(endpoint="/query", method="POST")
            assert rid > 0

            records = db.get_auth_log(days=7)
            assert len(records) == 1
            assert records[0]["token_id"] is None
            assert records[0]["client_ip"] is None
            assert records[0]["status_code"] == 200
        finally:
            db.close()

    def test_filter_by_token_id(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            db.record_auth_log(endpoint="/a", method="GET", token_id="sk-1")
            db.record_auth_log(endpoint="/b", method="GET", token_id="sk-2")
            db.record_auth_log(endpoint="/c", method="GET", token_id="sk-1")

            records = db.get_auth_log(days=7, token_id="sk-1")
            assert len(records) == 2
            assert all(r["token_id"] == "sk-1" for r in records)
        finally:
            db.close()

    def test_filter_by_days(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            db.record_auth_log(endpoint="/recent", method="GET")
            db.conn.execute(
                "INSERT INTO auth_log (endpoint, method, created_at) "
                "VALUES ('/old', 'GET', datetime('now', '-30 days'))"
            )
            db.conn.commit()

            records_7d = db.get_auth_log(days=7)
            assert len(records_7d) == 1
            assert records_7d[0]["endpoint"] == "/recent"

            records_60d = db.get_auth_log(days=60)
            assert len(records_60d) == 2
        finally:
            db.close()

    def test_limit(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            for i in range(10):
                db.record_auth_log(endpoint=f"/e{i}", method="GET")
            records = db.get_auth_log(days=7, limit=3)
            assert len(records) == 3
        finally:
            db.close()

    def test_table_created_on_fresh_db(self, tmp_path):
        db = ConvertDB(tmp_path / "fresh.db")
        db.open()
        try:
            rows = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='auth_log'"
            ).fetchall()
            assert len(rows) == 1
        finally:
            db.close()

    def test_indexes_exist(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            indexes = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='auth_log'"
            ).fetchall()
            names = {r["name"] for r in indexes}
            assert "idx_auth_log_created" in names
            assert "idx_auth_log_token" in names
        finally:
            db.close()


# ── API endpoint tests ────────────────────────────────────────


class TestAuthLogAPI:
    """Tests for GET /api/admin/auth-log."""

    def test_admin_auth_log_returns_records(self, tmp_path):
        db = ConvertDB(tmp_path / "convert.db")
        db.open()
        try:
            db.record_auth_log(
                endpoint="/api/search",
                method="GET",
                token_id="sk-test",
                client_ip="127.0.0.1",
                status_code=200,
            )
        finally:
            db.close()

        r = client.get(
            "/api/admin/auth-log",
            params={"raw_dir": str(tmp_path), "days": 7},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["records"]) == 1
        rec = data["records"][0]
        assert rec["endpoint"] == "/api/search"
        assert rec["token_id"] == "sk-test"

    def test_admin_auth_log_filter_token_id(self, tmp_path):
        db = ConvertDB(tmp_path / "convert.db")
        db.open()
        try:
            db.record_auth_log(endpoint="/a", method="GET", token_id="sk-1")
            db.record_auth_log(endpoint="/b", method="GET", token_id="sk-2")
        finally:
            db.close()

        r = client.get(
            "/api/admin/auth-log",
            params={"raw_dir": str(tmp_path), "token_id": "sk-1"},
        )
        assert r.status_code == 200
        records = r.json()["records"]
        assert len(records) == 1
        assert records[0]["token_id"] == "sk-1"

    def test_admin_auth_log_empty(self, tmp_path):
        db = ConvertDB(tmp_path / "convert.db")
        db.open()
        db.close()

        r = client.get(
            "/api/admin/auth-log",
            params={"raw_dir": str(tmp_path), "days": 7},
        )
        assert r.status_code == 200
        assert r.json()["records"] == []

    def test_admin_auth_log_not_found(self, tmp_path):
        r = client.get(
            "/api/admin/auth-log",
            params={"raw_dir": str(tmp_path / "nonexistent"), "days": 7},
        )
        assert r.status_code == 404


# ── Middleware integration tests ──────────────────────────────


class TestAuthLogMiddleware:
    """Tests that the middleware logs authenticated requests."""

    def test_middleware_logs_request(self, tmp_path):
        db_path = tmp_path / "convert.db"
        db = ConvertDB(db_path)
        db.open()
        db.close()

        with patch.object(auth_module, "_get_auth_log_db", return_value=None):
            auth_module.set_auth_log_db_path(str(db_path))
            try:
                r = client.get("/api/admin/auth-log", params={"raw_dir": str(tmp_path)})
                assert r.status_code == 200
            finally:
                auth_module.set_auth_log_db_path(None)
                auth_module._auth_log_db = None

    def test_middleware_failure_does_not_crash(self):
        with patch.object(auth_module, "_log_auth_request", side_effect=RuntimeError("boom")):
            r = client.get("/health")
            assert r.status_code == 200

    def test_log_auth_request_no_db(self):
        auth_module._auth_log_db = None
        auth_module._log_auth_request(
            endpoint="/test", method="GET", token_id=None, status_code=200
        )

    def test_log_auth_request_writes_to_db(self, tmp_path):
        db_path = tmp_path / "convert.db"
        db = ConvertDB(db_path)
        db.open()
        try:
            auth_module._auth_log_db = db
            auth_module._log_auth_request(
                endpoint="/api/search",
                method="GET",
                token_id="sk-xyz",
                client_ip="192.168.1.1",
                status_code=200,
            )
            auth_module._auth_log_db = None

            records = db.get_auth_log(days=7)
            assert len(records) == 1
            assert records[0]["endpoint"] == "/api/search"
            assert records[0]["token_id"] == "sk-xyz"
            assert records[0]["client_ip"] == "192.168.1.1"
        finally:
            db.close()
