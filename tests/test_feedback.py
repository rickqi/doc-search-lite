"""
Tests for search relevance feedback: POST /api/feedback, GET /api/admin/feedback,
and ConvertDB.record_feedback / get_feedback_summary.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.storage.convert_db import ConvertDB

client = TestClient(app)


# ── ConvertDB unit tests ──────────────────────────────────────


class TestConvertDBFeedback:
    """Direct ConvertDB record_feedback / get_feedback_summary tests."""

    def test_record_and_summary(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            rid = db.record_feedback(query="年假", rating=1, doc_id="abc123",
                                     doc_title="leave.md")
            assert isinstance(rid, int) and rid > 0

            db.record_feedback(query="报销", rating=-1, doc_id="def456",
                               doc_title="travel.md")
            db.record_feedback(query="报销", rating=-1, doc_id="def456",
                               doc_title="travel.md")

            summary = db.get_feedback_summary(days=30)
            assert summary["total"] == 3
            assert summary["total_up"] == 1
            assert summary["total_down"] == 2

            worst = summary["worst_rated_docs"]
            assert len(worst) >= 1
            assert worst[0]["doc_title"] == "travel.md"
            assert worst[0]["down_count"] == 2

            assert len(summary["recent"]) == 3
        finally:
            db.close()

    def test_record_feedback_minimal(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        try:
            rid = db.record_feedback(query="test", rating=-1)
            assert rid > 0
            summary = db.get_feedback_summary(days=30)
            assert summary["total"] == 1
            assert summary["total_down"] == 1
        finally:
            db.close()

    def test_feedback_table_created_on_fresh_db(self, tmp_path):
        db = ConvertDB(tmp_path / "fresh.db")
        db.open()
        try:
            rows = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='search_feedback'"
            ).fetchall()
            assert len(rows) == 1
        finally:
            db.close()


# ── API endpoint tests ────────────────────────────────────────


class TestFeedbackAPI:
    """Tests for POST /api/feedback and GET /api/admin/feedback."""

    def test_submit_feedback_thumbs_up(self):
        with patch("src.api._resolve_feedback_db", return_value=None):
            r = client.post("/api/feedback", json={
                "query": "年假如何申请",
                "rating": 1,
                "doc_id": "abc123",
                "doc_title": "年假制度.md",
            })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_submit_feedback_thumbs_down(self):
        with patch("src.api._resolve_feedback_db", return_value=None):
            r = client.post("/api/feedback", json={
                "query": "差旅报销",
                "rating": -1,
            })
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_submit_feedback_invalid_rating_zero(self):
        r = client.post("/api/feedback", json={
            "query": "test",
            "rating": 0,
        })
        assert r.status_code == 422

    def test_submit_feedback_invalid_rating_two(self):
        r = client.post("/api/feedback", json={
            "query": "test",
            "rating": 2,
        })
        assert r.status_code == 422

    def test_submit_feedback_missing_query(self):
        r = client.post("/api/feedback", json={"rating": 1})
        assert r.status_code == 422

    def test_submit_feedback_missing_rating(self):
        r = client.post("/api/feedback", json={"query": "test"})
        assert r.status_code == 422

    def test_submit_feedback_persisted_to_db(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        db = ConvertDB(raw_dir / "convert.db")
        db.open()
        db.close()

        r = client.post("/api/feedback", json={
            "query": "报销流程",
            "rating": 1,
            "doc_id": "rpt001",
            "doc_title": "财务制度.md",
            "index_path": str(raw_dir / "index"),
        })
        assert r.status_code == 200
        data = r.json()
        assert data["persisted"] is True

        db2 = ConvertDB(raw_dir / "convert.db")
        db2.open()
        try:
            summary = db2.get_feedback_summary(days=1)
            assert summary["total"] == 1
            assert summary["total_up"] == 1
        finally:
            db2.close()

    def test_admin_feedback_summary(self, tmp_path):
        db = ConvertDB(tmp_path / "convert.db")
        db.open()
        try:
            db.record_feedback(query="q1", rating=1, doc_title="doc_a.md")
            db.record_feedback(query="q2", rating=-1, doc_title="doc_b.md")
            db.record_feedback(query="q3", rating=-1, doc_title="doc_b.md")
        finally:
            db.close()

        r = client.get("/api/admin/feedback", params={"raw_dir": str(tmp_path), "days": 7})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3
        assert data["total_up"] == 1
        assert data["total_down"] == 2
        assert len(data["worst_rated_docs"]) >= 1

    def test_admin_feedback_not_found(self, tmp_path):
        r = client.get("/api/admin/feedback",
                       params={"raw_dir": str(tmp_path / "nonexistent"), "days": 7})
        assert r.status_code == 404

    def test_admin_feedback_empty(self, tmp_path):
        db = ConvertDB(tmp_path / "convert.db")
        db.open()
        db.close()

        r = client.get("/api/admin/feedback", params={"raw_dir": str(tmp_path), "days": 7})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["total_up"] == 0
        assert data["total_down"] == 0
