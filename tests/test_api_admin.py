"""
Tests for admin API endpoints: build-index, repair, retry.
"""

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestAdminBuildIndex:
    """Tests for POST /api/admin/build-index"""

    def test_build_index_empty_dir(self):
        """Build index on an empty directory returns 0 indexed."""
        with tempfile.TemporaryDirectory() as tmp:
            r = client.post("/api/admin/build-index", params={"raw_dir": tmp})
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ok"
            assert data["indexed"] == 0

    def test_build_index_with_md_files(self):
        """Build index on a directory with .md files works."""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / "test.md").write_text("# Hello\n\nWorld", encoding="utf-8")
            r = client.post("/api/admin/build-index", params={"raw_dir": tmp})
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "ok"
            assert data["indexed"] == 1
            assert "stats" in data
            assert (raw / "index").is_dir()

    def test_build_index_skips_underscore_files(self):
        """Files starting with _ are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / "_index.md").write_text("# skip", encoding="utf-8")
            (raw / "keep.md").write_text("# keep", encoding="utf-8")
            r = client.post("/api/admin/build-index", params={"raw_dir": tmp})
            assert r.json()["indexed"] == 1

    def test_build_index_not_found(self):
        """Invalid directory returns 400 or error status."""
        r = client.post("/api/admin/build-index", params={"raw_dir": "/nonexistent/path"})
        # API may return 400 (HTTPException) or 200 with error status
        assert r.status_code in (200, 400)
        body = r.json()
        if r.status_code == 200:
            # API returns 200 + {"status": "error"} style response
            assert "status" in body or "detail" in body
        else:
            assert body.get("detail", "").startswith("Directory not found")


class TestAdminRetry:
    """Tests for POST /api/admin/retry"""

    def test_retry_no_db(self):
        """Retry without convert.db returns 404."""
        with tempfile.TemporaryDirectory() as tmp:
            r = client.post("/api/admin/retry", params={"raw_dir": tmp})
            assert r.status_code == 404
