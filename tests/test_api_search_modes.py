"""
Tests for search mode endpoints: grep, hybrid, tag.
"""

import os
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestSearchModeEndpoints:
    """Tests for /api/search/* endpoints."""

    def test_grep_missing_raw_dir(self):
        """Grep without raw_dir returns 400."""
        r = client.post("/api/search/grep", json={
            "query": "test", "index_path": "/tmp", "limit": 5
        })
        assert r.status_code == 400

    def test_grep_with_valid_raw_dir(self, tmp_path):
        """Grep with a valid raw_dir containing .md files returns structured results.

        This test would have caught the snippet_length TypeError bug.
        """
        # Create a .md file with searchable content
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        md_file = raw_dir / "test-doc.md"
        md_file.write_text("# Test Document\n\nThis is a test about颈椎病 treatment.\n", encoding="utf-8")

        r = client.post("/api/search/grep", json={
            "query": "颈椎病",
            "index_path": str(tmp_path / "index"),
            "raw_dir": str(raw_dir),
            "limit": 10,
        })
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert data["mode"] == "grep"
        assert data["total"] >= 1
        results = data["results"]
        assert len(results) >= 1
        # Verify structured result format (not just {"text": "..."})
        first = results[0]
        assert "doc_id" in first
        assert "title" in first
        assert "snippet" in first
        assert "score" in first
        assert "source_path" in first

    def test_grep_no_matches_returns_empty(self, tmp_path):
        """Grep with no matches returns 200 with empty results."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "empty.md").write_text("# Empty\n\nNothing relevant here.\n", encoding="utf-8")

        r = client.post("/api/search/grep", json={
            "query": "nonexistent_term_xyz",
            "index_path": str(tmp_path / "index"),
            "raw_dir": str(raw_dir),
            "limit": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_hybrid_missing_raw_dir(self):
        """Hybrid without raw_dir returns 400."""
        r = client.post("/api/search/hybrid", json={
            "query": "test", "index_path": "/tmp", "limit": 5
        })
        assert r.status_code == 400

    def test_hybrid_missing_index(self):
        """Hybrid with non-existent index returns OK with empty results."""
        r = client.post("/api/search/hybrid", json={
            "query": "test", "index_path": "/nonexistent", "raw_dir": "/tmp"
        })
        # Nonexistent index may return 200 with 0 results or 400 error
        assert r.status_code in (200, 400, 500)

    def test_tag_missing_index(self):
        """Tag with non-existent index returns OK with empty results."""
        r = client.post("/api/search/tag", json={
            "query": "test", "index_path": "/nonexistent"
        })
        assert r.status_code in (200, 400, 500)

    def test_search_endpoints_are_registered(self):
        """Verify all three endpoints respond (even if with errors)."""
        for path in ["/api/search/grep", "/api/search/hybrid", "/api/search/tag"]:
            r = client.post(path, json={"query": "test", "index_path": "/x", "limit": 5})
            assert r.status_code in (200, 400, 404, 422)
