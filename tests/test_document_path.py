"""Tests for GET /document/path endpoint — read .md by file path.

Verifies the endpoint that the Pi extension's doc_read tool calls as
a source_path fallback (previously dead code — endpoint didn't exist).
"""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestGetDocumentByPath:
    """Tests for GET /document/path?path=<full_path>."""

    def test_read_valid_md_file(self, tmp_path):
        """Should return .md file content with frontmatter stripped."""
        md_file = tmp_path / "test.md"
        md_file.write_text(
            "---\ntitle: Test\ntags: [a]\n---\n\n# Hello World\n\nContent here.",
            encoding="utf-8",
        )

        resp = client.get("/document/path", params={"path": str(md_file)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "test"
        assert "# Hello World" in data["full_content"]
        assert "Content here." in data["full_content"]
        # Frontmatter should be stripped
        assert "title: Test" not in data["full_content"]
        assert "---" not in data["full_content"].split("\n")[0]

    def test_read_nonexistent_file(self):
        """Should return 404 for nonexistent file."""
        resp = client.get("/document/path", params={"path": "/nonexistent/file.md"})
        assert resp.status_code == 404

    def test_read_non_md_file(self, tmp_path):
        """Should return 400 for non-.md files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello", encoding="utf-8")

        resp = client.get("/document/path", params={"path": str(txt_file)})
        assert resp.status_code == 400

    def test_read_md_without_frontmatter(self, tmp_path):
        """Should handle .md files without frontmatter."""
        md_file = tmp_path / "plain.md"
        md_file.write_text("# Plain doc\n\nNo frontmatter here.", encoding="utf-8")

        resp = client.get("/document/path", params={"path": str(md_file)})
        assert resp.status_code == 200
        data = resp.json()
        assert "# Plain doc" in data["full_content"]

    def test_route_not_captured_by_doc_id(self, tmp_path):
        """'/document/path' should NOT match '/document/{doc_id}'."""
        # This is the critical test — verifies route ordering
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test", encoding="utf-8")

        resp = client.get("/document/path", params={"path": str(md_file)})
        # If route ordering is wrong, this would 422 (missing index_path param)
        # or return wrong data
        assert resp.status_code == 200
        assert resp.json()["title"] == "test"
