"""
Tests for P1+P2 API endpoints: sessions, DB panel, diagnostics, tokens,
suggest, rerank, and document/{doc_id}.

Covers 22 API routes across 7 test classes (~30 tests).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app, _searchers, _convert_dbs
import src.api as api_module

client = TestClient(app)


# ── Autouse: clear all module-level caches ─────────────────────


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear module-level caches to prevent cross-test pollution."""
    _searchers.clear()
    # Close any open db connections before clearing
    for db in _convert_dbs.values():
        try:
            db.close()
        except Exception:
            pass
    _convert_dbs.clear()
    api_module._session_manager = None
    yield
    # Cleanup after test
    _searchers.clear()
    for db in _convert_dbs.values():
        try:
            db.close()
        except Exception:
            pass
    _convert_dbs.clear()
    api_module._session_manager = None


# ── Helpers ────────────────────────────────────────────────────


def _build_index(raw_dir: Path) -> str:
    """Create .md files under *raw_dir*, build a Tantivy index, return index path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    r = client.post("/api/admin/build-index", params={"raw_dir": str(raw_dir)})
    assert r.status_code == 200, f"build-index failed: {r.text}"
    return str(raw_dir / "index")


def _create_convert_db(raw_dir: Path) -> str:
    """Create a real ConvertDB with test data, return str(raw_dir)."""
    from src.storage.convert_db import ConvertDB

    raw_dir.mkdir(parents=True, exist_ok=True)
    db = ConvertDB(raw_dir / "convert.db")
    db.open()
    try:
        dir_id = db.upsert_directory(relative_path=".", name="root")
        db.upsert_file(
            relative_path="test.md",
            directory_id=dir_id,
            filename="test.md",
            extension=".md",
            file_size=100,
            source_mtime="2024-01-01T00:00:00",
            source_hash="abc123",
        )
        db.upsert_file(
            relative_path="report.pdf",
            directory_id=dir_id,
            filename="report.pdf",
            extension=".pdf",
            file_size=5000,
            source_mtime="2024-01-02T00:00:00",
            source_hash="def456",
        )
        db.create_batch(batch_type="full", total_files=2)
    finally:
        db.close()
    return str(raw_dir)


def _path_for_url(p) -> str:
    """Convert a path to URL-safe forward-slash form for path parameters."""
    return str(p).replace("\\", "/")


# ── TestSessions (7 tests, 4 routes) ───────────────────────────


class TestSessions:
    """Tests for GET/POST/DELETE /api/sessions."""

    @pytest.fixture(autouse=True)
    def _fresh_session_manager(self):
        """Reset session manager singleton for clean isolation."""
        import src.web.session_manager as sm_mod

        old = sm_mod._default_manager
        sm_mod._default_manager = None
        api_module._session_manager = None
        yield
        sm_mod._default_manager = old
        api_module._session_manager = None

    def test_list_sessions(self):
        """GET /api/sessions returns a sessions list."""
        r = client.get("/api/sessions")
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_create_session_success(self, tmp_path):
        """POST /api/sessions with a valid index path returns session_id."""
        idx = tmp_path / "index"
        idx.mkdir()
        r = client.post("/api/sessions", params={"index_path": str(idx)})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert "created" in data
        assert data["indexes"] == 1

    def test_create_session_nonexistent_index(self):
        """POST /api/sessions with a non-existent index returns 404."""
        r = client.post("/api/sessions", params={"index_path": "/nonexistent/xyz_abc"})
        assert r.status_code == 404

    def test_get_session(self, tmp_path):
        """GET /api/sessions/{id} returns session details."""
        idx = tmp_path / "index"
        idx.mkdir()
        create = client.post("/api/sessions", params={"index_path": str(idx)})
        sid = create.json()["session_id"]

        r = client.get(f"/api/sessions/{sid}")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == sid
        assert "messages" in data
        assert "model" in data

    def test_get_session_not_found(self):
        """GET /api/sessions/{invalid} returns 404."""
        r = client.get("/api/sessions/invalid-id-12345")
        assert r.status_code == 404

    def test_delete_session(self, tmp_path):
        """DELETE /api/sessions/{id} removes the session."""
        idx = tmp_path / "index"
        idx.mkdir()
        create = client.post("/api/sessions", params={"index_path": str(idx)})
        sid = create.json()["session_id"]

        r = client.delete(f"/api/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

        # Verify it's gone
        r2 = client.get(f"/api/sessions/{sid}")
        assert r2.status_code == 404

    def test_delete_session_not_found(self):
        """DELETE /api/sessions/{invalid} returns 404."""
        r = client.delete("/api/sessions/invalid-id-67890")
        assert r.status_code == 404


# ── TestDbPanel (7 tests, 6 routes) ────────────────────────────


class TestDbPanel:
    """Tests for /api/db/{raw_dir}/... endpoints."""

    def test_db_stats(self, tmp_path):
        """GET /api/db/{raw_dir}/stats returns conversion stats."""
        raw = tmp_path / "raw"
        _create_convert_db(raw)
        r = client.get(f"/api/db/{_path_for_url(raw)}/stats")
        assert r.status_code == 200
        data = r.json()
        assert "by_status" in data
        assert "latest_batch" in data

    def test_db_stats_no_db(self, tmp_path):
        """GET /api/db/{raw_dir}/stats on a dir without convert.db returns 404."""
        r = client.get(f"/api/db/{_path_for_url(tmp_path / 'nodb')}/stats")
        assert r.status_code == 404

    def test_db_files(self, tmp_path):
        """GET /api/db/{raw_dir}/files returns file list."""
        raw = tmp_path / "raw"
        _create_convert_db(raw)
        r = client.get(f"/api/db/{_path_for_url(raw)}/files")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_db_batches(self, tmp_path):
        """GET /api/db/{raw_dir}/batches returns batch info."""
        raw = tmp_path / "raw"
        _create_convert_db(raw)
        r = client.get(f"/api/db/{_path_for_url(raw)}/batches")
        assert r.status_code == 200
        data = r.json()
        assert "latest_batch" in data
        assert "active_batch" in data

    def test_db_token_summary(self, tmp_path):
        """GET /api/db/{raw_dir}/token/summary returns token usage."""
        raw = tmp_path / "raw"
        _create_convert_db(raw)
        r = client.get(f"/api/db/{_path_for_url(raw)}/token/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["days"] == 7

    def test_db_budget(self, tmp_path):
        """GET /api/db/{raw_dir}/budget returns budget status."""
        raw = tmp_path / "raw"
        _create_convert_db(raw)
        r = client.get(f"/api/db/{_path_for_url(raw)}/budget")
        assert r.status_code == 200
        data = r.json()
        assert "budgets" in data

    def test_db_file_detail(self, tmp_path):
        """GET /api/db/{raw_dir}/files/{file_id} returns file detail."""
        raw = tmp_path / "raw"
        _create_convert_db(raw)
        r = client.get(f"/api/db/{_path_for_url(raw)}/files/1")
        assert r.status_code == 200
        data = r.json()
        assert "filename" in data


# ── TestDiagnostics (5 tests, 4 routes) ────────────────────────


class TestDiagnostics:
    """Tests for /api/admin/diagnostics/... endpoints."""

    def test_diagnostics_summary_no_db(self, tmp_path):
        """GET /api/admin/diagnostics/summary without convert.db returns 404."""
        r = client.get("/api/admin/diagnostics/summary",
                       params={"raw_dir": str(tmp_path / "nonexistent")})
        assert r.status_code == 404

    def test_diagnostics_slow_queries_no_db(self, tmp_path):
        """GET /api/admin/diagnostics/slow-queries without convert.db returns 404."""
        r = client.get("/api/admin/diagnostics/slow-queries",
                       params={"raw_dir": str(tmp_path / "nonexistent")})
        assert r.status_code == 404

    def test_diagnostics_step_breakdown_no_db(self, tmp_path):
        """GET /api/admin/diagnostics/step-breakdown without convert.db returns 404."""
        r = client.get("/api/admin/diagnostics/step-breakdown",
                       params={"raw_dir": str(tmp_path / "nonexistent")})
        assert r.status_code == 404

    def test_diagnostics_llm_calls_no_db(self, tmp_path):
        """GET /api/admin/diagnostics/llm-calls without convert.db returns 404."""
        r = client.get("/api/admin/diagnostics/llm-calls",
                       params={"raw_dir": str(tmp_path / "nonexistent")})
        assert r.status_code == 404

    def test_diagnostics_summary_with_db(self, tmp_path):
        """GET /api/admin/diagnostics/summary with a real convert.db returns 200."""
        raw = tmp_path / "raw"
        _create_convert_db(raw)
        r = client.get("/api/admin/diagnostics/summary",
                       params={"raw_dir": str(raw)})
        assert r.status_code == 200


# ── TestTokens (3 tests, 3 routes) ─────────────────────────────


class TestTokens:
    """Tests for POST/GET/DELETE /api/admin/tokens."""

    @pytest.fixture(autouse=True)
    def _mock_token_store(self, tmp_path):
        """Provide a real TokenStore backed by a temp file (open mode has _token_store=None)."""
        from src.web.auth import TokenStore

        store = TokenStore(tmp_path / "tokens.json")
        old = api_module._token_store
        api_module._token_store = store
        yield store
        api_module._token_store = old

    def test_create_token(self):
        """POST /api/admin/tokens creates a new token."""
        r = client.post("/api/admin/tokens", json={
            "name": "test-token",
            "scopes": ["search"],
        })
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert data["name"] == "test-token"
        assert "key" in data  # include_key=True on create

    def test_list_tokens(self):
        """GET /api/admin/tokens lists all tokens (keys redacted)."""
        client.post("/api/admin/tokens", json={"name": "t1", "scopes": ["*"]})
        r = client.get("/api/admin/tokens")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        assert isinstance(data["tokens"], list)

    def test_delete_token(self):
        """DELETE /api/admin/tokens/{id} revokes the token."""
        create = client.post("/api/admin/tokens", json={
            "name": "to-delete",
            "scopes": ["read"],
        })
        tid = create.json()["id"]

        r = client.delete(f"/api/admin/tokens/{tid}")
        assert r.status_code == 200
        assert r.json()["status"] == "revoked"


# ── TestSuggest (2 tests, 1 route) ─────────────────────────────


class TestSuggest:
    """Tests for GET /api/suggest."""

    def test_suggest_basic(self, tmp_path):
        """GET /api/suggest returns suggestions from indexed titles."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "annual_leave.md").write_text(
            "# Annual Leave Policy\n\nContent about leave.\n", encoding="utf-8"
        )
        idx = _build_index(raw)

        r = client.get("/api/suggest", params={
            "q": "ann",
            "index_path": idx,
        })
        assert r.status_code == 200
        data = r.json()
        assert "suggestions" in data
        assert data["query"] == "ann"
        assert isinstance(data["suggestions"], list)

    def test_suggest_no_index(self):
        """GET /api/suggest with non-existent index returns 404."""
        r = client.get("/api/suggest", params={
            "q": "test",
            "index_path": "/nonexistent/xyz_idx",
        })
        assert r.status_code == 404


# ── TestRerank (3 tests, 1 route) ──────────────────────────────


class TestRerank:
    """Tests for POST /rerank."""

    def test_rerank_missing_index_path(self):
        """POST /rerank without index_path returns 400."""
        r = client.post("/rerank", json={
            "query": "test",
            "doc_ids": ["doc1"],
        })
        assert r.status_code == 400

    def test_rerank_empty_doc_ids(self):
        """POST /rerank with empty doc_ids returns 400."""
        r = client.post("/rerank", json={
            "query": "test",
            "doc_ids": [],
            "index_path": "/some/path",
        })
        assert r.status_code == 400

    def test_rerank_graceful_degradation(self, tmp_path):
        """POST /rerank falls back to original order when reranker API fails."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc1.md").write_text("# Doc 1\n\nAnnual leave policy details.\n", encoding="utf-8")
        (raw / "doc2.md").write_text("# Doc 2\n\nTravel reimbursement guide.\n", encoding="utf-8")
        idx = _build_index(raw)

        # Get real doc_ids from the index
        search = client.post("/query", json={"query": "policy", "index_path": idx})
        doc_ids = [hit["doc_id"] for hit in search.json()["results"]]
        assert len(doc_ids) >= 1

        # Rerank — ZhipuAIReranker fails (no real API key), falls back gracefully
        r = client.post("/rerank", json={
            "query": "annual leave",
            "doc_ids": doc_ids,
            "index_path": idx,
        })
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert len(data["results"]) >= 1


# ── TestDocumentByDocId (2 tests, 1 route) ─────────────────────


class TestDocumentByDocId:
    """Tests for GET /document/{doc_id}."""

    def test_get_document_by_id(self, tmp_path):
        """GET /document/{doc_id} returns full document content."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "test_doc.md").write_text(
            "# Test Document\n\nFull searchable content here.\n", encoding="utf-8"
        )
        idx = _build_index(raw)

        # Get a valid doc_id from search
        search = client.post("/query", json={"query": "searchable", "index_path": idx})
        assert search.json()["total"] >= 1
        doc_id = search.json()["results"][0]["doc_id"]

        r = client.get(f"/document/{doc_id}", params={"index_path": idx})
        assert r.status_code == 200
        data = r.json()
        assert data["doc_id"] == doc_id
        assert "full_content" in data
        assert len(data["full_content"]) > 0

    def test_get_document_not_found(self, tmp_path):
        """GET /document/{nonexistent} returns 404."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "placeholder.md").write_text("# Placeholder\n", encoding="utf-8")
        idx = _build_index(raw)

        r = client.get("/document/nonexistent_id_00000", params={"index_path": idx})
        assert r.status_code == 404
