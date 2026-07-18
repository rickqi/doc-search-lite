"""Unit tests for ConvertDB SQLite state manager."""

import sqlite3

import pytest

from src.storage.convert_db import ConvertDB

# ── Helpers ──────────────────────────────────────────────────────────


def _insert_dir(db: ConvertDB, path: str = "root", **kwargs) -> int:
    """Shortcut to upsert a directory and return its id."""
    return db.upsert_directory(
        relative_path=path,
        parent_id=kwargs.get("parent_id"),
        depth=kwargs.get("depth", 0),
        name=kwargs.get("name", path.split("/")[-1]),
    )


def _insert_file(db: ConvertDB, dir_id: int, path: str = "a.pdf", **kwargs) -> int:
    """Shortcut to upsert a file and return its id."""
    return db.upsert_file(
        relative_path=path,
        directory_id=dir_id,
        filename=kwargs.get("filename", path.split("/")[-1]),
        extension=kwargs.get("extension", ".pdf"),
        file_size=kwargs.get("file_size", 100),
        source_mtime=kwargs.get("source_mtime", "2024-01-01T00:00:00"),
        source_hash=kwargs.get("source_hash", "deadbeef"),
    )


# ── Lifecycle ────────────────────────────────────────────────────────


class TestConvertDBLifecycle:
    """Test open / close / context manager."""

    def test_init(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        assert db.db_path == tmp_path / "test.db"
        assert db._conn is None

    def test_open_creates_db(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        assert db._conn is not None
        db.close()

    def test_open_returns_self(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        result = db.open()
        assert result is db
        db.close()

    def test_close(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        db.close()
        assert db._conn is None

    def test_close_idempotent(self, tmp_path):
        """close() is idempotent — safe to call multiple times."""
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        db.close()
        assert db._conn is None
        db.close()  # second close is a no-op, should not raise

    def test_context_manager(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db._conn is not None
        assert db._conn is None

    def test_conn_property_raises_when_not_open(self, tmp_path):
        db = ConvertDB(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="数据库未打开"):
            _ = db.conn

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "test.db"
        with ConvertDB(db_path) as db:
            assert db_path.exists()

    def test_wal_mode_enabled(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            row = db.conn.execute("PRAGMA journal_mode").fetchone()
            assert row[0].lower() == "wal"


# ── Schema ───────────────────────────────────────────────────────────


class TestConvertDBSchema:
    """Test schema creation and migration."""

    def test_tables_created(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            tables = {
                row[0]
                for row in db.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for expected in ("directories", "files", "batches", "skipped", "config", "token_usage"):
                assert expected in tables

    def test_schema_version_set(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            version = db._get_config("schema_version")
            assert version == "2.1"

    def test_migration_from_10_to_11(self, tmp_path):
        """Simulate a 1.0 database then open with current code triggers migration."""
        db_path = tmp_path / "test.db"
        # Manually create a minimal 1.0 schema (no token columns)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE directories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER,
                name TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                depth INTEGER NOT NULL DEFAULT 0,
                file_count INTEGER NOT NULL DEFAULT 0,
                total_size INTEGER NOT NULL DEFAULT 0,
                index_generated INTEGER NOT NULL DEFAULT 0,
                index_mtime TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                directory_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                extension TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                source_mtime TEXT,
                source_hash TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                converter TEXT,
                convert_time REAL,
                convert_at TEXT,
                convert_version TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                ocr_used INTEGER NOT NULL DEFAULT 0,
                ocr_model TEXT,
                output_path TEXT,
                output_size INTEGER,
                output_hash TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                total_files INTEGER NOT NULL DEFAULT 0,
                processed INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                finished_at TEXT,
                error_summary TEXT,
                config_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE skipped (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                reason TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # Insert a file row to verify migration preserves data
        conn.execute(
            "INSERT INTO directories (relative_path, name, depth) VALUES ('root', 'root', 0)"
        )
        conn.execute(
            "INSERT INTO files (directory_id, filename, relative_path, extension) VALUES (1, 'a.pdf', 'a.pdf', '.pdf')"
        )
        conn.execute("INSERT INTO config (key, value) VALUES ('schema_version', '1.0')")
        conn.commit()
        conn.close()

        # Open with ConvertDB — should trigger migration
        with ConvertDB(db_path) as db:
            assert db._get_config("schema_version") == "2.1"
            # migrated columns should exist
            columns = {row[1] for row in db.conn.execute("PRAGMA table_info(files)").fetchall()}
            assert "ocr_input_tokens" in columns
            assert "ocr_output_tokens" in columns
            assert "ocr_total_tokens" in columns
            assert "metadata_json" in columns
            # data preserved
            f = db.get_file("a.pdf")
            assert f is not None
            assert f["filename"] == "a.pdf"


# ── Config ───────────────────────────────────────────────────────────


class TestConvertDBConfig:
    """Test internal config get/set."""

    def test_set_and_get(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            db._set_config("test_key", "test_value")
            assert db._get_config("test_key") == "test_value"

    def test_get_missing_returns_default(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db._get_config("nonexistent", "fallback") == "fallback"

    def test_get_missing_returns_none(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db._get_config("nonexistent") is None

    def test_upsert_overwrites(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            db._set_config("k", "v1")
            db._set_config("k", "v2")
            assert db._get_config("k") == "v2"


# ── Directories ──────────────────────────────────────────────────────


class TestConvertDBDirectories:
    """Test directory CRUD operations."""

    def test_upsert_returns_id(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = db.upsert_directory("sub", parent_id=None, depth=0, name="sub")
            assert isinstance(dir_id, int)
            assert dir_id > 0

    def test_upsert_insert_and_get(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = db.upsert_directory("a/b", parent_id=None, depth=1, name="b")
            d = db.get_directory("a/b")
            assert d is not None
            assert d["id"] == dir_id
            assert d["name"] == "b"
            assert d["depth"] == 1

    def test_upsert_idempotent(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            id1 = db.upsert_directory("x", depth=0, name="x")
            id2 = db.upsert_directory("x", depth=1, name="x_updated")
            assert id1 == id2
            d = db.get_directory("x")
            assert d["name"] == "x_updated"
            assert d["depth"] == 1

    def test_get_directory_missing(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db.get_directory("no/such/path") is None

    def test_get_directory_by_id(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db, "docs")
            d = db.get_directory_by_id(dir_id)
            assert d is not None
            assert d["relative_path"] == "docs"

    def test_get_directory_by_id_missing(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db.get_directory_by_id(9999) is None

    def test_list_subdirectories(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            parent_id = _insert_dir(db, "parent")
            _insert_dir(db, "parent/child_a", parent_id=parent_id, depth=1, name="child_a")
            _insert_dir(db, "parent/child_b", parent_id=parent_id, depth=1, name="child_b")
            _insert_dir(db, "orphan", depth=0, name="orphan")

            children = db.list_subdirectories(parent_id)
            assert len(children) == 2
            names = {c["name"] for c in children}
            assert names == {"child_a", "child_b"}

    def test_list_subdirectories_empty(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            parent_id = _insert_dir(db, "empty_parent")
            assert db.list_subdirectories(parent_id) == []

    def test_update_directory_stats(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db, "stats_dir")
            db.update_directory_stats(dir_id, file_count=42, total_size=102400)
            d = db.get_directory_by_id(dir_id)
            assert d["file_count"] == 42
            assert d["total_size"] == 102400

    def test_set_index_generated(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db, "idx_dir")
            d = db.get_directory_by_id(dir_id)
            assert d["index_generated"] == 0

            db.set_index_generated(dir_id, True)
            d = db.get_directory_by_id(dir_id)
            assert d["index_generated"] == 1
            assert d["index_mtime"] is not None

    def test_set_index_generated_false(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db, "idx_dir2")
            db.set_index_generated(dir_id, True)
            db.set_index_generated(dir_id, False)
            d = db.get_directory_by_id(dir_id)
            assert d["index_generated"] == 0

    def test_defaults(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db, "default_dir")
            d = db.get_directory_by_id(dir_id)
            assert d["file_count"] == 0
            assert d["total_size"] == 0
            assert d["index_generated"] == 0
            assert d["index_mtime"] is None


# ── Files ────────────────────────────────────────────────────────────


class TestConvertDBFiles:
    """Test file CRUD operations."""

    def test_upsert_file_returns_id(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id)
            assert isinstance(fid, int)
            assert fid > 0

    def test_upsert_file_insert_and_get(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            _insert_file(db, dir_id, "report.pdf", filename="report.pdf", extension=".pdf")
            f = db.get_file("report.pdf")
            assert f is not None
            assert f["filename"] == "report.pdf"
            assert f["extension"] == ".pdf"
            assert f["status"] == "pending"
            assert f["file_size"] == 100

    def test_upsert_file_idempotent(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            id1 = _insert_file(db, dir_id, "a.pdf")
            id2 = _insert_file(db, dir_id, "a.pdf")
            assert id1 == id2

    def test_upsert_file_success_unchanged_stays_success(self, tmp_path):
        """File with status=success and same mtime/hash should remain success."""
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "stable.pdf",
                               source_mtime="2024-01-01T00:00:00", source_hash="abc")
            db.update_file_status(fid, "success", converter="pdfplumber")

            # Re-upsert with same mtime/hash
            fid2 = db.upsert_file(
                relative_path="stable.pdf", directory_id=dir_id,
                filename="stable.pdf", extension=".pdf",
                file_size=200, source_mtime="2024-01-01T00:00:00", source_hash="abc",
            )
            assert fid2 == fid
            f = db.get_file("stable.pdf")
            assert f["status"] == "success"

    def test_upsert_file_success_changed_resets_to_pending(self, tmp_path):
        """File with status=success and different hash should reset to pending."""
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "changed.pdf",
                               source_mtime="2024-01-01T00:00:00", source_hash="abc")
            db.update_file_status(fid, "success", converter="pdfplumber",
                                  output_path="/out/changed.md")

            # Re-upsert with different hash
            db.upsert_file(
                relative_path="changed.pdf", directory_id=dir_id,
                filename="changed.pdf", extension=".pdf",
                file_size=200, source_mtime="2024-01-01T00:00:00", source_hash="xyz",
            )
            f = db.get_file("changed.pdf")
            assert f["status"] == "pending"
            assert f["converter"] is None
            assert f["output_path"] is None
            assert f["attempt_count"] == 0

    def test_upsert_file_success_mtime_changed_resets(self, tmp_path):
        """File with status=success and changed mtime should reset to pending."""
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "mtime.pdf",
                               source_mtime="2024-01-01T00:00:00", source_hash="abc")
            db.update_file_status(fid, "success")

            db.upsert_file(
                relative_path="mtime.pdf", directory_id=dir_id,
                filename="mtime.pdf", extension=".pdf",
                file_size=100, source_mtime="2024-02-01T00:00:00", source_hash="abc",
            )
            f = db.get_file("mtime.pdf")
            assert f["status"] == "pending"

    def test_upsert_file_pending_stays_pending(self, tmp_path):
        """Re-upserting a pending file should keep it pending."""
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "pend.pdf", source_hash="h1")
            f = db.get_file("pend.pdf")
            assert f["status"] == "pending"

            # Re-upsert with different hash
            db.upsert_file(
                relative_path="pend.pdf", directory_id=dir_id,
                filename="pend.pdf", extension=".pdf",
                file_size=200, source_mtime="2024-01-01T00:00:00", source_hash="h2",
            )
            f = db.get_file("pend.pdf")
            assert f["status"] == "pending"

    def test_get_file_missing(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db.get_file("nonexistent.pdf") is None

    def test_get_file_by_id(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "byid.pdf")
            f = db.get_file_by_id(fid)
            assert f is not None
            assert f["relative_path"] == "byid.pdf"

    def test_get_file_by_id_missing(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db.get_file_by_id(9999) is None

    def test_get_pending_files(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            _insert_file(db, dir_id, "p1.pdf")
            _insert_file(db, dir_id, "p2.pdf")
            fid3 = _insert_file(db, dir_id, "p3.pdf")
            db.update_file_status(fid3, "success")

            pending = db.get_pending_files()
            assert len(pending) == 2
            paths = {f["relative_path"] for f in pending}
            assert paths == {"p1.pdf", "p2.pdf"}

    def test_get_pending_files_limit(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            for i in range(5):
                _insert_file(db, dir_id, f"file{i}.pdf")
            pending = db.get_pending_files(limit=3)
            assert len(pending) == 3

    def test_get_files_by_directory(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            d1 = _insert_dir(db, "dir1")
            d2 = _insert_dir(db, "dir2")
            _insert_file(db, d1, "dir1/a.pdf")
            _insert_file(db, d1, "dir1/b.pdf")
            _insert_file(db, d2, "dir2/c.pdf")

            files = db.get_files_by_directory(d1)
            assert len(files) == 2

    def test_get_files_by_status(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            f1 = _insert_file(db, dir_id, "ok.pdf")
            _insert_file(db, dir_id, "pending.pdf")
            db.update_file_status(f1, "success")

            success = db.get_files_by_status("success")
            assert len(success) == 1
            assert success[0]["relative_path"] == "ok.pdf"

            pending = db.get_files_by_status("pending")
            assert len(pending) == 1

    def test_get_files_by_status_empty(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db.get_files_by_status("failed") == []

    def test_get_files_by_extension(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            _insert_file(db, dir_id, "a.pdf", extension=".pdf")
            _insert_file(db, dir_id, "b.docx", extension=".docx")
            _insert_file(db, dir_id, "c.pdf", extension=".pdf")

            pdfs = db.get_files_by_extension(".pdf")
            assert len(pdfs) == 2

    def test_update_file_status_basic(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "s.pdf")
            db.update_file_status(fid, "converting")
            f = db.get_file_by_id(fid)
            assert f["status"] == "converting"

    def test_update_file_status_with_kwargs(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "kw.pdf")
            db.update_file_status(
                fid, "success",
                converter="pdfplumber",
                convert_time=1.23,
                output_path="/out/kw.md",
                output_size=2048,
                attempt_count=1,
            )
            f = db.get_file_by_id(fid)
            assert f["status"] == "success"
            assert f["converter"] == "pdfplumber"
            assert f["convert_time"] == 1.23
            assert f["output_path"] == "/out/kw.md"
            assert f["output_size"] == 2048
            assert f["attempt_count"] == 1

    def test_update_file_status_ignores_unknown_kwargs(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "unk.pdf")
            db.update_file_status(fid, "converting", bogus_field="ignored")
            f = db.get_file_by_id(fid)
            assert f["status"] == "converting"

    def test_count_files_all(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            _insert_file(db, dir_id, "a.pdf")
            _insert_file(db, dir_id, "b.pdf")
            assert db.count_files() == 2

    def test_count_files_by_status(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            _insert_file(db, dir_id, "a.pdf")
            fid = _insert_file(db, dir_id, "b.pdf")
            db.update_file_status(fid, "success")
            assert db.count_files("pending") == 1
            assert db.count_files("success") == 1
            assert db.count_files("failed") == 0

    def test_count_files_empty(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db.count_files() == 0

    def test_mark_file_skipped(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "skip.pdf")
            db.mark_file_skipped(fid, reason="unsupported_format", detail=".doc not supported")

            f = db.get_file_by_id(fid)
            assert f["status"] == "skipped"

            # Verify skipped record exists
            rows = db.conn.execute(
                "SELECT * FROM skipped WHERE file_id = ?", (fid,)
            ).fetchall()
            assert len(rows) == 1
            assert dict(rows[0])["reason"] == "unsupported_format"
            assert dict(rows[0])["detail"] == ".doc not supported"

    def test_full_status_lifecycle(self, tmp_path):
        """Test: pending → converting → success → re-detect change → pending."""
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id, "cycle.pdf",
                               source_mtime="2024-01-01", source_hash="h1")
            f = db.get_file("cycle.pdf")
            assert f["status"] == "pending"

            db.update_file_status(fid, "converting")
            assert db.get_file_by_id(fid)["status"] == "converting"

            db.update_file_status(fid, "success", converter="test")
            assert db.get_file_by_id(fid)["status"] == "success"

            # File changed — re-upsert with different hash
            db.upsert_file(
                relative_path="cycle.pdf", directory_id=dir_id,
                filename="cycle.pdf", extension=".pdf",
                file_size=100, source_mtime="2024-01-01", source_hash="h2",
            )
            assert db.get_file("cycle.pdf")["status"] == "pending"


# ── Batches ──────────────────────────────────────────────────────────


class TestConvertDBBatches:
    """Test batch lifecycle operations."""

    def test_create_batch(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            assert isinstance(bid, int)
            assert bid > 0

    def test_create_batch_with_config(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            config = {"converter": "pdf", "ocr": True}
            bid = db.create_batch("incremental", total_files=5, config=config)
            row = db.conn.execute("SELECT config_json FROM batches WHERE id = ?", (bid,)).fetchone()
            import json
            assert json.loads(row["config_json"]) == config

    def test_create_batch_no_config(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=3)
            row = db.conn.execute("SELECT config_json FROM batches WHERE id = ?", (bid,)).fetchone()
            assert row["config_json"] is None

    def test_update_batch_progress(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            db.update_batch_progress(bid, processed=5, success=4, failed=1, skipped=0)
            b = db.conn.execute("SELECT * FROM batches WHERE id = ?", (bid,)).fetchone()
            assert b["processed"] == 5
            assert b["success_count"] == 4
            assert b["failed_count"] == 1

    def test_complete_batch_default(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            db.complete_batch(bid)
            b = db.conn.execute("SELECT * FROM batches WHERE id = ?", (bid,)).fetchone()
            assert b["status"] == "completed"
            assert b["finished_at"] is not None

    def test_complete_batch_failed(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            db.complete_batch(bid, status="failed")
            b = db.conn.execute("SELECT * FROM batches WHERE id = ?", (bid,)).fetchone()
            assert b["status"] == "failed"

    def test_get_active_batch_running(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            active = db.get_active_batch()
            assert active is not None
            assert active["id"] == bid
            assert active["status"] == "running"

    def test_get_active_batch_interrupted(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            db.complete_batch(bid, status="interrupted")
            active = db.get_active_batch()
            assert active is not None
            assert active["status"] == "interrupted"

    def test_get_active_batch_none_when_completed(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            db.complete_batch(bid, status="completed")
            assert db.get_active_batch() is None

    def test_get_active_batch_returns_latest(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            db.create_batch("full", total_files=5)
            bid2 = db.create_batch("incremental", total_files=3)
            active = db.get_active_batch()
            assert active["id"] == bid2

    def test_get_latest_batch(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid1 = db.create_batch("full", total_files=5)
            bid2 = db.create_batch("incremental", total_files=3)
            latest = db.get_latest_batch()
            assert latest["id"] == bid2

    def test_get_latest_batch_none(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            assert db.get_latest_batch() is None

    def test_mark_interrupted_batches(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            bid = db.create_batch("full", total_files=3)
            f1 = _insert_file(db, dir_id, "f1.pdf")
            f2 = _insert_file(db, dir_id, "f2.pdf")
            db.update_file_status(f1, "converting")
            db.update_file_status(f2, "success")

            db.mark_interrupted_batches()

            # Batch should be interrupted
            b = db.conn.execute("SELECT * FROM batches WHERE id = ?", (bid,)).fetchone()
            assert b["status"] == "interrupted"

            # Converting file reset to pending
            assert db.get_file_by_id(f1)["status"] == "pending"
            # Success file unaffected
            assert db.get_file_by_id(f2)["status"] == "success"

    def test_mark_interrupted_batches_multiple(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            db.create_batch("full", total_files=10)
            db.create_batch("incremental", total_files=5)
            db.mark_interrupted_batches()

            running = db.conn.execute(
                "SELECT COUNT(*) as cnt FROM batches WHERE status = 'running'"
            ).fetchone()["cnt"]
            assert running == 0

            interrupted = db.conn.execute(
                "SELECT COUNT(*) as cnt FROM batches WHERE status = 'interrupted'"
            ).fetchone()["cnt"]
            assert interrupted == 2

    def test_batch_defaults(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=10)
            b = db.conn.execute("SELECT * FROM batches WHERE id = ?", (bid,)).fetchone()
            assert b["status"] == "running"
            assert b["processed"] == 0
            assert b["success_count"] == 0
            assert b["failed_count"] == 0
            assert b["skipped_count"] == 0
            assert b["finished_at"] is None

    def test_full_batch_lifecycle(self, tmp_path):
        """End-to-end: create → progress → complete."""
        with ConvertDB(tmp_path / "test.db") as db:
            bid = db.create_batch("full", total_files=3, config={"mode": "test"})
            assert db.get_active_batch()["id"] == bid

            db.update_batch_progress(bid, processed=1, success=1, failed=0, skipped=0)
            db.update_batch_progress(bid, processed=2, success=2, failed=0, skipped=0)
            db.update_batch_progress(bid, processed=3, success=2, failed=1, skipped=0)
            db.complete_batch(bid)

            assert db.get_active_batch() is None
            latest = db.get_latest_batch()
            assert latest["status"] == "completed"
            assert latest["processed"] == 3


# ── Token Usage ──────────────────────────────────────────────────────


class TestConvertDBTokenUsage:
    """Test token usage tracking."""

    def test_add_token_usage(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id)
            db.add_token_usage(fid, model="glm-4", input_tokens=100, output_tokens=50,
                               total_tokens=150, call_type="ocr")
            rows = db.conn.execute("SELECT * FROM token_usage").fetchall()
            assert len(rows) == 1
            r = dict(rows[0])
            assert r["model"] == "glm-4"
            assert r["input_tokens"] == 100
            assert r["total_tokens"] == 150
            assert r["call_type"] == "ocr"

    def test_add_token_usage_multiple(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid1 = _insert_file(db, dir_id, "a.pdf")
            fid2 = _insert_file(db, dir_id, "b.pdf")
            db.add_token_usage(fid1, "glm-4", 100, 50, 150, "ocr")
            db.add_token_usage(fid2, "glm-4", 200, 100, 300, "llm")
            rows = db.conn.execute("SELECT * FROM token_usage").fetchall()
            assert len(rows) == 2

    def test_get_token_summary_empty(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            summary = db.get_token_summary()
            assert summary["input_tokens"] == 0
            assert summary["output_tokens"] == 0
            assert summary["total_tokens"] == 0
            assert summary["by_model"] == []

    def test_get_token_summary(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id)
            db.add_token_usage(fid, "glm-4", 100, 50, 150, "ocr")
            db.add_token_usage(fid, "glm-4", 200, 80, 280, "llm")
            db.add_token_usage(fid, "glm-4-flash", 50, 20, 70, "ocr")

            summary = db.get_token_summary()
            assert summary["input_tokens"] == 350
            assert summary["output_tokens"] == 150
            assert summary["total_tokens"] == 500

            by_model = {m["model"]: m for m in summary["by_model"]}
            assert by_model["glm-4"]["total_tokens"] == 430
            assert by_model["glm-4"]["call_count"] == 2
            assert by_model["glm-4-flash"]["total_tokens"] == 70
            assert by_model["glm-4-flash"]["call_count"] == 1

    def test_token_usage_defaults(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id)
            db.add_token_usage(fid, model="test-model")
            rows = db.conn.execute("SELECT * FROM token_usage").fetchall()
            r = dict(rows[0])
            assert r["input_tokens"] == 0
            assert r["output_tokens"] == 0
            assert r["total_tokens"] == 0
            assert r["call_type"] == "ocr"


# ── Stats ────────────────────────────────────────────────────────────


class TestConvertDBStats:
    """Test get_stats aggregation."""

    def test_stats_empty_db(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            stats = db.get_stats()
            assert stats["directory_count"] == 0
            assert stats["file_total"] == 0
            assert stats["status_counts"] == {}
            assert stats["batch_count"] == 0
            assert stats["skipped_count"] == 0
            assert stats["latest_batch"] is None

    def test_stats_with_data(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db, "docs")
            _insert_dir(db, "docs/sub")

            f1 = _insert_file(db, dir_id, "a.pdf")
            f2 = _insert_file(db, dir_id, "b.pdf")
            f3 = _insert_file(db, dir_id, "c.pdf")
            db.update_file_status(f1, "success")
            db.update_file_status(f2, "failed", last_error="timeout")
            db.mark_file_skipped(f3, "unsupported_format")

            db.create_batch("full", total_files=3)

            stats = db.get_stats()
            assert stats["directory_count"] == 2
            assert stats["file_total"] == 3
            assert stats["status_counts"]["success"] == 1
            assert stats["status_counts"]["failed"] == 1
            assert stats["status_counts"]["skipped"] == 1
            assert stats["batch_count"] == 1
            assert stats["skipped_count"] == 1
            assert stats["latest_batch"] is not None
            assert stats["latest_batch"]["batch_type"] == "full"

    def test_stats_multiple_batches(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            db.create_batch("full", total_files=10)
            db.create_batch("incremental", total_files=5)
            stats = db.get_stats()
            assert stats["batch_count"] == 2
            assert stats["latest_batch"]["batch_type"] == "incremental"


# ── Edge Cases ───────────────────────────────────────────────────────


class TestConvertDBEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_many_directories(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            parent_id = _insert_dir(db, "root")
            child_ids = []
            for i in range(50):
                cid = db.upsert_directory(
                    f"root/sub{i}", parent_id=parent_id, depth=1, name=f"sub{i}"
                )
                child_ids.append(cid)
            children = db.list_subdirectories(parent_id)
            assert len(children) == 50

    def test_many_files(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            for i in range(100):
                _insert_file(db, dir_id, f"file{i:04d}.pdf")
            assert db.count_files() == 100
            assert db.count_files("pending") == 100

    def test_unicode_paths(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = db.upsert_directory("公司制度/人事管理", depth=1, name="人事管理")
            fid = db.upsert_file(
                relative_path="公司制度/人事管理/年假规定.pdf",
                directory_id=dir_id, filename="年假规定.pdf",
                extension=".pdf", file_size=2048,
                source_mtime="2024-01-01", source_hash="hash",
            )
            assert db.get_directory("公司制度/人事管理")["name"] == "人事管理"
            assert db.get_file("公司制度/人事管理/年假规定.pdf")["filename"] == "年假规定.pdf"

    def test_update_file_status_with_metadata_json(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id)
            import json
            meta = {"pages": 10, "has_images": True}
            db.update_file_status(fid, "success", metadata_json=json.dumps(meta))
            f = db.get_file_by_id(fid)
            assert json.loads(f["metadata_json"]) == meta

    def test_update_file_status_with_ocr_tokens(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            fid = _insert_file(db, dir_id)
            db.update_file_status(
                fid, "success",
                ocr_input_tokens=100, ocr_output_tokens=50, ocr_total_tokens=150,
            )
            f = db.get_file_by_id(fid)
            assert f["ocr_input_tokens"] == 100
            assert f["ocr_output_tokens"] == 50
            assert f["ocr_total_tokens"] == 150

    def test_foreign_key_constraint_files_to_dirs(self, tmp_path):
        """Files require a valid directory_id (FK enforced)."""
        with ConvertDB(tmp_path / "test.db") as db, pytest.raises(sqlite3.IntegrityError):
            db.upsert_file(
                relative_path="orphan.pdf", directory_id=9999,
                filename="orphan.pdf", extension=".pdf",
                file_size=100, source_mtime="2024-01-01", source_hash="h",
            )

    def test_foreign_key_constraint_skipped_to_files(self, tmp_path):
        """Skipped requires a valid file_id (FK enforced)."""
        with ConvertDB(tmp_path / "test.db") as db, pytest.raises(sqlite3.IntegrityError):
            db.mark_file_skipped(9999, "test_reason")

    def test_duplicate_directory_path_unique(self, tmp_path):
        """Inserting same relative_path twice uses upsert, no error."""
        with ConvertDB(tmp_path / "test.db") as db:
            id1 = db.upsert_directory("dup", depth=0, name="dup")
            id2 = db.upsert_directory("dup", depth=1, name="dup_v2")
            assert id1 == id2

    def test_duplicate_file_path_unique(self, tmp_path):
        """Inserting same relative_path twice uses upsert, no error."""
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            id1 = _insert_file(db, dir_id, "dup.pdf")
            id2 = _insert_file(db, dir_id, "dup.pdf")
            assert id1 == id2

    def test_get_files_by_directory_ordered_by_filename(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            _insert_file(db, dir_id, "c.pdf", filename="c.pdf")
            _insert_file(db, dir_id, "a.pdf", filename="a.pdf")
            _insert_file(db, dir_id, "b.pdf", filename="b.pdf")
            files = db.get_files_by_directory(dir_id)
            names = [f["filename"] for f in files]
            assert names == ["a.pdf", "b.pdf", "c.pdf"]

    def test_get_files_by_extension_ordered_by_filename(self, tmp_path):
        with ConvertDB(tmp_path / "test.db") as db:
            dir_id = _insert_dir(db)
            _insert_file(db, dir_id, "z.pdf", extension=".pdf")
            _insert_file(db, dir_id, "a.pdf", extension=".pdf")
            files = db.get_files_by_extension(".pdf")
            names = [f["filename"] for f in files]
            assert names == ["a.pdf", "z.pdf"]

    def test_reopen_preserves_data(self, tmp_path):
        """Closing and reopening the DB should preserve all data."""
        db_path = tmp_path / "persist.db"
        with ConvertDB(db_path) as db:
            dir_id = _insert_dir(db, "persist_dir")
            _insert_file(db, dir_id, "persist.pdf")
            bid = db.create_batch("full", total_files=1)

        # Reopen and verify
        with ConvertDB(db_path) as db:
            assert db.get_directory("persist_dir") is not None
            assert db.get_file("persist.pdf") is not None
            assert db.get_latest_batch()["id"] == bid
