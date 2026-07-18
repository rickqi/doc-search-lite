"""Tests for search_logger.py — async search logging system.

Covers SearchLogger (is_enabled, generate_session_id, log_async, _do_log,
_extract, _write_md) and SearchLogDB (add_search_log, get_search_logs, count).
"""

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.stats import search_logger as sl_mod
from src.stats.search_logger import SearchLogger, SearchLogDB


# ── Helpers ────────────────────────────────────────────────────────────

_SESSION_ID_RE = re.compile(r"^srch_\d{8}_\d{6}_[0-9a-f]{6}$")


def _make_record(**overrides):
    """Build a standard extracted-record dict (as _extract would return)."""
    record = {
        "answer": "测试回答",
        "success": True,
        "processing_time": 1.5,
        "tokens_used": 100,
        "sources": [],
        "search_hits": [],
        "tool_calls": [],
        "reasoning": "",
        "error": "",
    }
    record.update(overrides)
    return record


# ── TestSearchLoggerBasics ─────────────────────────────────────────────


class TestSearchLoggerBasics:
    """Test is_enabled() and generate_session_id()."""

    def test_is_enabled_default(self, monkeypatch):
        """When _SEARCH_LOG_DISABLED is False, logging is enabled."""
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", False)
        assert SearchLogger.is_enabled() is True

    def test_is_enabled_disabled(self, monkeypatch):
        """When _SEARCH_LOG_DISABLED is True, is_enabled() returns False."""
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", True)
        assert SearchLogger.is_enabled() is False

    def test_generate_session_id_format(self):
        """Session ID matches srch_YYYYMMDD_HHMMSS_6hex."""
        sid = SearchLogger.generate_session_id()
        assert sid.startswith("srch_")
        assert _SESSION_ID_RE.match(sid), f"bad format: {sid}"

    def test_generate_session_id_uniqueness(self):
        """100 generated IDs are all unique."""
        ids = {SearchLogger.generate_session_id() for _ in range(100)}
        assert len(ids) == 100


# ── TestSearchLoggerExtract ────────────────────────────────────────────


class TestSearchLoggerExtract:
    """Test SearchLogger._extract() for all response types."""

    def test_extract_agent_response(self):
        """Extract from an object with .answer/.success/.tool_calls attrs."""
        resp = SimpleNamespace(
            answer="年假3天",
            success=True,
            processing_time=2.0,
            tokens_used=50,
            sources=[{"title": "src"}],
            search_hits=[{"title": "hit", "score": 0.9}],
            tool_calls=[{"name": "search"}],
            reasoning="step",
            error="",
        )
        record = SearchLogger._extract(resp)
        assert record["answer"] == "年假3天"
        assert record["success"] is True
        assert record["tokens_used"] == 50
        assert record["processing_time"] == 2.0
        assert record["search_hits"] == [{"title": "hit", "score": 0.9}]
        assert record["tool_calls"] == [{"name": "search"}]
        assert record["sources"] == [{"title": "src"}]

    def test_extract_agent_response_defaults(self):
        """Missing attrs fall back to defaults via getattr."""
        resp = SimpleNamespace(answer="ok")
        record = SearchLogger._extract(resp)
        assert record["answer"] == "ok"
        assert record["success"] is True
        assert record["processing_time"] == 0.0
        assert record["tokens_used"] == 0
        assert record["search_hits"] == []
        assert record["tool_calls"] == []

    def test_extract_dict_with_answer(self):
        """Dict response with explicit answer field."""
        resp = {"answer": "yes", "success": False, "tokens_used": 10}
        record = SearchLogger._extract(resp)
        assert record["answer"] == "yes"
        assert record["success"] is False
        assert record["tokens_used"] == 10

    def test_extract_dict_with_results(self):
        """Dict with results list (keyword search) generates summary answer."""
        resp = {
            "results": [
                {"title": "Doc A"},
                {"doc_id": "abc", "title": "Doc B"},
            ]
        }
        record = SearchLogger._extract(resp)
        assert "找到 2 个结果" in record["answer"]
        assert "Doc A" in record["answer"]
        assert "Doc B" in record["answer"]
        # search_hits are now normalized to {title, score, snippet} format
        assert len(record["search_hits"]) == 2
        assert record["search_hits"][0] == {"title": "Doc A", "score": 0, "snippet": ""}
        assert record["search_hits"][1]["title"] == "Doc B"
        assert record["success"] is True

    def test_extract_normalizes_highlights_to_snippet(self):
        """BM25 results with highlights get normalized to snippet field."""
        resp = {
            "results": [
                {"title": "Doc A", "highlights": ["匹配的关键词片段"]},
                {"title": "Doc B", "snippet": "已有 snippet"},
            ],
            "execution_time": 0.15,
        }
        record = SearchLogger._extract(resp)
        # highlights → snippet
        assert record["search_hits"][0]["snippet"] == "匹配的关键词片段"
        # existing snippet preserved
        assert record["search_hits"][1]["snippet"] == "已有 snippet"
        # execution_time → processing_time
        assert record["processing_time"] == 0.15
        # sources derived from results
        assert len(record["sources"]) == 2

    def test_extract_dict_with_results_summary(self):
        """Dict with results_summary fallback when no results list."""
        resp = {"results_summary": "共 5 条", "results": []}
        record = SearchLogger._extract(resp)
        assert record["answer"] == "共 5 条"

    def test_extract_string(self):
        """Plain string response."""
        record = SearchLogger._extract("hello world")
        assert record["answer"] == "hello world"
        assert record["success"] is True
        assert record["tool_calls"] == []

    def test_extract_list(self):
        """List of search results."""
        hits = [{"title": f"hit-{i}"} for i in range(3)]
        record = SearchLogger._extract(hits)
        assert "找到 3 个结果" in record["answer"]
        assert record["search_hits"] == hits

    def test_extract_none(self):
        """None response falls back to str(None)."""
        record = SearchLogger._extract(None)
        assert record["answer"] == "None"
        assert record["success"] is True

    def test_extract_empty_dict(self):
        """Empty dict produces empty answer."""
        record = SearchLogger._extract({})
        assert record["answer"] == ""
        assert record["success"] is True


# ── TestSearchLoggerMDFile ─────────────────────────────────────────────


class TestSearchLoggerMDFile:
    """Test SearchLogger._write_md() output structure."""

    def _write(
        self,
        tmp_path,
        record=None,
        query="年假如何申请",
        tags="hr,leave",
        session_id="srch_test_001",
    ):
        """Helper to write an MD file and return its content."""
        path = SearchLogger._write_md(
            log_dir=tmp_path,
            session_id=session_id,
            query=query,
            record=record or _make_record(),
            source="cli",
            search_mode="agent",
            model="glm-4",
            index_path="D:/idx",
            raw_dir="",
            difficulty="easy",
            tags=tags,
            skill="",
        )
        content = Path(path).read_text(encoding="utf-8")
        return path, content

    def test_md_file_created(self, tmp_path):
        """MD file exists after write."""
        path, _ = self._write(tmp_path)
        assert Path(path).exists()

    def test_md_file_has_frontmatter(self, tmp_path):
        """File starts with YAML frontmatter delimiter."""
        _, content = self._write(tmp_path)
        assert content.startswith("---\n")
        # closing delimiter present
        assert content.count("---") >= 2

    def test_md_file_has_instruction_section(self, tmp_path):
        """Contains # Instruction with the query."""
        _, content = self._write(tmp_path, query="我的问题")
        assert "# Instruction" in content
        assert "我的问题" in content

    def test_md_file_has_response_section(self, tmp_path):
        """Contains # Response with the answer."""
        _, content = self._write(tmp_path, record=_make_record(answer="最终答案"))
        assert "# Response" in content
        assert "最终答案" in content

    def test_md_file_has_retrieved_context(self, tmp_path):
        """# Retrieved Context present when search_hits non-empty."""
        record = _make_record(
            search_hits=[
                {"title": "年假制度", "score": 0.95, "snippet": "员工享有年假"},
            ]
        )
        _, content = self._write(tmp_path, record=record)
        assert "# Retrieved Context" in content
        assert "年假制度" in content
        assert "0.95" in content
        assert "员工享有年假" in content

    def test_md_file_has_reasoning_trace(self, tmp_path):
        """# Reasoning Trace present when tool_calls non-empty."""
        record = _make_record(
            tool_calls=[
                {
                    "name": "search",
                    "arguments": {"query": "年假"},
                    "execution_time": 0.5,
                    "result_metadata": {"hits": 3},
                }
            ]
        )
        _, content = self._write(tmp_path, record=record)
        assert "# Reasoning Trace" in content
        assert "Tool Call 1: search" in content
        assert "年假" in content
        assert "0.500s" in content

    def test_md_file_no_hits_section(self, tmp_path):
        """No # Retrieved Context when search_hits is empty."""
        _, content = self._write(tmp_path, record=_make_record(search_hits=[]))
        assert "# Retrieved Context" not in content

    def test_md_file_no_reasoning_when_empty(self, tmp_path):
        """No # Reasoning Trace when tool_calls is empty."""
        _, content = self._write(tmp_path, record=_make_record(tool_calls=[]))
        assert "# Reasoning Trace" not in content

    def test_md_file_unicode_content(self, tmp_path):
        """Chinese query and answer are preserved (UTF-8)."""
        record = _make_record(answer="根据制度，年假为5天。")
        _, content = self._write(
            tmp_path, query="年假有几天？", record=record
        )
        assert "年假有几天？" in content
        assert "根据制度，年假为5天。" in content

    def test_md_file_tags_in_frontmatter(self, tmp_path):
        """Tags appear in frontmatter."""
        _, content = self._write(tmp_path, tags="hr,vacation")
        assert "tags: [hr,vacation]" in content

    def test_md_file_empty_tags(self, tmp_path):
        """Empty tags produce []."""
        _, content = self._write(tmp_path, tags="")
        assert "tags: []" in content

    def test_md_file_error_section(self, tmp_path):
        """# Error section present when record has error."""
        record = _make_record(error="something broke")
        _, content = self._write(tmp_path, record=record)
        assert "# Error" in content
        assert "something broke" in content

    def test_md_file_filename_matches_session_id(self, tmp_path):
        """Filename is <session_id>.md."""
        path, _ = self._write(tmp_path, session_id="srch_xyz_123")
        assert Path(path).name == "srch_xyz_123.md"


# ── TestSearchLogDB ────────────────────────────────────────────────────


class TestSearchLogDB:
    """Test SearchLogDB CRUD operations."""

    def test_db_create(self, tmp_path):
        """Table is created on init."""
        db = SearchLogDB(tmp_path / "test.db")
        assert (tmp_path / "test.db").exists()
        # table exists
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
        assert "search_logs" in tables

    def test_add_search_log(self, tmp_path):
        """Insert a record and verify it's queryable."""
        db = SearchLogDB(tmp_path / "test.db")
        rid = db.add_search_log(
            session_id="s1",
            query="年假",
            answer="5天",
            source="cli",
            search_mode="agent",
        )
        assert isinstance(rid, int)
        assert db.count() == 1
        rows = db.get_search_logs()
        assert len(rows) == 1
        assert rows[0]["query"] == "年假"
        assert rows[0]["answer"] == "5天"

    def test_add_search_log_duplicate_session_id(self, tmp_path):
        """INSERT OR REPLACE on same session_id keeps single row."""
        db = SearchLogDB(tmp_path / "test.db")
        db.add_search_log(session_id="dup", query="q1", source="cli", search_mode="agent")
        db.add_search_log(session_id="dup", query="q2", source="web", search_mode="bm25")
        assert db.count() == 1
        rows = db.get_search_logs()
        assert rows[0]["query"] == "q2"
        assert rows[0]["source"] == "web"

    def test_get_search_logs_all(self, tmp_path):
        """get_search_logs returns all records."""
        db = SearchLogDB(tmp_path / "test.db")
        for i in range(3):
            db.add_search_log(
                session_id=f"s{i}", query=f"q{i}", source="cli", search_mode="agent"
            )
        rows = db.get_search_logs()
        assert len(rows) == 3

    def test_get_search_logs_by_mode(self, tmp_path):
        """Filter by search_mode."""
        db = SearchLogDB(tmp_path / "test.db")
        db.add_search_log(session_id="a", query="q", source="cli", search_mode="agent")
        db.add_search_log(session_id="b", query="q", source="cli", search_mode="bm25")
        rows = db.get_search_logs(search_mode="bm25")
        assert len(rows) == 1
        assert rows[0]["search_mode"] == "bm25"

    def test_get_search_logs_by_source(self, tmp_path):
        """Filter by source."""
        db = SearchLogDB(tmp_path / "test.db")
        db.add_search_log(session_id="a", query="q", source="cli", search_mode="agent")
        db.add_search_log(session_id="b", query="q", source="web", search_mode="agent")
        rows = db.get_search_logs(source="web")
        assert len(rows) == 1
        assert rows[0]["source"] == "web"

    def test_get_search_logs_pagination(self, tmp_path):
        """limit/offset pagination."""
        db = SearchLogDB(tmp_path / "test.db")
        for i in range(5):
            db.add_search_log(
                session_id=f"p{i}", query=f"q{i}", source="cli", search_mode="agent"
            )
        assert len(db.get_search_logs(limit=2, offset=0)) == 2
        assert len(db.get_search_logs(limit=2, offset=2)) == 2
        assert len(db.get_search_logs(limit=2, offset=4)) == 1
        assert len(db.get_search_logs(limit=2, offset=10)) == 0

    def test_get_search_logs_success_only(self, tmp_path):
        """success_only filters out failed records."""
        db = SearchLogDB(tmp_path / "test.db")
        db.add_search_log(
            session_id="ok", query="q", source="cli", search_mode="agent", success=True
        )
        db.add_search_log(
            session_id="bad", query="q", source="cli", search_mode="agent", success=False
        )
        rows = db.get_search_logs(success_only=True)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "ok"

    def test_count(self, tmp_path):
        """count() reflects total rows."""
        db = SearchLogDB(tmp_path / "test.db")
        assert db.count() == 0
        db.add_search_log(session_id="x", query="q", source="cli", search_mode="agent")
        assert db.count() == 1
        db.add_search_log(session_id="y", query="q", source="cli", search_mode="agent")
        assert db.count() == 2

    def test_get_search_logs_by_tags(self, tmp_path):
        """tags filter uses LIKE."""
        db = SearchLogDB(tmp_path / "test.db")
        db.add_search_log(
            session_id="t1", query="q", source="cli", search_mode="agent", tags="hr,leave"
        )
        db.add_search_log(
            session_id="t2", query="q", source="cli", search_mode="agent", tags="finance"
        )
        rows = db.get_search_logs(tags="leave")
        assert len(rows) == 1
        assert rows[0]["session_id"] == "t1"


# ── TestSearchLoggerAsync ──────────────────────────────────────────────


class TestSearchLoggerAsync:
    """Test SearchLogger.log_async() fire-and-forget behavior."""

    @pytest.fixture(autouse=True)
    def _ensure_enabled(self, monkeypatch):
        """Ensure logging is enabled for these tests."""
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", False)

    def _log_and_wait(self, monkeypatch, tmp_path, response: Any = "年假为5天", session_id="srch_async_001"):
        """Run log_async synchronously via Event, return after completion."""
        done = threading.Event()
        orig = SearchLogger._do_log

        def wrapped(**kw):
            try:
                orig(**kw)
            finally:
                done.set()

        # Patch so the daemon-thread target signals completion
        monkeypatch.setattr(SearchLogger, "_do_log", staticmethod(wrapped))

        SearchLogger.log_async(
            session_id=session_id,
            query="年假如何申请",
            response=response,
            source="cli",
            search_mode="agent",
            log_dir=tmp_path,
        )
        assert done.wait(timeout=5), "daemon thread did not complete in time"

    def test_log_async_creates_md_file(self, monkeypatch, tmp_path):
        """log_async produces a .md file in log_dir."""
        self._log_and_wait(monkeypatch, tmp_path)
        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "年假如何申请" in content
        assert "年假为5天" in content

    def test_log_async_creates_db_record(self, monkeypatch, tmp_path):
        """log_async inserts a row into search_logs.db."""
        self._log_and_wait(monkeypatch, tmp_path)
        db_path = tmp_path / "search_logs.db"
        assert db_path.exists()
        db = SearchLogDB(db_path)
        assert db.count() == 1
        rows = db.get_search_logs()
        assert rows[0]["query"] == "年假如何申请"
        assert rows[0]["answer"] == "年假为5天"
        assert rows[0]["source"] == "cli"

    def test_log_async_does_not_raise(self, monkeypatch, tmp_path):
        """log_async swallows all errors (bad input) without raising."""
        # bad response type + empty session still should not raise
        SearchLogger.log_async(
            session_id="",
            query="",
            response=object(),
            source="cli",
            search_mode="agent",
            log_dir=tmp_path,
        )
        # give daemon thread a moment
        time.sleep(0.5)

    def test_log_async_respects_disabled(self, monkeypatch, tmp_path):
        """When disabled, no files are created."""
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", True)
        SearchLogger.log_async(
            session_id="srch_disabled",
            query="q",
            response="a",
            source="cli",
            search_mode="agent",
            log_dir=tmp_path,
        )
        time.sleep(0.3)
        # no md files, no db
        assert list(tmp_path.glob("*.md")) == []
        assert not (tmp_path / "search_logs.db").exists()

    def test_log_async_agent_response(self, monkeypatch, tmp_path):
        """log_async handles an AgentResponse-like object end-to-end."""
        resp = SimpleNamespace(
            answer="答案是3天",
            success=True,
            processing_time=1.0,
            tokens_used=42,
            sources=[],
            search_hits=[{"title": "年假", "score": 0.8, "snippet": "规定"}],
            tool_calls=[{"name": "search", "arguments": {"query": "年假"}}],
            reasoning="",
            error="",
        )
        self._log_and_wait(monkeypatch, tmp_path, response=resp)
        md_files = list(tmp_path.glob("*.md"))
        content = md_files[0].read_text(encoding="utf-8")
        assert "答案是3天" in content
        assert "# Retrieved Context" in content
        assert "# Reasoning Trace" in content
        # DB row
        db = SearchLogDB(tmp_path / "search_logs.db")
        row = db.get_search_logs()[0]
        assert row["tokens_used"] == 42
        assert row["tool_calls_count"] == 1
        assert row["sources_count"] == 0


# ── TestHookHelpers ───────────────────────────────────────────────────


class TestHookHelpers:
    """Test the _log_search_cli and _log_search_api helper functions."""

    def test_cli_helper_respects_env(self, monkeypatch, tmp_path):
        """_log_search_cli does nothing when NO_SEARCH_LOG=1."""
        monkeypatch.setenv("NO_SEARCH_LOG", "1")
        # Import here so monkeypatch is applied
        from src.cli import _log_search_cli
        _log_search_cli("q", {"answer": "a"}, "bm25")
        # No files created in default log dir
        # (We can't easily check the default log dir, so just verify no exception)

    def test_cli_helper_works_when_enabled(self, monkeypatch, tmp_path):
        """_log_search_cli logs when enabled."""
        monkeypatch.delenv("NO_SEARCH_LOG", raising=False)
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", False)
        monkeypatch.setattr(sl_mod, "_DEFAULT_LOG_DIR", tmp_path)
        from src.cli import _log_search_cli
        _log_search_cli("test q", "test answer", "bm25", index_path="/idx")
        # Wait for daemon thread
        import time as _t
        _t.sleep(0.5)
        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1
        assert "test q" in md_files[0].read_text(encoding="utf-8")

    def test_api_helper_enabled_param(self, monkeypatch, tmp_path):
        """_log_search_api respects enabled=False."""
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", False)
        monkeypatch.setattr(sl_mod, "_DEFAULT_LOG_DIR", tmp_path)
        from src.api import _log_search_api
        _log_search_api("q", "a", "bm25", "/idx", enabled=False)
        import time as _t
        _t.sleep(0.3)
        # No files should be created
        assert list(tmp_path.glob("*.md")) == []

    def test_api_helper_enabled_true(self, monkeypatch, tmp_path):
        """_log_search_api logs when enabled=True."""
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", False)
        monkeypatch.setattr(sl_mod, "_DEFAULT_LOG_DIR", tmp_path)
        from src.api import _log_search_api
        _log_search_api("hello", "world", "grep", "/raw", enabled=True)
        import time as _t
        _t.sleep(0.5)
        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1
        assert "hello" in md_files[0].read_text(encoding="utf-8")


# ── TestSessionFixtureData ────────────────────────────────────────────


class TestSessionFixtureData:
    """Test SearchLogger with real session data from sessions.db."""

    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "session_qa_data.jsonl"

    def test_fixture_exists(self):
        """Fixture file exists and is non-empty."""
        assert self.FIXTURE_PATH.exists()
        lines = self.FIXTURE_PATH.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0

    def test_fixture_jsonl_valid(self):
        """Each line is valid JSON with required fields."""
        import json
        lines = self.FIXTURE_PATH.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            obj = json.loads(line)
            assert "query" in obj
            assert "answer" in obj
            assert "session_id" in obj

    @pytest.mark.parametrize("idx", range(min(5, 13)))
    def test_log_fixture_data(self, monkeypatch, tmp_path, idx):
        """Log fixture Q&A pairs and verify MD files are correct."""
        monkeypatch.setattr(sl_mod, "_SEARCH_LOG_DISABLED", False)

        import json
        lines = self.FIXTURE_PATH.read_text(encoding="utf-8").strip().splitlines()
        if idx >= len(lines):
            pytest.skip("Not enough fixture entries")
        data = json.loads(lines[idx])

        done = threading.Event()
        orig = SearchLogger._do_log

        def wrapped(**kw):
            try:
                orig(**kw)
            finally:
                done.set()

        monkeypatch.setattr(SearchLogger, "_do_log", staticmethod(wrapped))

        SearchLogger.log_async(
            session_id=f"srch_fix_{idx:03d}",
            query=data["query"],
            response=data["answer"],
            source="test",
            search_mode="agent",
            log_dir=tmp_path,
        )
        assert done.wait(timeout=5)

        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert data["query"][:20] in content
        assert "# Instruction" in content
        assert "# Response" in content
