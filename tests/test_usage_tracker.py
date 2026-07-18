"""Tests for Phase 3 infrastructure: Schema v2.0 migration + UsageTracker."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.converter.base import ConvertResult
from src.converter.coordinator import ConverterCoordinator
from src.search.reranker import ZhipuAIReranker
from src.stats.usage_tracker import UsageTracker
from src.storage.convert_db import ConvertDB

# 鈹€鈹€ Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


def _open_db(tmp_path: Path) -> ConvertDB:
    """Create and open a fresh ConvertDB in tmp_path."""
    db = ConvertDB(tmp_path / "test.db")
    db.open()
    return db


def _insert_dir(db: ConvertDB, path: str = "root") -> int:
    return db.upsert_directory(relative_path=path, name=path.split("/")[-1])


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Schema v2.0 Migration
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class TestSchemaV2Migration:
    """Test schema v2.0 migration adds new columns, tables, and indexes."""

    def test_new_columns_exist_after_migration(self, tmp_path):
        """New columns should exist on token_usage after opening a fresh DB."""
        db = _open_db(tmp_path)
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(token_usage)").fetchall()}
        for col in ("cost_millicents", "source_dir", "batch_id", "session_id", "request_meta"):
            assert col in columns, f"Column {col} missing from token_usage"
        db.close()

    def test_pricing_table_created(self, tmp_path):
        """Pricing table should exist with correct columns."""
        db = _open_db(tmp_path)
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "pricing" in tables
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(pricing)").fetchall()}
        assert "model" in columns
        assert "input_price" in columns
        assert "output_price" in columns
        db.close()

    def test_pricing_defaults_inserted(self, tmp_path):
        """Default pricing data should be inserted."""
        db = _open_db(tmp_path)
        rows = db.conn.execute("SELECT model FROM pricing ORDER BY model").fetchall()
        models = [r[0] for r in rows]
        assert "glm-ocr" in models
        assert "zai/glm-4" in models
        assert "zai/glm-4-flash" in models
        assert "zai/glm-4-plus" in models
        assert "deepseek/deepseek-chat" in models
        assert "rerank" in models
        assert len(models) == 6
        db.close()

    def test_budget_table_created(self, tmp_path):
        """Budget table should exist with correct columns."""
        db = _open_db(tmp_path)
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "budget" in tables
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(budget)").fetchall()}
        assert "name" in columns
        assert "limit_cents" in columns
        assert "period" in columns
        db.close()

    def test_schema_version_is_20(self, tmp_path):
        """Schema version should be 2.0 after migration."""
        db = _open_db(tmp_path)
        version = db._get_config("schema_version")
        assert version == "2.1"
        db.close()

    def test_migration_idempotent(self, tmp_path):
        """Running migration twice should not fail (idempotent)."""
        db = _open_db(tmp_path)
        db.close()
        # Reopen triggers _migrate_schema again
        db = _open_db(tmp_path)
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(token_usage)").fetchall()}
        assert "cost_millicents" in columns
        # Pricing should still have exactly 6 rows (INSERT OR IGNORE)
        count = db.conn.execute("SELECT COUNT(*) FROM pricing").fetchone()[0]
        assert count == 6
        version = db._get_config("schema_version")
        assert version == "2.1"
        db.close()

    def test_indexes_created(self, tmp_path):
        """New indexes should exist on token_usage."""
        db = _open_db(tmp_path)
        indexes = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_token_%'"
            ).fetchall()
        }
        assert "idx_token_type" in indexes
        assert "idx_token_source" in indexes
        assert "idx_token_created" in indexes
        assert "idx_token_session" in indexes
        db.close()

    def test_migration_from_11(self, tmp_path):
        """A DB created at v1.1 should migrate to v2.0 successfully."""
        # Create a v1.1 DB manually
        db = ConvertDB(tmp_path / "test.db")
        db.open()
        # Force version to 1.1
        db._set_config("schema_version", "1.1")
        db.conn.commit()
        db.close()

        # Reopen 鈥?should trigger migration
        db = _open_db(tmp_path)
        version = db._get_config("schema_version")
        assert version == "2.1"
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(token_usage)").fetchall()}
        assert "cost_millicents" in columns
        db.close()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# ConvertDB New Methods
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class TestConvertDBNewMethods:
    """Test new methods added to ConvertDB for Phase 3."""

    def test_add_token_usage_extended(self, tmp_path):
        """add_token_usage_extended should insert a row and return its id."""
        db = _open_db(tmp_path)
        row_id = db.add_token_usage_extended(
            call_type="llm_chat",
            model="zai/glm-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_millicents=750,
            source_dir="test_dir",
            session_id="abc12345",
            request_meta='{"temperature": 0.7}',
        )
        assert row_id > 0
        row = db.conn.execute(
            "SELECT * FROM token_usage WHERE id = ?", (row_id,)
        ).fetchone()
        assert dict(row)["call_type"] == "llm_chat"
        assert dict(row)["model"] == "zai/glm-4"
        assert dict(row)["cost_millicents"] == 750
        assert dict(row)["source_dir"] == "test_dir"
        assert dict(row)["session_id"] == "abc12345"
        db.close()

    def test_add_token_usage_extended_minimal(self, tmp_path):
        """Minimal call with only required fields should work."""
        db = _open_db(tmp_path)
        row_id = db.add_token_usage_extended(call_type="ocr", model="glm-ocr")
        assert row_id > 0
        row = dict(db.conn.execute(
            "SELECT * FROM token_usage WHERE id = ?", (row_id,)
        ).fetchone())
        assert row["input_tokens"] == 0
        assert row["cost_millicents"] == 0
        assert row["source_dir"] is None
        db.close()

    def test_get_pricing_existing(self, tmp_path):
        """get_pricing should return data for a known model."""
        db = _open_db(tmp_path)
        result = db.get_pricing("glm-ocr")
        assert result is not None
        assert result["input_price"] == 0.005
        assert result["output_price"] == 0.005
        db.close()

    def test_get_pricing_unknown(self, tmp_path):
        """get_pricing should return None for unknown model."""
        db = _open_db(tmp_path)
        result = db.get_pricing("nonexistent-model")
        assert result is None
        db.close()

    def test_calculate_cost(self, tmp_path):
        """calculate_cost should compute correct millicents."""
        db = _open_db(tmp_path)
        # glm-ocr: input=0.005, output=0.005 per 1K tokens
        # 1000 input * 0.005 * 100 = 500, 500 output * 0.005 * 100 = 250
        cost = db.calculate_cost("glm-ocr", 1000, 500)
        assert cost == 750
        db.close()

    def test_calculate_cost_unknown_model(self, tmp_path):
        """calculate_cost should return 0 for unknown model."""
        db = _open_db(tmp_path)
        cost = db.calculate_cost("unknown", 1000, 500)
        assert cost == 0
        db.close()

    def test_get_token_usage_summary(self, tmp_path):
        """get_token_usage_summary should aggregate by call_type."""
        db = _open_db(tmp_path)
        db.add_token_usage_extended("ocr", "glm-ocr", input_tokens=100, total_tokens=100)
        db.add_token_usage_extended("ocr", "glm-ocr", input_tokens=200, total_tokens=200)
        db.add_token_usage_extended("llm_chat", "zai/glm-4", input_tokens=500, total_tokens=500)

        result = db.get_token_usage_summary()
        assert "ocr" in result["by_type"]
        assert result["by_type"]["ocr"]["call_count"] == 2
        assert result["by_type"]["ocr"]["input_tokens"] == 300
        assert "llm_chat" in result["by_type"]
        assert result["total"]["call_count"] == 3
        db.close()

    def test_get_token_usage_summary_with_source_dir(self, tmp_path):
        """get_token_usage_summary should filter by source_dir."""
        db = _open_db(tmp_path)
        db.add_token_usage_extended("ocr", "glm-ocr", input_tokens=100, source_dir="dir_a")
        db.add_token_usage_extended("ocr", "glm-ocr", input_tokens=200, source_dir="dir_b")

        result = db.get_token_usage_summary(source_dir="dir_a")
        assert result["total"]["call_count"] == 1
        assert result["total"]["input_tokens"] == 100
        db.close()

    def test_get_token_usage_daily(self, tmp_path):
        """get_token_usage_daily should return daily breakdown."""
        db = _open_db(tmp_path)
        db.add_token_usage_extended("ocr", "glm-ocr", input_tokens=100)
        db.add_token_usage_extended("llm_chat", "zai/glm-4", input_tokens=200)

        result = db.get_token_usage_daily(days=1)
        assert len(result) >= 1
        today_entry = result[0]
        assert today_entry["call_count"] == 2
        assert today_entry["input_tokens"] == 300
        db.close()

    def test_get_token_usage_by_model(self, tmp_path):
        """get_token_usage_by_model should aggregate by model."""
        db = _open_db(tmp_path)
        db.add_token_usage_extended("ocr", "glm-ocr", input_tokens=100, total_tokens=100)
        db.add_token_usage_extended("llm_chat", "zai/glm-4", input_tokens=500, output_tokens=50, total_tokens=550)

        result = db.get_token_usage_by_model()
        assert len(result) == 2
        # Ordered by total_tokens DESC
        assert result[0]["model"] == "zai/glm-4"
        assert result[0]["total_tokens"] == 550
        assert result[1]["model"] == "glm-ocr"
        assert result[1]["total_tokens"] == 100
        db.close()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# UsageTracker Core
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class TestUsageTracker:
    """Test UsageTracker class."""

    def _make_tracker(self, tmp_path, source_dir=None):
        """Create a fresh tracker with a real ConvertDB."""
        db = _open_db(tmp_path)
        tracker = UsageTracker(db, source_dir=source_dir)
        return tracker, db

    def test_start_session(self, tmp_path):
        """start_session should return a short session ID."""
        tracker, db = self._make_tracker(tmp_path)
        sid = tracker.start_session()
        assert sid is not None
        assert len(sid) == 8
        assert tracker.session_id == sid
        db.close()

    def test_start_session_generates_unique_ids(self, tmp_path):
        """Each start_session call should produce a new ID."""
        tracker, db = self._make_tracker(tmp_path)
        sid1 = tracker.start_session()
        sid2 = tracker.start_session()
        assert sid1 != sid2
        db.close()

    def test_record_llm(self, tmp_path):
        """record_llm should write an llm_chat record with auto cost."""
        tracker, db = self._make_tracker(tmp_path)
        tracker.start_session()
        row_id = tracker.record_llm(
            model="zai/glm-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            request_meta={"temperature": 0.7},
        )
        assert row_id > 0
        row = dict(db.conn.execute(
            "SELECT * FROM token_usage WHERE id = ?", (row_id,)
        ).fetchone())
        assert row["call_type"] == "llm_chat"
        assert row["model"] == "zai/glm-4"
        assert row["input_tokens"] == 100
        assert row["cost_millicents"] > 0
        assert row["session_id"] == tracker.session_id
        assert row["request_meta"] is not None
        meta = json.loads(row["request_meta"])
        assert meta["temperature"] == 0.7
        db.close()

    def test_record_ocr(self, tmp_path):
        """record_ocr should write an ocr record."""
        tracker, db = self._make_tracker(tmp_path, source_dir="my_docs")
        row_id = tracker.record_ocr(
            model="glm-ocr",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
        )
        assert row_id > 0
        row = dict(db.conn.execute(
            "SELECT * FROM token_usage WHERE id = ?", (row_id,)
        ).fetchone())
        assert row["call_type"] == "ocr"
        assert row["source_dir"] == "my_docs"
        assert row["cost_millicents"] > 0
        db.close()

    def test_record_rerank(self, tmp_path):
        """record_rerank should write a rerank record."""
        tracker, db = self._make_tracker(tmp_path)
        row_id = tracker.record_rerank(input_tokens=1000, total_tokens=1000)
        assert row_id > 0
        row = dict(db.conn.execute(
            "SELECT * FROM token_usage WHERE id = ?", (row_id,)
        ).fetchone())
        assert row["call_type"] == "rerank"
        assert row["model"] == "rerank"
        assert row["input_tokens"] == 1000
        db.close()

    def test_get_summary(self, tmp_path):
        """get_summary should return aggregated usage."""
        tracker, db = self._make_tracker(tmp_path, source_dir="test_src")
        tracker.record_llm(model="zai/glm-4", input_tokens=100, total_tokens=100)
        tracker.record_ocr(model="glm-ocr", input_tokens=200, total_tokens=200)

        summary = tracker.get_summary()
        assert summary["total"]["call_count"] == 2
        assert "llm_chat" in summary["by_type"]
        assert "ocr" in summary["by_type"]
        db.close()

    def test_get_daily(self, tmp_path):
        """get_daily should return today's usage."""
        tracker, db = self._make_tracker(tmp_path)
        tracker.record_llm(model="zai/glm-4", input_tokens=100)

        daily = tracker.get_daily(days=1)
        assert len(daily) >= 1
        assert daily[0]["call_count"] == 1
        db.close()

    def test_get_by_model(self, tmp_path):
        """get_by_model should return per-model aggregation."""
        tracker, db = self._make_tracker(tmp_path)
        tracker.record_llm(model="zai/glm-4", input_tokens=100, total_tokens=100)
        tracker.record_ocr(model="glm-ocr", input_tokens=200, total_tokens=200)

        by_model = tracker.get_by_model()
        assert len(by_model) == 2
        models = {r["model"] for r in by_model}
        assert "zai/glm-4" in models
        assert "glm-ocr" in models
        db.close()

    def test_source_dir_filtering(self, tmp_path):
        """Records should be associated with the tracker's source_dir."""
        tracker, db = self._make_tracker(tmp_path, source_dir="project_x")
        tracker.record_llm(model="zai/glm-4", input_tokens=100)

        # Unfiltered summary sees the record
        all_summary = db.get_token_usage_summary()
        assert all_summary["total"]["call_count"] == 1

        # Filtered summary for project_x sees it
        filtered = db.get_token_usage_summary(source_dir="project_x")
        assert filtered["total"]["call_count"] == 1

        # Filtered summary for other dir sees nothing
        other = db.get_token_usage_summary(source_dir="other_project")
        assert other["total"]["call_count"] == 0
        db.close()

    def test_session_id_tracking(self, tmp_path):
        """All records in a session should share the same session_id."""
        tracker, db = self._make_tracker(tmp_path)
        sid = tracker.start_session()
        id1 = tracker.record_llm(model="zai/glm-4", input_tokens=100)
        id2 = tracker.record_ocr(model="glm-ocr", input_tokens=50)

        row1 = dict(db.conn.execute(
            "SELECT session_id FROM token_usage WHERE id = ?", (id1,)
        ).fetchone())
        row2 = dict(db.conn.execute(
            "SELECT session_id FROM token_usage WHERE id = ?", (id2,)
        ).fetchone())
        assert row1["session_id"] == sid
        assert row2["session_id"] == sid
        db.close()

    def test_no_session_id_when_not_started(self, tmp_path):
        """Records without start_session should have NULL session_id."""
        tracker, db = self._make_tracker(tmp_path)
        assert tracker.session_id is None
        row_id = tracker.record_llm(model="zai/glm-4", input_tokens=100)
        row = dict(db.conn.execute(
            "SELECT session_id FROM token_usage WHERE id = ?", (row_id,)
        ).fetchone())
        assert row["session_id"] is None
        db.close()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# LLMClient Integration Tests
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class TestLLMClientIntegration:
    """Test LLMClient records usage via UsageTracker."""

    def test_chat_records_to_tracker(self, tmp_path):
        """chat() should call record_llm when usage_tracker is provided."""
        from unittest.mock import MagicMock

        from src.agent.llm_client import LLMClient
        from src.utils.config import Config

        db = _open_db(tmp_path)
        tracker = UsageTracker(db)

        config = Config(
            glm_api_key="test-key",
            glm_base_url="https://api.test.com/v1",
            llm_model="glm-4",
        )
        client = LLMClient(config, usage_tracker=tracker)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "zai/glm-4"

        with MagicMock() as mock_completion:
            from src.agent import llm_client as lc_mod
            original = lc_mod.completion
            lc_mod.completion = lambda **kw: mock_response
            try:
                result = client.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.5,
                )
            finally:
                lc_mod.completion = original

        assert result.usage["prompt_tokens"] == 10
        # Verify tracker recorded the call
        row = db.conn.execute("SELECT * FROM token_usage ORDER BY id DESC LIMIT 1").fetchone()
        assert dict(row)["call_type"] == "llm_chat"
        assert dict(row)["input_tokens"] == 10
        assert dict(row)["output_tokens"] == 20
        assert dict(row)["total_tokens"] == 30
        meta = json.loads(dict(row)["request_meta"])
        assert meta["temperature"] == 0.5
        db.close()

    def test_chat_no_tracker_works(self, tmp_path):
        """chat() should work without a tracker (backward compatible)."""
        from unittest.mock import MagicMock

        from src.agent.llm_client import LLMClient
        from src.utils.config import Config

        config = Config(
            glm_api_key="test-key",
            glm_base_url="https://api.test.com/v1",
            llm_model="glm-4",
        )
        client = LLMClient(config)  # No tracker

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_response.model = "zai/glm-4"

        from src.agent import llm_client as lc_mod
        original = lc_mod.completion
        lc_mod.completion = lambda **kw: mock_response
        try:
            result = client.chat(messages=[{"role": "user", "content": "hi"}])
        finally:
            lc_mod.completion = original

        assert result.content == "Hello"
        assert result.usage["total_tokens"] == 15

    def test_chat_with_tools_tracks_each_call(self, tmp_path):
        """chat_with_tools() should track each individual chat() call via tracker."""
        from unittest.mock import MagicMock

        from src.agent.base import Tool
        from src.agent.llm_client import LLMClient
        from src.utils.config import Config

        db = _open_db(tmp_path)
        tracker = UsageTracker(db)

        config = Config(
            glm_api_key="test-key",
            glm_base_url="https://api.test.com/v1",
            llm_model="glm-4",
        )
        client = LLMClient(config, usage_tracker=tracker)

        # First response: tool call; second: final answer
        call_count = [0]
        def fake_completion(**kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.model = "zai/glm-4"
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50
            resp.usage.total_tokens = 150
            if call_count[0] == 1:
                resp.choices[0].message.content = ""
                resp.choices[0].finish_reason = "tool_calls"
                tc = MagicMock()
                tc.id = "tc1"
                tc.function.name = "test_tool"
                tc.function.arguments = '{"query": "test"}'
                resp.choices[0].message.tool_calls = [tc]
            else:
                resp.choices[0].message.content = "Final answer"
                resp.choices[0].finish_reason = "stop"
                resp.choices[0].message.tool_calls = None
            return resp

        class FakeTool(Tool):
            @property
            def name(self):
                return "test_tool"
            @property
            def description(self):
                return "A test tool"
            def execute(self, **kw):
                return "tool result"
            def to_openai_tool(self):
                return {"type": "function", "function": {"name": "test_tool", "parameters": {}}}

        from src.agent import llm_client as lc_mod
        original = lc_mod.completion
        lc_mod.completion = fake_completion
        try:
            result = client.chat_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[FakeTool()],
                max_iterations=5,
            )
        finally:
            lc_mod.completion = original

        # Should have 2 LLM calls tracked
        rows = db.conn.execute("SELECT * FROM token_usage WHERE call_type='llm_chat'").fetchall()
        assert len(rows) == 2
        db.close()

    def test_chat_skips_tracking_on_empty_usage(self, tmp_path):
        """chat() should not call record_llm when usage is empty."""
        from unittest.mock import MagicMock

        from src.agent.llm_client import LLMClient
        from src.utils.config import Config

        db = _open_db(tmp_path)
        tracker = UsageTracker(db)

        config = Config(
            glm_api_key="test-key",
            glm_base_url="https://api.test.com/v1",
            llm_model="glm-4",
        )
        client = LLMClient(config, usage_tracker=tracker)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None  # No usage
        mock_response.model = "zai/glm-4"

        from src.agent import llm_client as lc_mod
        original = lc_mod.completion
        lc_mod.completion = lambda **kw: mock_response
        try:
            result = client.chat(messages=[{"role": "user", "content": "hi"}])
        finally:
            lc_mod.completion = original

        # No records should be written
        rows = db.conn.execute("SELECT * FROM token_usage").fetchall()
        assert len(rows) == 0
        db.close()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Reranker Integration Tests
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class TestRerankerIntegration:
    """Test ZhipuAIReranker records usage via UsageTracker."""

    def _make_reranker_with_tracker(self, tmp_path):
        """Create a reranker with a real tracker and DB."""
        db = _open_db(tmp_path)
        tracker = UsageTracker(db)
        reranker = ZhipuAIReranker(api_key="test-key", usage_tracker=tracker)
        return reranker, tracker, db

    def test_rerank_records_to_tracker(self, tmp_path):
        """rerank() should record usage via tracker when API succeeds."""
        reranker, tracker, db = self._make_reranker_with_tracker(tmp_path)

        mock_urlopen = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.95}],
            "usage": {"prompt_tokens": 50, "total_tokens": 60},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with patch("src.search.reranker.urllib.request.urlopen", mock_urlopen):
            results = reranker.rerank("test query", ["doc1"], top_n=1)

        assert len(results) == 1
        # Check tracker recorded the rerank call
        row = db.conn.execute("SELECT * FROM token_usage ORDER BY id DESC LIMIT 1").fetchone()
        assert dict(row)["call_type"] == "rerank"
        assert dict(row)["model"] == "rerank"
        assert dict(row)["input_tokens"] == 50
        assert dict(row)["total_tokens"] == 60
        db.close()

    def test_rerank_no_tracker_works(self, tmp_path):
        """rerank() should work without a tracker (backward compatible)."""
        reranker = ZhipuAIReranker(api_key="test-key")  # No tracker

        mock_urlopen = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"prompt_tokens": 30, "total_tokens": 40},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with patch("src.search.reranker.urllib.request.urlopen", mock_urlopen):
            results = reranker.rerank("test", ["doc1"], top_n=1)

        assert len(results) == 1
        assert results[0].relevance_score == 0.9

    def test_rerank_no_tracking_on_fallback(self, tmp_path):
        """Fallback (no API key) should not record anything to tracker."""
        reranker, tracker, db = self._make_reranker_with_tracker(tmp_path)
        reranker._api_key = ""  # No key 鈫?fallback

        results = reranker.rerank("test", ["doc1"], top_n=1)
        assert len(results) == 1

        # No records should be written (fallback doesn't call _call_api)
        rows = db.conn.execute("SELECT * FROM token_usage").fetchall()
        assert len(rows) == 0
        db.close()

    def test_rerank_multiple_calls_accumulate(self, tmp_path):
        """Multiple rerank calls should each record separately."""
        reranker, tracker, db = self._make_reranker_with_tracker(tmp_path)

        mock_urlopen = MagicMock()
        def make_resp(data):
            resp = MagicMock()
            resp.read.return_value = json.dumps(data).encode("utf-8")
            resp.__enter__ = lambda s: resp
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        call_data = [
            {"results": [{"index": 0, "relevance_score": 0.9}], "usage": {"prompt_tokens": 10, "total_tokens": 15}},
            {"results": [{"index": 1, "relevance_score": 0.8}], "usage": {"prompt_tokens": 20, "total_tokens": 25}},
        ]
        mock_urlopen.side_effect = [make_resp(d) for d in call_data]

        with patch("src.search.reranker.urllib.request.urlopen", mock_urlopen):
            reranker.rerank("q1", ["d1"], top_n=1)
            reranker.rerank("q2", ["d1", "d2"], top_n=1)

        rows = db.conn.execute("SELECT * FROM token_usage WHERE call_type='rerank'").fetchall()
        assert len(rows) == 2
        assert dict(rows[0])["input_tokens"] == 10
        assert dict(rows[1])["input_tokens"] == 20
        db.close()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# SearchAgent Session Integration Tests
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class TestSearchAgentSession:
    """Test SearchAgent session management with UsageTracker."""

    def _make_agent_with_tracker(self, tmp_path):
        """Create a SearchAgent with a real UsageTracker."""
        from unittest.mock import MagicMock, PropertyMock

        from src.agent.llm_client import ChatResponse
        from src.agent.search_agent import SearchAgent
        from src.agent.tools.read import ReadTool
        from src.agent.tools.search import SearchTool

        db = _open_db(tmp_path)
        tracker = UsageTracker(db)

        config = MagicMock()
        config.glm_api_key = "test-key"
        config.glm_base_url = "https://api.test.com"
        config.llm_model = "glm-4"
        config.litellm_model = "zai/glm-4"
        config.llm_temperature = 0.7
        config.llm_max_tokens = 1000
        config.active_api_key = "test-key"
        config.active_base_url = "https://api.test.com"

        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = ChatResponse(
            content="Test answer",
            usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        )

        search_tool = MagicMock(spec=SearchTool)
        type(search_tool).name = PropertyMock(return_value="search")
        type(search_tool).description = PropertyMock(return_value="search")
        search_tool.to_openai_tool.return_value = {"type": "function", "function": {"name": "search", "parameters": {}}}

        read_tool = MagicMock(spec=ReadTool)
        type(read_tool).name = PropertyMock(return_value="read")
        type(read_tool).description = PropertyMock(return_value="read")
        read_tool.to_openai_tool.return_value = {"type": "function", "function": {"name": "read", "parameters": {}}}

        agent = SearchAgent(
            config=config,
            search_tool=search_tool,
            read_tool=read_tool,
            llm_client=mock_llm,
            mode="tool_loop",
            usage_tracker=tracker,
        )
        return agent, tracker, db

    def test_run_starts_session(self, tmp_path):
        """run() should start a session when tracker is provided."""
        agent, tracker, db = self._make_agent_with_tracker(tmp_path)

        assert agent._session_id is None
        response = agent.run("test query")
        assert agent._session_id is not None
        assert len(agent._session_id) == 8
        assert tracker.session_id == agent._session_id
        db.close()

    def test_run_without_tracker_no_session(self, tmp_path):
        """run() should work without tracker, no session started."""
        from unittest.mock import MagicMock, PropertyMock

        from src.agent.llm_client import ChatResponse
        from src.agent.search_agent import SearchAgent
        from src.agent.tools.read import ReadTool
        from src.agent.tools.search import SearchTool

        config = MagicMock()
        config.glm_api_key = "test-key"

        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = ChatResponse(
            content="Answer", usage={"total_tokens": 50},
        )

        search_tool = MagicMock(spec=SearchTool)
        type(search_tool).name = PropertyMock(return_value="search")
        type(search_tool).description = PropertyMock(return_value="search")

        read_tool = MagicMock(spec=ReadTool)
        type(read_tool).name = PropertyMock(return_value="read")
        type(read_tool).description = PropertyMock(return_value="read")

        agent = SearchAgent(
            config=config,
            search_tool=search_tool,
            read_tool=read_tool,
            llm_client=mock_llm,
            mode="tool_loop",
        )
        assert agent._usage_tracker is None
        response = agent.run("test query")
        assert agent._session_id is None  # No session without tracker

    def test_session_id_persists_across_runs(self, tmp_path):
        """Session ID from first run() should persist to second run()."""
        agent, tracker, db = self._make_agent_with_tracker(tmp_path)

        agent.run("first query")
        sid1 = agent._session_id
        assert sid1 is not None

        agent.run("second query")
        sid2 = agent._session_id
        assert sid2 is not None
        # start_session generates a new ID each call
        # The session_id is the latest one
        assert len(sid2) == 8

        db.close()

    def test_pipeline_mode_starts_session(self, tmp_path):
        """Pipeline mode should also start a session."""
        from unittest.mock import MagicMock, PropertyMock

        from src.agent.llm_client import ChatResponse
        from src.agent.search_agent import SearchAgent
        from src.agent.tools.read import ReadTool
        from src.agent.tools.search import SearchTool

        db = _open_db(tmp_path)
        tracker = UsageTracker(db)

        config = MagicMock()
        config.glm_api_key = "test-key"
        config.llm_model = "glm-4"
        config.litellm_model = "zai/glm-4"

        mock_llm = MagicMock()
        # First call: query analysis 鈫?"direct" for greeting
        mock_llm.chat.return_value = ChatResponse(
            content='{"action": "direct", "search_query": "浣犲ソ"}',
            usage={"total_tokens": 20},
        )

        search_tool = MagicMock(spec=SearchTool)
        type(search_tool).name = PropertyMock(return_value="search")
        type(search_tool).description = PropertyMock(return_value="search")

        read_tool = MagicMock(spec=ReadTool)
        type(read_tool).name = PropertyMock(return_value="read")
        type(read_tool).description = PropertyMock(return_value="read")

        agent = SearchAgent(
            config=config,
            search_tool=search_tool,
            read_tool=read_tool,
            llm_client=mock_llm,
            mode="pipeline",
            usage_tracker=tracker,
        )

        response = agent.run("浣犲ソ")
        assert agent._session_id is not None
        db.close()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# OCR Pipeline Integration Tests
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲


class TestCoordinatorOCRTracking:
    """Test ConverterCoordinator records OCR usage via UsageTracker."""

    def _make_coordinator_with_tracker(self, tmp_path):
        """Create a coordinator with OCR config and a real UsageTracker."""
        from src.converter.ocr import OCRServiceConfig

        db = _open_db(tmp_path)
        tracker = UsageTracker(db, source_dir="test_docs")

        ocr_config = OCRServiceConfig(
            api_key="test-key",
            model="glm-ocr",
        )
        coordinator = ConverterCoordinator(
            ocr_config=ocr_config,
            usage_tracker=tracker,
        )
        return coordinator, tracker, db

    def test_scanned_pdf_records_ocr_to_tracker(self, tmp_path):
        """Scanned PDF OCR fallback should record tokens via UsageTracker."""
        coordinator, tracker, db = self._make_coordinator_with_tracker(tmp_path)

        # Create a minimal PDF
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n%%EOF"
        )
        output_dir = tmp_path / "output"

        # Mock OCRService.recognize to return OCR with token usage
        from unittest.mock import patch

        from src.converter.ocr import OCRResult

        # Create a fake image file for OCR to process
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / "page_1.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_ocr_result = OCRResult(
            success=True,
            text="Extracted text from scanned page",
            token_usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
        )

        # Patch the PDF converter to produce minimal-text result (triggers OCR detection)
        from src.converter.pdf import PDFConverter

        original_convert = PDFConverter.convert

        def fake_pdf_convert(self, source, output_dir, options=None):
            result = original_convert(self, source, output_dir, options)
            # Force it to look like a scanned PDF
            result.markdown = "  "  # Minimal text 鈫?triggers OCR
            result.metadata["page_count"] = 1
            return result

        # Patch _render_pdf_pages to return our fake image on-demand
        def fake_render(self, source, out_dir, dpi=150):
            return [img_path], None  # No temp_dir to clean up

        with patch.object(PDFConverter, "convert", fake_pdf_convert), patch.object(
            ConverterCoordinator, "_render_pdf_pages", fake_render
        ):
            # Ensure OCR service is initialized
            coordinator._get_ocr_service()
            with patch.object(
                coordinator._ocr_service, "recognize", return_value=mock_ocr_result
            ):
                result = coordinator.convert(pdf_path, output_dir)

        # Verify OCR was used
        assert result.ocr_used is True
        assert result.ocr_model == "glm-ocr"

        # Verify tracker recorded the OCR call
        rows = db.conn.execute(
            "SELECT * FROM token_usage WHERE call_type='ocr'"
        ).fetchall()
        assert len(rows) >= 1
        row = dict(rows[0])
        assert row["model"] == "glm-ocr"
        assert row["input_tokens"] == 500
        assert row["output_tokens"] == 200
        assert row["total_tokens"] == 700
        assert row["source_dir"] == "test_docs"
        assert row["cost_millicents"] > 0
        db.close()

    def test_coordinator_without_tracker_still_works(self, tmp_path):
        """Coordinator without UsageTracker should work (backward compatible)."""
        from src.converter.ocr import OCRServiceConfig

        coordinator = ConverterCoordinator(
            ocr_config=OCRServiceConfig(api_key="test-key"),
        )
        assert coordinator._usage_tracker is None

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n%%EOF"
        )
        output_dir = tmp_path / "output"
        result = coordinator.convert(pdf_path, output_dir)
        assert isinstance(result, ConvertResult)

    def test_coordinator_tracker_exception_does_not_break_ocr(self, tmp_path):
        """If UsageTracker.record_ocr raises, OCR result should still be returned."""
        from unittest.mock import MagicMock, patch

        from src.converter.ocr import OCRResult, OCRServiceConfig

        db = _open_db(tmp_path)
        tracker = UsageTracker(db, source_dir="test_docs")
        # Make the tracker fail
        tracker.record_ocr = MagicMock(side_effect=RuntimeError("DB locked"))

        ocr_config = OCRServiceConfig(api_key="test-key", model="glm-ocr")
        coordinator = ConverterCoordinator(
            ocr_config=ocr_config,
            usage_tracker=tracker,
        )

        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n%%EOF"
        )
        output_dir = tmp_path / "output"

        mock_ocr_result = OCRResult(
            success=True,
            text="Text from OCR",
            token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        from src.converter.pdf import PDFConverter

        def fake_pdf_convert(self, source, output_dir, options=None):
            result = ConvertResult(
                success=True,
                markdown="  ",
                source_file=source,
                converter_name="pdfplumber",
                converter_version="0.1.0",
            )
            result.metadata["page_count"] = 1
            output_dir.mkdir(parents=True, exist_ok=True)
            img_path = output_dir / "page_1.png"
            img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            result.images = [img_path]
            return result

        with patch.object(PDFConverter, "convert", fake_pdf_convert):
            coordinator._get_ocr_service()
            with patch.object(
                coordinator._ocr_service, "recognize", return_value=mock_ocr_result
            ):
                result = coordinator.convert(pdf_path, output_dir)

        # OCR should still succeed despite tracker failure
        assert result.ocr_used is True
        assert "Text from OCR" in result.markdown
        tracker.record_ocr.assert_called_once()
        db.close()


class TestImageConverterOCRTracking:
    """Test ImageConverter records OCR usage via UsageTracker."""

    def test_image_convert_records_to_tracker(self, tmp_path):
        """ImageConverter.convert should record OCR via usage_tracker in options."""
        from src.converter.image import ImageConverter
        from src.converter.ocr import OCRResult

        db = _open_db(tmp_path)
        tracker = UsageTracker(db, source_dir="img_docs")

        converter = ImageConverter()

        # Create a fake PNG file
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        output_dir = tmp_path / "output"

        mock_ocr_result = OCRResult(
            success=True,
            text="Extracted text from image",
            token_usage={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400},
        )

        from unittest.mock import patch
        with patch("src.converter.image.OCRService") as MockOCRService:
            mock_service = MockOCRService.return_value
            mock_service.recognize.return_value = mock_ocr_result

            result = converter.convert(
                img_path,
                output_dir,
                options={
                    "ocr_api_key": "test-key",
                    "usage_tracker": tracker,
                },
            )

        assert result.success is True
        assert result.ocr_used is True

        # Verify tracker recorded the OCR call
        rows = db.conn.execute(
            "SELECT * FROM token_usage WHERE call_type='ocr'"
        ).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["model"] == "glm-ocr"
        assert row["input_tokens"] == 300
        assert row["output_tokens"] == 100
        assert row["total_tokens"] == 400
        assert row["source_dir"] == "img_docs"
        assert row["cost_millicents"] > 0
        db.close()

    def test_image_convert_without_tracker_still_works(self, tmp_path):
        """ImageConverter.convert should work without usage_tracker (backward compatible)."""
        from unittest.mock import patch

        from src.converter.image import ImageConverter
        from src.converter.ocr import OCRResult

        converter = ImageConverter()

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        output_dir = tmp_path / "output"

        mock_ocr_result = OCRResult(
            success=True,
            text="Some text",
            token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        with patch("src.converter.image.OCRService") as MockOCRService:
            mock_service = MockOCRService.return_value
            mock_service.recognize.return_value = mock_ocr_result

            result = converter.convert(
                img_path,
                output_dir,
                options={"ocr_api_key": "test-key"},
            )

        assert result.success is True
        assert result.ocr_used is True

    def test_image_convert_tracker_exception_handled(self, tmp_path):
        """ImageConverter should not fail if usage_tracker raises."""
        from unittest.mock import MagicMock, patch

        from src.converter.image import ImageConverter
        from src.converter.ocr import OCRResult

        converter = ImageConverter()

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        output_dir = tmp_path / "output"

        mock_tracker = MagicMock()
        mock_tracker.record_ocr.side_effect = RuntimeError("DB error")

        mock_ocr_result = OCRResult(
            success=True,
            text="Text",
            token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        with patch("src.converter.image.OCRService") as MockOCRService:
            mock_service = MockOCRService.return_value
            mock_service.recognize.return_value = mock_ocr_result

            result = converter.convert(
                img_path,
                output_dir,
                options={
                    "ocr_api_key": "test-key",
                    "usage_tracker": mock_tracker,
                },
            )

        # Should still succeed
        assert result.success is True
        assert result.ocr_used is True
        mock_tracker.record_ocr.assert_called_once()

    def test_image_convert_no_tokens_skips_tracker(self, tmp_path):
        """ImageConverter should not call tracker when OCR returns no token_usage."""
        from unittest.mock import MagicMock, patch

        from src.converter.image import ImageConverter
        from src.converter.ocr import OCRResult

        converter = ImageConverter()

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        output_dir = tmp_path / "output"

        mock_tracker = MagicMock()

        mock_ocr_result = OCRResult(
            success=True,
            text="Text",
            token_usage={},  # Empty token usage
        )

        with patch("src.converter.image.OCRService") as MockOCRService:
            mock_service = MockOCRService.return_value
            mock_service.recognize.return_value = mock_ocr_result

            result = converter.convert(
                img_path,
                output_dir,
                options={
                    "ocr_api_key": "test-key",
                    "usage_tracker": mock_tracker,
                },
            )

        assert result.success is True
        # Empty token_usage 鈫?should NOT call tracker (falsy dict)
        mock_tracker.record_ocr.assert_not_called()

