"""Tests for StatsReporter — report export in JSON/CSV/Markdown/HTML."""

import csv
import io
import json
from pathlib import Path

from src.stats.reporter import StatsReporter
from src.storage.convert_db import ConvertDB

# ── Helpers ──────────────────────────────────────────────────────────


def _open_db(tmp_path: Path) -> ConvertDB:
    """Create and open a fresh ConvertDB in tmp_path."""
    db = ConvertDB(tmp_path / "test.db")
    db.open()
    return db


def _seed_data(db: ConvertDB):
    """Insert sample token usage data."""
    db.add_token_usage_extended(
        "ocr", "glm-ocr",
        input_tokens=500, output_tokens=200, total_tokens=700,
        cost_millicents=350, source_dir="docs-a",
    )
    db.add_token_usage_extended(
        "llm_chat", "zai/glm-4",
        input_tokens=1000, output_tokens=500, total_tokens=1500,
        cost_millicents=75000, source_dir="docs-a",
    )
    db.add_token_usage_extended(
        "rerank", "rerank",
        input_tokens=100, total_tokens=100,
        cost_millicents=0, source_dir="docs-a",
    )


# ════════════════════════════════════════════════════════════════════════
# StatsReporter Tests
# ════════════════════════════════════════════════════════════════════════


class TestStatsReporter:
    """Test StatsReporter class."""

    def _make_reporter(self, tmp_path):
        """Create a StatsReporter with a fresh DB and seed data."""
        db = _open_db(tmp_path)
        _seed_data(db)
        reporter = StatsReporter(db)
        return reporter, db

    def test_generate_summary(self, tmp_path):
        """generate_summary should return dict with summary, daily, models."""
        reporter, db = self._make_reporter(tmp_path)
        data = reporter.generate_summary()

        assert "summary" in data
        assert "daily" in data
        assert "models" in data

        # Summary has by_type and total
        assert "by_type" in data["summary"]
        assert "total" in data["summary"]
        assert "ocr" in data["summary"]["by_type"]
        assert "llm_chat" in data["summary"]["by_type"]

        # Total counts
        assert data["summary"]["total"]["call_count"] == 3
        db.close()

    def test_generate_summary_with_filters(self, tmp_path):
        """generate_summary with source_dir should filter results."""
        reporter, db = self._make_reporter(tmp_path)

        # Filter by source_dir that has data
        data = reporter.generate_summary(source_dir="docs-a")
        assert data["summary"]["total"]["call_count"] == 3

        # Filter by non-existent source_dir
        data_empty = reporter.generate_summary(source_dir="nonexistent")
        assert data_empty["summary"]["total"]["call_count"] == 0
        db.close()

    def test_export_json(self, tmp_path):
        """export_json should produce valid JSON with expected structure."""
        reporter, db = self._make_reporter(tmp_path)
        data = reporter.generate_summary()
        content = reporter.export_json(data)

        parsed = json.loads(content)
        assert "summary" in parsed
        assert "daily" in parsed
        assert "models" in parsed
        assert parsed["summary"]["total"]["call_count"] == 3
        db.close()

    def test_export_csv(self, tmp_path):
        """export_csv should produce valid CSV with expected columns."""
        reporter, db = self._make_reporter(tmp_path)
        data = reporter.generate_summary()
        content = reporter.export_csv(data)

        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        # Header row
        assert rows[0][0] == "section"
        assert rows[0][2] == "call_count"

        # Find summary rows
        summary_rows = [r for r in rows[1:] if r[0] == "summary"]
        assert len(summary_rows) >= 1

        # Find daily rows
        daily_rows = [r for r in rows[1:] if r[0] == "daily"]
        assert len(daily_rows) >= 1

        # Find model rows
        model_rows = [r for r in rows[1:] if r[0] == "model"]
        assert len(model_rows) >= 1
        db.close()

    def test_export_markdown(self, tmp_path):
        """export_markdown should produce markdown with tables."""
        reporter, db = self._make_reporter(tmp_path)
        data = reporter.generate_summary()
        content = reporter.export_markdown(data)

        assert "# API 用量统计报告" in content
        assert "## 总体汇总" in content
        assert "ocr" in content
        assert "llm_chat" in content
        assert "|------|" in content  # Markdown table separator
        db.close()

    def test_export_html(self, tmp_path):
        """export_html should produce valid HTML with tables."""
        reporter, db = self._make_reporter(tmp_path)
        data = reporter.generate_summary()
        content = reporter.export_html(data)

        assert "<!DOCTYPE html>" in content
        assert "<table>" in content
        assert "总体汇总" in content
        assert "ocr" in content
        assert "</html>" in content
        # Should have CSS embedded
        assert "<style>" in content
        db.close()

    def test_export_html_escaping(self, tmp_path):
        """export_html should escape special characters in model names."""
        db = _open_db(tmp_path)
        db.add_token_usage_extended(
            "llm_chat", "test<script>alert(1)</script>",
            input_tokens=100, total_tokens=100,
        )
        reporter = StatsReporter(db)
        data = reporter.generate_summary()
        content = reporter.export_html(data)

        # Script tags should be escaped
        assert "<script>" not in content
        assert "&lt;script&gt;" in content
        db.close()

    def test_export_to_file(self, tmp_path):
        """All export methods should write to file when output_path is provided."""
        reporter, db = self._make_reporter(tmp_path)
        data = reporter.generate_summary()

        # JSON
        json_path = tmp_path / "report.json"
        reporter.export_json(data, output_path=json_path)
        assert json_path.exists()
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert "summary" in parsed

        # CSV
        csv_path = tmp_path / "report.csv"
        reporter.export_csv(data, output_path=csv_path)
        assert csv_path.exists()
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "section,key" in csv_content

        # Markdown
        md_path = tmp_path / "report.md"
        reporter.export_markdown(data, output_path=md_path)
        assert md_path.exists()
        md_content = md_path.read_text(encoding="utf-8")
        assert "# API" in md_content

        # HTML
        html_path = tmp_path / "report.html"
        reporter.export_html(data, output_path=html_path)
        assert html_path.exists()
        html_content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content

        db.close()

    def test_export_creates_parent_dirs(self, tmp_path):
        """Export should create parent directories if they don't exist."""
        reporter, db = self._make_reporter(tmp_path)
        data = reporter.generate_summary()

        nested_path = tmp_path / "subdir" / "deep" / "report.json"
        reporter.export_json(data, output_path=nested_path)
        assert nested_path.exists()
        db.close()

    def test_generate_summary_empty_db(self, tmp_path):
        """generate_summary on empty DB should return zero totals."""
        db = _open_db(tmp_path)
        reporter = StatsReporter(db)
        data = reporter.generate_summary()

        assert data["summary"]["total"]["call_count"] == 0
        assert data["summary"]["total"]["input_tokens"] == 0
        assert data["daily"] == [] or len(data["daily"]) == 0
        assert data["models"] == [] or len(data["models"]) == 0
        db.close()
