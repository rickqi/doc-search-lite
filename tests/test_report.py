"""Unit tests for BenchmarkReporter — report generation in text/markdown/JSON/HTML."""

import json

import pytest

from src.search.benchmark import BenchmarkResult, ModeResult, QuerySpec
from src.search.report import BenchmarkReporter


@pytest.fixture
def sample_benchmark_result():
    """Create a sample BenchmarkResult for testing."""
    queries = [
        QuerySpec(query="年假", expected_files=["年假制度.md"], category="hr"),
        QuerySpec(query="报销", expected_files=["报销.md"]),
    ]
    results = [
        ModeResult(
            mode="bm25", query="年假", success=True,
            latency=0.01, result_count=5,
            result_files=["年假制度.md"], scores=[0.9],
            hit_rate=1.0, mrr=1.0,
        ),
        ModeResult(
            mode="grep", query="年假", success=True,
            latency=0.05, result_count=3,
            result_files=["年假制度.md"], scores=[],
            hit_rate=1.0, mrr=0.5,
        ),
        ModeResult(
            mode="bm25", query="报销", success=True,
            latency=0.02, result_count=2,
            result_files=["报销.md"], scores=[0.8],
            hit_rate=1.0, mrr=1.0,
        ),
        ModeResult(
            mode="grep", query="报销", success=False,
            latency=0.001, error="no index",
        ),
    ]
    return BenchmarkResult(
        queries=queries,
        results=results,
        modes_tested=["bm25", "grep"],
        index_path="/test/index",
        total_time=0.1,
    )


@pytest.fixture
def empty_benchmark_result():
    """Create an empty BenchmarkResult."""
    return BenchmarkResult(
        queries=[], results=[], modes_tested=[], index_path="/x",
    )


@pytest.fixture
def single_mode_result():
    """Create a single-mode BenchmarkResult."""
    queries = [QuerySpec(query="test")]
    results = [
        ModeResult(
            mode="bm25", query="test", success=True,
            latency=0.03, result_count=10,
            hit_rate=0.8, mrr=0.6,
        ),
    ]
    return BenchmarkResult(
        queries=queries,
        results=results,
        modes_tested=["bm25"],
        index_path="/idx",
        total_time=0.05,
    )


class TestBenchmarkReporterGenerate:
    """Test the main generate() dispatch method."""

    def test_text_default(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result)
        assert isinstance(report, str)
        assert "搜索基准测试报告" in report

    def test_text_explicit(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="text")
        assert isinstance(report, str)
        assert "搜索基准测试报告" in report

    def test_markdown(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="markdown")
        assert isinstance(report, str)
        assert "# 搜索基准测试报告" in report

    def test_md_alias(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="md")
        assert "# 搜索基准测试报告" in report

    def test_json(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="json")
        parsed = json.loads(report)
        assert "queries" in parsed
        assert "results" in parsed
        assert "modes_tested" in parsed

    def test_html(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "<html" in report
        assert "</html>" in report


class TestTextReport:
    """Test plain text report content."""

    def test_contains_header(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="text")
        assert "搜索基准测试报告" in report
        assert "=" * 60 in report

    def test_contains_index_path(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="text")
        assert "/test/index" in report

    def test_contains_mode_names(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="text")
        assert "bm25" in report
        assert "grep" in report

    def test_contains_query_details(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="text")
        assert "年假" in report
        assert "报销" in report

    def test_empty_result(self, empty_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(empty_benchmark_result, fmt="text")
        assert isinstance(report, str)
        assert "搜索基准测试报告" in report


class TestMarkdownReport:
    """Test Markdown report content."""

    def test_has_title(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="markdown")
        assert "# 搜索基准测试报告" in report

    def test_has_overview_table(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="markdown")
        assert "## 总览" in report
        assert "| 模式 |" in report
        assert "|------|" in report

    def test_has_per_query_section(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="markdown")
        assert "## 逐查询详情" in report
        assert '"年假"' in report

    def test_has_conclusions(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="markdown")
        assert "## 结论" in report

    def test_mode_comparison(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="markdown")
        # Both modes should appear in the overview table
        assert "| bm25 " in report
        assert "| grep " in report

    def test_category_in_query_header(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="markdown")
        assert "[hr]" in report

    def test_empty_result_markdown(self, empty_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(empty_benchmark_result, fmt="markdown")
        assert "# 搜索基准测试报告" in report


class TestJsonReport:
    """Test JSON report content."""

    def test_valid_json(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="json")
        parsed = json.loads(report)
        assert isinstance(parsed, dict)

    def test_json_structure(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="json")
        parsed = json.loads(report)
        assert len(parsed["queries"]) == 2
        assert len(parsed["results"]) == 4
        assert parsed["index_path"] == "/test/index"

    def test_json_unicode(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="json")
        # ensure_ascii=False means Chinese chars are not escaped
        assert "年假" in report

    def test_empty_json(self, empty_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(empty_benchmark_result, fmt="json")
        parsed = json.loads(report)
        assert parsed["queries"] == []


class TestHtmlReport:
    """Test HTML report content."""

    def test_has_html_structure(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "<!DOCTYPE html>" in report
        assert "<html" in report
        assert "</html>" in report

    def test_has_overview_table(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "<table>" in report
        assert "<th>模式</th>" in report

    def test_has_query_sections(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "<details>" in report
        assert "年假" in report

    def test_has_latency_chart(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "<svg" in report
        assert "bm25" in report

    def test_has_css(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "<style>" in report
        assert "--bg:" in report

    def test_has_conclusions(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "结论" in report

    def test_best_worst_color_coding(self, sample_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(sample_benchmark_result, fmt="html")
        assert "best" in report or "worst" in report

    def test_empty_html(self, empty_benchmark_result):
        reporter = BenchmarkReporter()
        report = reporter.generate(empty_benchmark_result, fmt="html")
        assert "<html" in report


class TestConclusions:
    """Test _conclusions method."""

    def test_empty_summary(self):
        reporter = BenchmarkReporter()
        result = reporter._conclusions({})
        assert result == "无测试结果。"

    def test_single_mode(self):
        reporter = BenchmarkReporter()
        summary = {
            "bm25": {"avg_latency": 0.01, "avg_hit_rate": 0.9, "avg_mrr": 0.8},
        }
        result = reporter._conclusions(summary)
        assert "bm25" in result
        assert "0.0100s" in result

    def test_comparison(self):
        reporter = BenchmarkReporter()
        summary = {
            "bm25": {"avg_latency": 0.01, "avg_hit_rate": 0.9, "avg_mrr": 0.8},
            "grep": {"avg_latency": 0.05, "avg_hit_rate": 0.7, "avg_mrr": 0.5},
        }
        result = reporter._conclusions(summary)
        assert "对比结果" in result
        assert "bm25 延迟最低" in result
        assert "bm25 命中率最高" in result
