"""Unit tests for BenchmarkRunner — search mode benchmarking."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.search.benchmark import BenchmarkResult, BenchmarkRunner, ModeResult, QuerySpec


class TestQuerySpec:
    """Test QuerySpec dataclass."""

    def test_minimal_creation(self):
        q = QuerySpec(query="年假")
        assert q.query == "年假"
        assert q.expected_files == []
        assert q.category == ""

    def test_full_creation(self):
        q = QuerySpec(
            query="test",
            expected_files=["file1.md", "file2.md"],
            category="hr",
        )
        assert q.query == "test"
        assert q.expected_files == ["file1.md", "file2.md"]
        assert q.category == "hr"


class TestModeResult:
    """Test ModeResult dataclass."""

    def test_minimal_creation(self):
        r = ModeResult(mode="bm25", query="test", success=True)
        assert r.mode == "bm25"
        assert r.query == "test"
        assert r.success is True
        assert r.latency == 0.0
        assert r.result_count == 0
        assert r.result_files == []
        assert r.scores == []
        assert r.hit_rate == 0.0
        assert r.mrr == 0.0
        assert r.error == ""
        assert r.run_index == 0

    def test_full_creation(self):
        r = ModeResult(
            mode="grep",
            query="test",
            success=True,
            latency=0.05,
            result_count=10,
            result_files=["a.md", "b.md"],
            scores=[0.9, 0.8],
            hit_rate=0.5,
            mrr=0.33,
            run_index=2,
        )
        assert r.latency == 0.05
        assert r.result_count == 10
        assert r.hit_rate == 0.5
        assert r.mrr == 0.33
        assert r.run_index == 2

    def test_failed_result(self):
        r = ModeResult(
            mode="bm25", query="test", success=False, error="crash",
        )
        assert r.success is False
        assert r.error == "crash"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_creation(self):
        queries = [QuerySpec(query="q1")]
        results = [ModeResult(mode="bm25", query="q1", success=True)]
        br = BenchmarkResult(
            queries=queries,
            results=results,
            modes_tested=["bm25"],
            index_path="/idx",
        )
        assert len(br.queries) == 1
        assert len(br.results) == 1
        assert br.modes_tested == ["bm25"]
        assert br.index_path == "/idx"
        assert br.total_time == 0.0

    def test_empty_result(self):
        br = BenchmarkResult(
            queries=[], results=[], modes_tested=[], index_path="/x",
        )
        assert br.queries == []
        assert br.results == []


class TestLoadQueries:
    """Test load_queries static method."""

    def test_load_valid_jsonl(self, tmp_path):
        jsonl = tmp_path / "queries.jsonl"
        lines = [
            json.dumps({"query": "年假", "expected_files": ["年假.md"], "category": "hr"}),
            json.dumps({"query": "test", "expected_files": []}),
            json.dumps({"query": "bare"}),
        ]
        jsonl.write_text("\n".join(lines), encoding="utf-8")

        queries = BenchmarkRunner.load_queries(jsonl)
        assert len(queries) == 3
        assert queries[0].query == "年假"
        assert queries[0].expected_files == ["年假.md"]
        assert queries[0].category == "hr"
        assert queries[1].expected_files == []
        assert queries[2].category == ""

    def test_load_skips_blank_and_comment_lines(self, tmp_path):
        jsonl = tmp_path / "queries.jsonl"
        content = "\n".join([
            "# This is a comment",
            json.dumps({"query": "first"}),
            "",
            "   ",
            json.dumps({"query": "second"}),
        ])
        jsonl.write_text(content, encoding="utf-8")

        queries = BenchmarkRunner.load_queries(jsonl)
        assert len(queries) == 2
        assert queries[0].query == "first"
        assert queries[1].query == "second"

    def test_load_empty_file(self, tmp_path):
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("", encoding="utf-8")

        queries = BenchmarkRunner.load_queries(jsonl)
        assert queries == []


class TestCalcRelevance:
    """Test _calc_relevance static method."""

    def test_perfect_hit(self):
        hr, mrr = BenchmarkRunner._calc_relevance(
            result_files=["a.md", "b.md", "c.md"],
            expected_files=["a.md"],
        )
        assert hr == 1.0
        assert mrr == 1.0

    def test_hit_at_rank_2(self):
        hr, mrr = BenchmarkRunner._calc_relevance(
            result_files=["x.md", "a.md"],
            expected_files=["a.md"],
        )
        assert hr == 1.0
        assert mrr == pytest.approx(0.5)

    def test_no_hits(self):
        hr, mrr = BenchmarkRunner._calc_relevance(
            result_files=["x.md", "y.md"],
            expected_files=["a.md"],
        )
        assert hr == 0.0
        assert mrr == 0.0

    def test_empty_expected(self):
        hr, mrr = BenchmarkRunner._calc_relevance(
            result_files=["a.md"],
            expected_files=[],
        )
        assert hr == 0.0
        assert mrr == 0.0

    def test_partial_hit_rate(self):
        hr, mrr = BenchmarkRunner._calc_relevance(
            result_files=["a.md"],
            expected_files=["a.md", "b.md"],
        )
        assert hr == 0.5

    def test_substring_match(self):
        """expected_files match via substring."""
        hr, mrr = BenchmarkRunner._calc_relevance(
            result_files=["path/to/年假制度.docx.md"],
            expected_files=["年假制度"],
        )
        assert hr == 1.0
        assert mrr == 1.0


class TestAggregateByMode:
    """Test aggregate_by_mode static method."""

    def test_single_mode_single_result(self):
        results = [
            ModeResult(mode="bm25", query="q", success=True, latency=0.1, hit_rate=1.0, mrr=1.0, result_count=5),
        ]
        summary = BenchmarkRunner.aggregate_by_mode(results)
        assert "bm25" in summary
        assert summary["bm25"]["avg_latency"] == 0.1
        assert summary["bm25"]["avg_hit_rate"] == 1.0
        assert summary["bm25"]["success_rate"] == 1.0

    def test_multiple_modes(self):
        results = [
            ModeResult(mode="bm25", query="q", success=True, latency=0.1, hit_rate=0.8),
            ModeResult(mode="grep", query="q", success=True, latency=0.2, hit_rate=0.6),
        ]
        summary = BenchmarkRunner.aggregate_by_mode(results)
        assert len(summary) == 2
        assert summary["bm25"]["avg_latency"] < summary["grep"]["avg_latency"]

    def test_failed_excluded_from_averages(self):
        results = [
            ModeResult(mode="bm25", query="q", success=True, latency=0.1, hit_rate=1.0, mrr=1.0, result_count=5),
            ModeResult(mode="bm25", query="q", success=False, latency=5.0, error="fail"),
        ]
        summary = BenchmarkRunner.aggregate_by_mode(results)
        # Only the success contributes to avg_latency
        assert summary["bm25"]["avg_latency"] == 0.1
        assert summary["bm25"]["success_rate"] == 0.5

    def test_all_failed(self):
        results = [
            ModeResult(mode="bm25", query="q", success=False, error="e"),
        ]
        summary = BenchmarkRunner.aggregate_by_mode(results)
        assert summary["bm25"]["avg_latency"] == 0.0
        assert summary["bm25"]["success_rate"] == 0.0

    def test_empty_input(self):
        summary = BenchmarkRunner.aggregate_by_mode([])
        assert summary == {}

    def test_averages_over_multiple_runs(self):
        results = [
            ModeResult(mode="bm25", query="q", success=True, latency=0.1, hit_rate=0.5, mrr=0.5, result_count=3),
            ModeResult(mode="bm25", query="q", success=True, latency=0.3, hit_rate=1.0, mrr=1.0, result_count=7),
        ]
        summary = BenchmarkRunner.aggregate_by_mode(results)
        assert summary["bm25"]["avg_latency"] == pytest.approx(0.2)
        assert summary["bm25"]["avg_hit_rate"] == pytest.approx(0.75)
        assert summary["bm25"]["avg_result_count"] == pytest.approx(5.0)


class TestBenchmarkRunnerInit:
    """Test BenchmarkRunner construction."""

    def test_default_raw_dir(self):
        runner = BenchmarkRunner(index_path=Path("/idx"))
        assert runner._index_path == Path("/idx")
        assert runner._raw_dir == Path("/idx").parent

    def test_explicit_raw_dir(self):
        runner = BenchmarkRunner(index_path=Path("/idx"), raw_dir=Path("/raw"))
        assert runner._raw_dir == Path("/raw")


class TestBenchmarkRunnerSingle:
    """Test _run_single method."""

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_bm25_mode_success(self, mock_bm25):
        mock_bm25.return_value = (["a.md", "b.md"], [0.9, 0.8])
        runner = BenchmarkRunner(index_path=Path("/idx"), raw_dir=Path("/raw"))

        spec = QuerySpec(query="test", expected_files=["a.md"])
        result = runner._run_single(spec, "bm25")

        assert result.success is True
        assert result.mode == "bm25"
        assert result.result_count == 2
        assert result.hit_rate == 1.0
        assert result.mrr == 1.0
        assert result.latency > 0

    @patch.object(BenchmarkRunner, "_search_grep")
    def test_grep_mode_success(self, mock_grep):
        mock_grep.return_value = (["c.md"], [])
        runner = BenchmarkRunner(index_path=Path("/idx"), raw_dir=Path("/raw"))

        spec = QuerySpec(query="test")
        result = runner._run_single(spec, "grep")

        assert result.success is True
        assert result.mode == "grep"
        assert result.result_count == 1

    def test_unknown_mode(self):
        runner = BenchmarkRunner(index_path=Path("/idx"))

        spec = QuerySpec(query="test")
        result = runner._run_single(spec, "unknown_mode")

        assert result.success is False
        assert "Unknown mode" in result.error

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_exception_captured(self, mock_bm25):
        mock_bm25.side_effect = RuntimeError("boom")
        runner = BenchmarkRunner(index_path=Path("/idx"))

        spec = QuerySpec(query="test")
        result = runner._run_single(spec, "bm25")

        assert result.success is False
        assert "boom" in result.error

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_run_index_propagated(self, mock_bm25):
        mock_bm25.return_value = ([], [])
        runner = BenchmarkRunner(index_path=Path("/idx"))

        spec = QuerySpec(query="test")
        result = runner._run_single(spec, "bm25", run_index=3)
        assert result.run_index == 3


class TestBenchmarkRunnerRun:
    """Test full run() method."""

    @patch.object(BenchmarkRunner, "_search_bm25")
    @patch.object(BenchmarkRunner, "_search_grep")
    def test_run_all_modes(self, mock_grep, mock_bm25):
        mock_bm25.return_value = (["a.md"], [0.9])
        mock_grep.return_value = (["a.md"], [])

        runner = BenchmarkRunner(index_path=Path("/idx"))
        queries = [QuerySpec(query="q1")]
        result = runner.run(queries, modes=["bm25", "grep"], warmup=0)

        assert len(result.results) == 2  # 1 query × 2 modes
        assert "bm25" in result.modes_tested
        assert "grep" in result.modes_tested
        assert result.total_time >= 0

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_run_single_mode(self, mock_bm25):
        mock_bm25.return_value = (["a.md"], [0.9])

        runner = BenchmarkRunner(index_path=Path("/idx"))
        queries = [QuerySpec(query="q1")]
        result = runner.run(queries, modes=["bm25"], warmup=0)

        assert len(result.results) == 1
        assert result.results[0].mode == "bm25"

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_run_multiple_queries(self, mock_bm25):
        mock_bm25.return_value = (["a.md"], [0.9])

        runner = BenchmarkRunner(index_path=Path("/idx"))
        queries = [QuerySpec(query="q1"), QuerySpec(query="q2")]
        result = runner.run(queries, modes=["bm25"], warmup=0)

        assert len(result.results) == 2
        assert result.queries == queries

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_run_with_repetitions(self, mock_bm25):
        mock_bm25.return_value = (["a.md"], [0.9])

        runner = BenchmarkRunner(index_path=Path("/idx"))
        queries = [QuerySpec(query="q1")]
        result = runner.run(queries, modes=["bm25"], runs=3, warmup=0)

        assert len(result.results) == 3
        assert all(r.query == "q1" for r in result.results)

    @patch.object(BenchmarkRunner, "_search_bm25")
    @patch.object(BenchmarkRunner, "_search_grep")
    def test_run_with_warmup(self, mock_grep, mock_bm25):
        mock_bm25.return_value = (["a.md"], [0.9])
        mock_grep.return_value = (["a.md"], [])

        runner = BenchmarkRunner(index_path=Path("/idx"))
        queries = [QuerySpec(query="q1")]
        result = runner.run(queries, modes=["bm25"], warmup=2)

        # warmup=2 means 2 discarded runs + 1 real run = 3 calls to _search_bm25
        assert mock_bm25.call_count == 3  # 2 warmup + 1 actual
        assert len(result.results) == 1

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_run_empty_queries(self, mock_bm25):
        runner = BenchmarkRunner(index_path=Path("/idx"))
        result = runner.run([], modes=["bm25"], warmup=0)

        assert len(result.results) == 0
        mock_bm25.assert_not_called()

    @patch.object(BenchmarkRunner, "_search_bm25")
    def test_default_modes(self, mock_bm25):
        mock_bm25.return_value = ([], [])

        runner = BenchmarkRunner(index_path=Path("/idx"))
        queries = [QuerySpec(query="q")]
        result = runner.run(queries, warmup=0)

        assert result.modes_tested == ["bm25", "grep"]
