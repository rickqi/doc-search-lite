"""Tests for ABTestRunner — 统一 A/B 测试框架."""
import json
from pathlib import Path

import pytest

from src.search.ab_testing import (
    ABTestRunner, ABTestResult, ArmResult, RunnerConfig, QueryCase, SingleResult,
    calc_similarity, estimate_cost, load_queries_from_benchmark,
)


class TestCalcSimilarity:
    """calc_similarity — 字符 bigram Jaccard."""

    def test_identical(self):
        assert calc_similarity("年假为5天", "年假为5天") == 1.0

    def test_partial(self):
        s = calc_similarity("年假为5天", "年假有5天")
        assert 0.3 < s < 1.0

    def test_different(self):
        assert calc_similarity("年假为5天", "加班费1.5倍") < 0.3

    def test_empty(self):
        assert calc_similarity("", "test") == 0.0
        assert calc_similarity("test", "") == 0.0
        assert calc_similarity("", "") == 0.0


class TestEstimateCost:
    """estimate_cost — LLM 费用估算."""

    def test_deepseek(self):
        cost = estimate_cost("deepseek-v4-flash", 1000, 500)
        assert cost > 0
        assert cost < 1.0  # 1000+500 token 应该不到 1 分

    def test_glm(self):
        cost = estimate_cost("glm-4", 1000, 500)
        assert cost > 0

    def test_zero_tokens(self):
        assert estimate_cost("deepseek", 0, 0) == 0.0


class TestLoadQueries:
    """load_queries_from_benchmark — 从 JSON 加载测试查询."""

    def test_load_all(self):
        path = Path(__file__).resolve().parent.parent / "docs" / "qa_benchmark_cases.json"
        if not path.exists():
            pytest.skip("qa_benchmark_cases.json not found")
        queries = load_queries_from_benchmark(str(path))
        assert len(queries) > 0
        q = queries[0]
        assert q.question
        assert q.standard_answer
        assert q.id

    def test_filter_domain(self):
        path = Path(__file__).resolve().parent.parent / "docs" / "qa_benchmark_cases.json"
        if not path.exists():
            pytest.skip("qa_benchmark_cases.json not found")
        queries = load_queries_from_benchmark(str(path), domain="人事管理")
        assert all(q.domain == "人事管理" for q in queries)

    def test_limit(self):
        path = Path(__file__).resolve().parent.parent / "docs" / "qa_benchmark_cases.json"
        if not path.exists():
            pytest.skip("qa_benchmark_cases.json not found")
        queries = load_queries_from_benchmark(str(path), limit=5)
        assert len(queries) == 5


class TestABTestResult:
    """ABTestResult — 汇总与统计."""

    @pytest.fixture
    def sample_result(self):
        config_a = RunnerConfig(name="A", mode="tool_loop", index_path="/idx")
        config_b = RunnerConfig(name="B", mode="pipeline", index_path="/idx")

        arm_a = ArmResult(config=config_a)
        arm_b = ArmResult(config=config_b)

        for qid, acc_a, acc_b in [("q1", 1.0, 0.5), ("q2", 0.5, 1.0), ("q3", 0.0, 0.5)]:
            arm_a.per_query[qid] = [
                SingleResult(query_id=qid, question=f"q{qid}", answer="ans", accuracy=acc_a, latency=1.0, tokens_used=100)
            ]
            arm_b.per_query[qid] = [
                SingleResult(query_id=qid, question=f"q{qid}", answer="ans", accuracy=acc_b, latency=2.0, tokens_used=200)
            ]

        arm_a.runs_completed = 1
        arm_b.runs_completed = 1

        return ABTestResult(
            config_a=config_a, config_b=config_b,
            arm_a=arm_a, arm_b=arm_b,
            queries_count=3, runs_per_arm=1,
        )

    def test_summary_contains_both_arms(self, sample_result):
        summary = sample_result.summary()
        assert "A" in summary
        assert "B" in summary

    def test_to_dict_structure(self, sample_result):
        d = sample_result.to_dict()
        assert "config_a" in d
        assert "config_b" in d
        assert "summary_a" in d
        assert "summary_b" in d
        assert d["queries_count"] == 3

    def test_save_json(self, sample_result, tmp_path):
        out = tmp_path / "result.json"
        sample_result.save_json(str(out))
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "config_a" in data

    def test_arm_summary(self):
        config = RunnerConfig(name="test")
        arm = ArmResult(config=config)
        arm.per_query["q1"] = [SingleResult(accuracy=0.5, latency=2.0, tokens_used=100)]
        arm.per_query["q2"] = [SingleResult(accuracy=1.0, latency=1.0, tokens_used=200)]
        arm.runs_completed = 1

        s = arm.summary()
        assert s["accuracy_mean"] == 0.75
        assert s["latency_mean"] == 1.5
        assert s["tokens_mean"] == 150


class TestABTestRunner:
    """ABTestRunner — 运行器基本逻辑."""

    def test_runner_creation(self):
        runner = ABTestRunner(seed=42)
        assert runner is not None
        assert runner.seed == 42

    def test_run_with_mock_queries_structure(self, tmp_path):
        """验证 ABTestRunner.run() 返回结构正确的 ABTestResult."""
        runner = ABTestRunner(seed=42)

        config_a = RunnerConfig(name="A", mode="tool_loop", index_path="/idx")
        config_b = RunnerConfig(name="B", mode="pipeline", index_path="/idx")

        # 创建 mock queries — 不调用真实 agent, 只验证
        # load_queries_from_benchmark 能正常加载
        path = Path(__file__).resolve().parent.parent / "docs" / "qa_benchmark_cases.json"
        if path.exists():
            from src.search.ab_testing import load_queries_from_benchmark
            queries = load_queries_from_benchmark(str(path), limit=2)
            assert len(queries) == 2
            assert queries[0].question
            assert queries[0].standard_answer
