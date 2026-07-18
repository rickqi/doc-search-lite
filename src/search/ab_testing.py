"""ABTestRunner — 统一 A/B 测试框架.

支持对比任意两个 Agent 配置（CLI vs MCP、不同模型、不同索引），
提供统计显著性检验、效果量计算、成本估算和报告生成。

用法::

    runner = ABTestRunner()

    # 定义 A/B 配置
    config_a = RunnerConfig(name="CLI Agent", mode="tool_loop",
        index_path=r"D:\\docs\raw\\制度\\index", raw_dir=r"D:\\docs\raw\\制度")
    config_b = RunnerConfig(name="MCP Pipeline", mode="pipeline",
        index_path=r"D:\\docs\raw\\制度\\index", raw_dir=r"D:\\docs\raw\\制度")

    # 加载测试查询
    queries = load_queries_from_benchmark("docs/qa_benchmark_cases.json", limit=10)

    # 运行 A/B 测试
    result = runner.run(config_a, config_b, queries, runs=3)

    # 生成报告
    print(result.summary())
    result.save_json("docs/ab_test_result.json")
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# ── 数据类 ──────────────────────────────────────────────────────────


@dataclass
class RunnerConfig:
    """A/B 测试的单臂配置."""

    name: str = ""                     # 配置名称 (如 "CLI Agent", "MCP Pipeline")
    mode: str = "tool_loop"            # agent 模式: tool_loop / pipeline
    index_path: str = ""               # 索引路径
    raw_dir: str = ""                  # raw 目录
    model: str = ""                    # 模型名 (默认使用配置)
    description: str = ""              # 可选的描述


@dataclass
class QueryCase:
    """单条测试查询."""

    id: str = ""
    question: str = ""
    standard_answer: str = ""
    domain: str = ""
    difficulty: str = ""
    query_type: str = ""
    source_file: str = ""
    keywords: list[str] = field(default_factory=list)

    @staticmethod
    def from_benchmark_case(c: dict) -> QueryCase:
        return QueryCase(
            id=c.get("id", ""),
            question=c.get("question", ""),
            standard_answer=c.get("answer", ""),
            domain=c.get("domain", ""),
            difficulty=c.get("difficulty", ""),
            query_type=c.get("query_type", ""),
            source_file=c.get("source_file", ""),
            keywords=c.get("keywords", []),
        )


@dataclass
class SingleResult:
    """单个查询在单次运行中的结果."""

    query_id: str = ""
    question: str = ""
    answer: str = ""
    standard_answer: str = ""
    success: bool = False
    accuracy: float = 0.0            # LLM 语义评分 (0.0/0.5/1.0)
    similarity: float = 0.0          # bigram Jaccard 相似度
    latency: float = 0.0             # 处理时间 (秒)
    tokens_used: int = 0             # Token 消耗
    cost_cents: float = 0.0          # 费用估算 (分)
    tool_calls_count: int = 0        # 工具调用次数
    error: str = ""


@dataclass
class ArmResult:
    """单臂在多次运行中的汇总结果."""

    config: RunnerConfig = field(default_factory=RunnerConfig)
    per_query: dict[str, list[SingleResult]] = field(default_factory=dict)  # query_id -> [run0, run1, ...]
    runs_completed: int = 0

    def get_accuracy(self) -> list[float]:
        """所有查询×所有运行的正确率列表."""
        scores = []
        for _qid, results in self.per_query.items():
            for r in results:
                scores.append(r.accuracy)
        return scores

    def get_latency(self) -> list[float]:
        vals = []
        for _qid, results in self.per_query.items():
            for r in results:
                vals.append(r.latency)
        return vals

    def get_tokens(self) -> list[int]:
        vals = []
        for _qid, results in self.per_query.items():
            for r in results:
                vals.append(r.tokens_used)
        return vals

    def get_cost(self) -> list[float]:
        vals = []
        for _qid, results in self.per_query.items():
            for r in results:
                if hasattr(r, 'cost_cents'):
                    vals.append(r.cost_cents)
        return vals

    def mean(self, values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    def stdev(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = self.mean(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    def summary(self) -> dict:
        acc = self.get_accuracy()
        lat = self.get_latency()
        tok = self.get_tokens()
        cst = self.get_cost()
        return {
            "config": asdict(self.config),
            "runs": self.runs_completed,
            "accuracy_mean": round(self.mean(acc), 3),
            "accuracy_std": round(self.stdev(acc), 3),
            "latency_mean": round(self.mean(lat), 2),
            "latency_std": round(self.stdev(lat), 2),
            "tokens_mean": round(self.mean(tok), 0),
            "tokens_std": round(self.stdev(tok), 0),
            "cost_cents_mean": round(self.mean(cst), 2),
            "cost_cents_std": round(self.stdev(cst), 2),
            "total_cost_cents": round(sum(cst), 2),
        }


@dataclass
class ABTestResult:
    """完整的 A/B 测试结果."""

    config_a: RunnerConfig = field(default_factory=RunnerConfig)
    config_b: RunnerConfig = field(default_factory=RunnerConfig)
    arm_a: ArmResult = field(default_factory=ArmResult)
    arm_b: ArmResult = field(default_factory=ArmResult)
    queries_count: int = 0
    runs_per_arm: int = 0
    started_at: str = ""
    finished_at: str = ""

    def summary(self) -> str:
        lines = ["=" * 70]
        lines.append(f"A/B 测试结果 — {self.config_a.name} vs {self.config_b.name}")
        lines.append(f"查询数: {self.queries_count} | 运行轮次: {self.runs_per_arm} | 总运行: {self.queries_count * self.runs_per_arm * 2}")
        lines.append("=" * 70)

        # 汇总统计
        sa = self.arm_a.summary()
        sb = self.arm_b.summary()
        lines.append(f"\n{'指标':<20s} {'A':>12s} {'B':>12s} {'差异':>12s} {'显著性':>10s}")

        for metric, unit, lower_better in [
            ("accuracy", "", False),
            ("latency", "s", True),
            ("tokens", "", True),
            ("cost_cents", "¢", True),
        ]:
            ma = sa[f"{metric}_mean"]
            mb = sb[f"{metric}_mean"]
            std_a = sa.get(f"{metric}_std", 0)
            std_b = sb.get(f"{metric}_std", 0)
            diff = mb - ma if lower_better else ma - mb

            # Z-test for accuracy, otherwise compare means
            if metric == "accuracy":
                sig = self._z_test(self.arm_a.get_accuracy(), self.arm_b.get_accuracy())
            else:
                sig = self._z_test_means(ma, std_a, len(self.arm_a.get_accuracy()),
                                          mb, std_b, len(self.arm_b.get_accuracy()))

            # Cohen's d
            pooled_std = math.sqrt((std_a**2 + std_b**2) / 2) if std_a or std_b else 1
            d = (mb - ma) / max(pooled_std, 0.001)

            lines.append(
                f"{metric:<20s} {ma:>8.2f}{unit:<4s} {mb:>8.2f}{unit:<4s} "
                f"{diff:>+8.2f}{unit:<4s} {'p<0.05' if sig else 'n.s.':>10s}"
            )

        lines.append("\n--- 效果量 (Cohen's d) ---")
        for metric in ["accuracy", "latency", "tokens", "cost_cents"]:
            d = self._cohens_d(
                self.arm_a.summary().get(f"{metric}_mean", 0),
                self.arm_b.summary().get(f"{metric}_mean", 0),
                self.arm_a.summary().get(f"{metric}_std", 0),
                self.arm_b.summary().get(f"{metric}_std", 0),
            )
            label = {"accuracy": "正确率", "latency": "延迟", "tokens": "Token", "cost_cents": "费用"}.get(metric, metric)
            eff = "大" if abs(d) > 0.8 else ("中" if abs(d) > 0.5 else ("小" if abs(d) > 0.2 else "极小"))
            lines.append(f"  {label}: d={d:.3f} ({eff})")

        lines.append(f"\n⏱ {self.started_at} → {self.finished_at}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "config_a": asdict(self.config_a),
            "config_b": asdict(self.config_b),
            "summary_a": self.arm_a.summary(),
            "summary_b": self.arm_b.summary(),
            "queries_count": self.queries_count,
            "runs_per_arm": self.runs_per_arm,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def save_json(self, path: str):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 统计方法 ──

    def _z_test(self, scores_a: list[float], scores_b: list[float]) -> bool:
        """双比例 z-test. 返回是否显著 (p<0.05)."""
        n = min(len(scores_a), len(scores_b))
        if n < 10:
            return False
        # 二值化: >=0.5 为正确
        correct_a = sum(1 for s in scores_a[:n] if s >= 0.5)
        correct_b = sum(1 for s in scores_b[:n] if s >= 0.5)
        p1 = correct_a / n
        p2 = correct_b / n
        pooled = (correct_a + correct_b) / (2 * n)
        if pooled == 0 or pooled == 1:
            return False
        se = math.sqrt(pooled * (1 - pooled) * 2 / n)
        z = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return p_value < 0.05

    def _z_test_means(self, m1, std1, n1, m2, std2, n2) -> bool:
        """双样本 z-test (均值)."""
        if n1 < 5 or n2 < 5:
            return False
        se = math.sqrt(std1**2 / n1 + std2**2 / n2)
        if se == 0:
            return False
        z = (m2 - m1) / se
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return p_value < 0.05

    def _cohens_d(self, m1, m2, std1, std2) -> float:
        """Cohen's d 效果量."""
        pooled = math.sqrt((std1**2 + std2**2) / 2) if std1 or std2 else 1
        if pooled == 0:
            return 0.0
        return (m2 - m1) / pooled


# ── 查询加载 ────────────────────────────────────────────────────────


def load_queries_from_benchmark(path: str, domain: str | None = None,
                                 difficulty: str | None = None,
                                 limit: int | None = None) -> list[QueryCase]:
    """从 qa_benchmark_cases.json 加载测试查询.

    Args:
        path: JSON 文件路径 (如 docs/qa_benchmark_cases.json)
        domain: 筛选领域 (如 "人事管理", None=全部)
        difficulty: 筛选难度 (easy/medium/hard, None=全部)
        limit: 最大查询数

    Returns:
        List[QueryCase] 列表
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", data) if isinstance(data, dict) else data
    result = []
    for c in cases:
        if domain and c.get("domain") != domain:
            continue
        if difficulty and c.get("difficulty") != difficulty:
            continue
        result.append(QueryCase.from_benchmark_case(c))
    if limit:
        result = result[:limit]
    return result


# ── 成本估算 ────────────────────────────────────────────────────────


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """估算 LLM 调用费用 (分, cents).

    参考价格 (每百万 token):
    - DeepSeek V4 Flash: input $0.30 / output $0.60
    - GLM-4: input $0.50 / output $1.00
    """
    rates = {
        "deepseek": (0.30, 0.60),
        "glm": (0.50, 1.00),
    }
    model_lower = model.lower()
    if "deepseek" in model_lower:
        in_rate, out_rate = rates["deepseek"]
    else:
        in_rate, out_rate = rates["glm"]
    cost = (input_tokens / 1_000_000 * in_rate + output_tokens / 1_000_000 * out_rate) * 100
    return round(cost, 4)


# ── Agent 执行器 ──────────────────────────────────────────────────────


def run_agent_query(config: RunnerConfig, query: str, query_id: str = "") -> SingleResult:
    """执行单个 Agent 查询并返回结果.

    同步调用 CLI agent 或 MCP pipeline.
    返回包含答案、耗时、Token 消耗的结构化结果。
    """
    from src.agent.base import AgentResponse
    from src.agent.search_agent import create_search_agent
    from src.utils.config import Config

    cfg = Config.from_env()
    if config.model:
        cfg.llm_model = config.model

    t0 = time.time()
    result = SingleResult(query_id=query_id, question=query)

    try:
        agent = create_search_agent(
            config=cfg,
            index_path=config.index_path,
            raw_dir=config.raw_dir or None,
            use_rerank=True,
            mode=config.mode,
        )
        agent._no_log = True

        response: AgentResponse = agent.run(query=query)
        elapsed = time.time() - t0

        result.success = response.success
        result.answer = response.answer or ""
        result.latency = round(elapsed, 2)
        result.tokens_used = response.tokens_used or 0
        result.tool_calls_count = len(getattr(response, "tool_calls", []))
        result.cost_cents = estimate_cost(
            config.model or cfg.llm_model,
            response.tokens_used or 0,
            response.tokens_used or 0,
        )

    except Exception as e:
        elapsed = time.time() - t0
        result.success = False
        result.error = str(e)[:200]
        result.latency = round(elapsed, 2)

    return result


# ── 评分 ────────────────────────────────────────────────────────────


def calc_similarity(text1: str, text2: str) -> float:
    """字符 bigram Jaccard 相似度."""
    if not text1 or not text2:
        return 0.0
    def bigrams(text):
        chars = list(text.replace(" ", "").replace("\n", ""))
        return set(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    b1 = bigrams(text1)
    b2 = bigrams(text2)
    if not b1 or not b2:
        return 0.0
    intersection = b1 & b2
    union = b1 | b2
    return round(len(intersection) / len(union), 3)


def llm_score_answer(question: str, standard_answer: str, agent_answer: str) -> float:
    """LLM 语义评分 (0.0/0.5/1.0)."""
    if not agent_answer or not standard_answer:
        return 0.0
    try:
        from src.agent.llm_client import ChatMessage, LLMClient
        from src.utils.config import Config
        config = Config.from_env()
        llm = LLMClient(config)

        prompt = (
            "你是一个严谨的评分员。请判断以下 Agent 的回答是否与标准答案在语义上一致。\n\n"
            f"问题: {question}\n\n"
            f"标准答案: {standard_answer}\n\n"
            f"Agent 回答: {agent_answer}\n\n"
            "评分标准:\n"
            "1.0 — 语义完全一致，核心信息都覆盖\n"
            "0.5 — 部分一致，有遗漏但未错\n"
            "0.0 — 语义不一致、错误、或无法回答\n\n"
            "只输出分数，不要输出其他内容:"
        )
        resp = llm.chat(messages=[ChatMessage(role="user", content=prompt)],
                        temperature=0.1, max_tokens=10)
        score_str = resp.content.strip()
        for val in ["0.0", "0.5", "1.0"]:
            if val in score_str:
                return float(val)
        return 0.5
    except Exception:
        return 0.0


# ── 主运行器 ────────────────────────────────────────────────────────


class ABTestRunner:
    """统一 A/B 测试运行器.

    用法::

        runner = ABTestRunner()
        result = runner.run(config_a, config_b, queries, runs=3)
        print(result.summary())
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def run(self, config_a: RunnerConfig, config_b: RunnerConfig,
            queries: list[QueryCase], runs: int = 1) -> ABTestResult:
        """运行 A/B 测试.

        Args:
            config_a: A 配置.
            config_b: B 配置.
            queries: 测试查询列表.
            runs: 每臂运行轮次 (用于测量方差).

        Returns:
            ABTestResult 包含完整结果.
        """
        started_at = datetime.now().isoformat()
        result = ABTestResult(
            config_a=config_a,
            config_b=config_b,
            arm_a=ArmResult(config=config_a),
            arm_b=ArmResult(config=config_b),
            queries_count=len(queries),
            runs_per_arm=runs,
            started_at=started_at,
        )

        # 顺序随机化: 交替 A→B / B→A
        for run_idx in range(runs):
            # 随机化查询顺序
            shuffled = list(queries)
            random.shuffle(shuffled)

            # 交替臂顺序
            arms = [(config_a, result.arm_a), (config_b, result.arm_b)]
            if run_idx % 2 == 1:
                arms.reverse()

            for qcase in shuffled:
                for cfg, arm in arms:
                    sr = run_agent_query(cfg, qcase.question, qcase.id)
                    sr.standard_answer = qcase.standard_answer
                    sr.similarity = calc_similarity(sr.answer, qcase.standard_answer)
                    sr.accuracy = llm_score_answer(qcase.question, qcase.standard_answer, sr.answer)

                    if qcase.id not in arm.per_query:
                        arm.per_query[qcase.id] = []
                    arm.per_query[qcase.id].append(sr)

            result.arm_a.runs_completed = run_idx + 1
            result.arm_b.runs_completed = run_idx + 1

        result.finished_at = datetime.now().isoformat()
        return result
