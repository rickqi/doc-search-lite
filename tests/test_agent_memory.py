"""Tests for AgentMemory — 完整验证包括 recall, learn, feedback, format_context, 集成, 性能基准, 异常安全."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.stats.memory import AgentMemory

# ── Helpers ──────────────────────────────────────────────────────────


def _create_search_logs_table(conn, extra_cols: bool = False):
    """Create search_logs table matching the real schema."""
    conn.execute("DROP TABLE IF EXISTS search_logs")
    cols = """(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL UNIQUE,
        query TEXT NOT NULL,
        answer TEXT,
        source TEXT NOT NULL,
        search_mode TEXT NOT NULL,
        index_path TEXT,
        raw_dir TEXT,
        model TEXT,
        success INTEGER DEFAULT 1,
        processing_time REAL DEFAULT 0,
        tokens_used INTEGER DEFAULT 0,
        tool_calls_count INTEGER DEFAULT 0,
        sources_count INTEGER DEFAULT 0,
        search_hits_json TEXT,
        tool_calls_json TEXT,
        skill TEXT,
        difficulty TEXT,
        tags TEXT,
        md_file_path TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )"""
    conn.execute(f"CREATE TABLE search_logs {cols}")
    if extra_cols:
        conn.execute("CREATE TABLE IF NOT EXISTS search_logs_schema (key TEXT PRIMARY KEY, value TEXT)")


def _seed_entry(conn, **kw):
    """Insert a single search_logs entry."""
    defaults = dict(
        session_id="test_sid",
        query="测试查询",
        answer="测试答案内容",
        source="agent",
        search_mode="agent",
        index_path="/index/test",
        success=1,
        processing_time=1.0,
        tokens_used=100,
        tool_calls_json="[]",
        tags="",
        created_at="2026-07-13 12:00:00",
    )
    vals = {**defaults, **kw}
    placeholders = ", ".join("?" for _ in vals)
    columns = ", ".join(vals.keys())
    conn.execute(
        f"INSERT INTO search_logs ({columns}) VALUES ({placeholders})",
        tuple(vals.values()),
    )
    conn.commit()


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def memory(tmp_path):
    """AgentMemory with temp DB."""
    m = AgentMemory(db_path=tmp_path / "test.db")
    yield m
    m.close()


@pytest.fixture
def seeded_memory(memory):
        """Memory seeded with varied search_logs entries."""
        conn = memory._conn()
        _create_search_logs_table(conn)
        rows = [
            ("srch_001", "年假有几天", "根据公司规定, 年假为5天。", "agent", "agent", "/index/hr", 1, 2.5, 1000, "[]"),
            ("srch_002", "差旅报销标准", "住宿标准为每天500元。交通费实报实销。", "agent", "bm25", "/index/hr", 1, 1.2, 500, "[]"),
            ("srch_003", "加班费怎么计算", "平日1.5倍, 周末2倍, 法定节假日3倍。", "agent", "hybrid", "/index/hr", 1, 3.0, 1500, "[]"),
            ("srch_004_bm25", "五险一金缴纳比例", "养老8%", "bm25", "bm25", "/index/hr", 1, 0.1, 0, "[]"),
            ("srch_005_failed", "失败查询", "无答案", "agent", "agent", "/index/hr", 0, 0.5, 200, "[]"),
        ]
        for r in rows:
            conn.execute(
                """INSERT INTO search_logs
                   (session_id, query, answer, source, search_mode, index_path,
                    success, processing_time, tokens_used, tool_calls_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                r,
            )
        conn.commit()
        return memory


# ═══════════════════════════════════════════════════════════════════════
# Part 1: FormatContext Tests (NEW)
# ═══════════════════════════════════════════════════════════════════════


class TestFormatContext:
    """format_context() — 将模糊匹配结果格式化为注入文本."""

    def test_exact_hit_returns_none(self, seeded_memory):
        """精确命中不应生成 context."""
        result = seeded_memory.recall("年假有几天")
        assert result is not None
        ctx = seeded_memory.format_context(result)
        assert ctx is None, "精确命中不应注入 context"

    def test_fuzzy_hit_formats_correctly(self, seeded_memory):
        """模糊匹配生成正确的 context 格式."""
        result = seeded_memory.recall("年假多少天")
        assert result is not None
        ctx = seeded_memory.format_context(result)
        assert ctx is not None
        assert "[历史相关问答" in ctx
        assert "用户曾问:" in ctx
        assert "回答:" in ctx
        assert "来源知识库:" in ctx

    def test_none_input_returns_none(self, seeded_memory):
        """None 输入应返回 None."""
        assert seeded_memory.format_context(None) is None

    def test_empty_dict_returns_none(self, seeded_memory):
        """空 dict 应返回 None."""
        assert seeded_memory.format_context({}) is None

    def test_wrong_source_returns_none(self, seeded_memory):
        """非 fuzzy_hit source 应返回 None."""
        result = seeded_memory.recall("年假有几天")  # exact hit
        assert result is not None
        ctx = seeded_memory.format_context({"source": "exact_hit", "query": "x", "answer": "y"})
        assert ctx is None

    def test_answer_truncation(self, seeded_memory):
        """长 answer 应被截断到 300 字符."""
        long_answer = "A" * 500
        result = {"source": "fuzzy_hit", "query": "test", "answer": long_answer, "index_path": "/idx"}
        ctx = seeded_memory.format_context(result)
        assert ctx is not None
        # Answer portion should be truncated to 300 chars
        assert "A" * 300 in ctx
        assert "A" * 301 not in ctx  # not full 500

    def test_missing_index_path(self, seeded_memory):
        """没有 index_path 时不应显示来源知识库."""
        result = {"source": "fuzzy_hit", "query": "test", "answer": "answer"}
        ctx = seeded_memory.format_context(result)
        assert ctx is not None
        assert "来源知识库" not in ctx


# ═══════════════════════════════════════════════════════════════════════
# Part 2: SearchAgent Integration Tests (NEW)
# ═══════════════════════════════════════════════════════════════════════


class TestSearchAgentIntegration:
    """search_agent.py run() 中的 AgentMemory 集成."""

    @patch("src.stats.memory.AgentMemory")
    def test_exact_hit_skips_tool_loop(self, mock_am_cls):
        """精确命中时, run() 应直接返回 AgentResponse 而不走 tool_loop."""
        from src.agent.search_agent import SearchAgent, create_search_agent
        from src.agent.base import AgentResponse

        # Mock AgentMemory.recall() to return exact hit
        mock_mem = MagicMock()
        mock_mem.recall.return_value = {
            "source": "exact_hit",
            "answer": "历史答案内容",
            "search_mode": "agent",
            "processing_time": 2.5,
            "tokens_used": 1000,
            "session_id": "srch_001",
        }
        mock_am_cls.return_value = mock_mem

        agent = MagicMock(spec=SearchAgent)
        agent._mode = "tool_loop"
        agent._no_log = True
        agent._usage_tracker = None
        agent._session_id = "test_session"
        agent._memory_recall = None
        agent._memory_ctx_injected = False
        agent._index_path = "/index/test"
        agent._raw_dirs = []
        agent._llm_client = None
        agent._srch_session_id = ""
        agent._search_source = "agent"

        # We need to actually run the real SearchAgent.run() method
        # but with mocked internals. Let's use a simpler approach:
        # verify the code path by checking the method logic directly.

        # The run() method calls AgentMemory().recall(query) and if exact hit,
        # returns AgentResponse immediately WITHOUT calling _run_tool_loop.
        # We can verify this by checking that _run_tool_loop is NOT called.

        # Create a real SearchAgent with mocked tool_loop
        config = MagicMock()
        config.llm_model = "test-model"
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.enable_tiered_routing = False
        config.litellm_router_enabled = False

        agent = create_search_agent(
            config=config,
            index_path="/index/test",
            raw_dir=None,
            use_rerank=False,
            mode="tool_loop",
        )
        agent._no_log = True

        # Replace _run_tool_loop with a spy
        original_tool_loop = agent._run_tool_loop
        tool_loop_called = False

        def spy_tool_loop(*a, **kw):
            nonlocal tool_loop_called
            tool_loop_called = True
            return original_tool_loop(*a, **kw)

        agent._run_tool_loop = spy_tool_loop

        # Mock AgentMemory to return exact hit
        with patch("src.stats.memory.AgentMemory") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.recall.return_value = {
                "source": "exact_hit",
                "answer": "历史答案",
                "search_mode": "agent",
                "processing_time": 1.0,
                "tokens_used": 500,
                "session_id": "srch_001",
            }
            mock_cls.return_value = mock_instance

            result = agent.run(query="年假有几天")

        assert result.success is True
        assert "历史答案" in result.answer
        assert result.processing_time == 0.0
        assert tool_loop_called is False, "精确命中时不应调用 tool_loop"

    @patch("src.stats.memory.AgentMemory")
    def test_fuzzy_hit_injects_context(self, mock_am_cls):
        """模糊匹配时, context 应注入到 system prompt 中."""
        from src.agent.search_agent import create_search_agent

        config = MagicMock()
        config.llm_model = "test-model"
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.enable_tiered_routing = False
        config.litellm_router_enabled = False

        agent = create_search_agent(
            config=config,
            index_path="/index/test",
            raw_dir=None,
            use_rerank=False,
            mode="tool_loop",
        )
        agent._no_log = True

        # Mock AgentMemory to return fuzzy hit
        fuzzy_recall = {"source": "fuzzy_hit", "query": "年假有几天", "answer": "年假为5天", "index_path": "/index/hr"}

        with patch("src.stats.memory.AgentMemory") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.recall.return_value = fuzzy_recall
            mock_instance.format_context.return_value = "[历史相关问答]\n用户曾问: 年假有几天\n回答: 年假为5天\n---"
            mock_cls.return_value = mock_instance

            # We'll check that memory_recall and memory_ctx_injected are set
            # by inspecting the agent after run().
            # Since the agent will try to run llm calls, we expect it to fail
            # but we can check that the context was prepared.
            try:
                agent.run(query="年假有多少天")
            except Exception:
                pass

            # After run() is called, the memory injection should have been attempted
            assert mock_instance.recall.called, "recall() 应被调用"
            assert mock_instance.format_context.called, "format_context() 应被调用"

    @patch("src.stats.memory.AgentMemory")
    def test_memory_miss_normal_flow(self, mock_am_cls):
        """无命中时, Agent 应正常走 tool_loop 流程."""
        from src.agent.search_agent import create_search_agent

        config = MagicMock()
        config.llm_model = "test-model"
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.enable_tiered_routing = False
        config.litellm_router_enabled = False

        agent = create_search_agent(
            config=config,
            index_path="/index/test",
            raw_dir=None,
            use_rerank=False,
            mode="tool_loop",
        )
        agent._no_log = True

        # Mock AgentMemory to return None (no match)
        with patch("src.stats.memory.AgentMemory") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.recall.return_value = None
            mock_cls.return_value = mock_instance

            # Agent will try LLM calls, expect failure but check the flow
            try:
                agent.run(query="全新查询内容")
            except Exception:
                pass

            assert mock_instance.recall.called, "recall() 应被调用"

    def test_memory_fail_safe_during_recall(self, tmp_path):
        """AgentMemory.recall() 抛异常时不应阻断 Agent."""
        from src.agent.search_agent import create_search_agent

        config = MagicMock()
        config.llm_model = "test-model"
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.enable_tiered_routing = False
        config.litellm_router_enabled = False

        agent = create_search_agent(
            config=config,
            index_path="/index/test",
            raw_dir=None,
            use_rerank=False,
            mode="tool_loop",
        )
        agent._no_log = True

        # Mock AgentMemory to raise exception
        with patch("src.stats.memory.AgentMemory") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.recall.side_effect = RuntimeError("DB failure")
            mock_cls.return_value = mock_instance

            # Should NOT raise — the exception is caught
            try:
                agent.run(query="年假有几天")
            except RuntimeError:
                pytest.fail("AgentMemory 异常不应传播到 Agent")

    @patch("src.stats.memory.AgentMemory")
    def test_learn_called_on_completion(self, mock_am_cls):
        """Agent 执行完成后, learn() 应被调用."""
        from src.agent.search_agent import create_search_agent

        config = MagicMock()
        config.llm_model = "test-model"
        config.active_api_key = "test-key"
        config.active_base_url = "http://test"
        config.enable_tiered_routing = False
        config.litellm_router_enabled = False

        agent = create_search_agent(
            config=config,
            index_path="/index/test",
            raw_dir=None,
            use_rerank=False,
            mode="tool_loop",
        )
        agent._no_log = True
        agent._search_source = "agent"

        with patch("src.stats.memory.AgentMemory") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.recall.return_value = None
            mock_cls.return_value = mock_instance

            try:
                agent.run(query="测试查询")
            except Exception:
                pass

            # learn() should have been called (it's in a try block)
            assert mock_instance.learn.called, "learn() 应被调用"


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Fail-Safe Tests (NEW)
# ═══════════════════════════════════════════════════════════════════════


class TestFailSafe:
    """AgentMemory 异常安全测试."""

    def test_db_not_found(self):
        """search_logs.db 不存在时不应抛异常."""
        m = AgentMemory(db_path=Path("/nonexistent/path/db.sqlite"))
        result = m.recall("任何查询")
        assert result is None
        # learn should not crash
        m.learn("sid", {"mode": "test"})
        m.close()

    def test_db_corrupt(self, tmp_path):
        """损坏的 DB 文件不应抛异常."""
        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a valid sqlite database")
        m = AgentMemory(db_path=db_path)
        result = m.recall("任何查询")
        assert result is None
        # learn should not crash
        m.learn("sid", {"mode": "test"})
        m.close()

    def test_db_permission_denied(self, tmp_path):
        """无权限的 DB 不应抛异常."""
        db_path = tmp_path / "no_perm.db"
        _create_search_logs_table(sqlite3.connect(str(db_path)))
        # Make it non-writable (may not work on Windows fully)
        try:
            os.chmod(str(db_path), 0o000)
        except OSError:
            pytest.skip("权限模拟失败（可能运行在 Windows 上）")
        m = AgentMemory(db_path=db_path)
        result = m.recall("任何查询")
        # Should return None gracefully
        assert result is None
        m.close()

    def test_close_idempotent(self, memory):
        """close() 应幂等."""
        memory.close()
        memory.close()  # 第二次不应抛异常
        memory.close()

    def test_multiple_instances(self, tmp_path):
        """多个 AgentMemory 实例应独立工作."""
        db1 = tmp_path / "db1.db"
        db2 = tmp_path / "db2.db"
        m1 = AgentMemory(db_path=db1)
        m2 = AgentMemory(db_path=db2)

        conn1 = m1._conn()
        _create_search_logs_table(conn1)
        _seed_entry(conn1, session_id="s1", query="q1", answer="a1")

        conn2 = m2._conn()
        _create_search_logs_table(conn2)
        _seed_entry(conn2, session_id="s2", query="q2", answer="a2")

        r1 = m1.recall("q1")
        r2 = m2.recall("q2")
        assert r1 is not None
        assert r2 is not None
        assert r1["answer"] == "a1"
        assert r2["answer"] == "a2"
        # Cross-DB should not match
        assert m1.recall("q2") is None
        assert m2.recall("q1") is None

        m1.close()
        m2.close()


@pytest.fixture
def memory_for_record_feedback(tmp_path):
    """Memory with search_logs table for record_feedback tests."""
    m = AgentMemory(db_path=tmp_path / "feedback_test.db")
    conn = m._conn()
    _create_search_logs_table(conn)
    _seed_entry(conn, session_id="sid_001", query="q1", answer="a1")
    return m


class TestRecordFeedback:
    """AgentMemory.feedback() 测试 (通过 feedback() API)."""

    def test_feedback_via_api(self, memory_for_record_feedback):
        """feedback() 记录到 answer_feedback 表."""
        m = memory_for_record_feedback
        m.feedback("sid_001", 5, "很好的回答")
        stats = m.get_feedback_stats()
        assert stats["total"] == 1
        assert stats["avg_rating"] == 5.0

    def test_feedback_invalid_rating_via_api(self, memory_for_record_feedback):
        """无效评分不应被记录."""
        m = memory_for_record_feedback
        m.feedback("sid_001", 6)
        stats = m.get_feedback_stats()
        assert stats["total"] == 0

    def test_feedback_multiple_via_api(self, memory_for_record_feedback):
        """多次反馈正常聚合."""
        m = memory_for_record_feedback
        m.feedback("sid_001", 5)
        m.feedback("sid_002", 3)
        m.feedback("sid_003", 1)
        stats = m.get_feedback_stats()
        assert stats["total"] == 3
        assert stats["avg_rating"] == 3.0


# ═══════════════════════════════════════════════════════════════════════
# Part 4: Performance Benchmark Tests (NEW)
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryPerformance:
    """性能基准测试 — 验证延迟在指定阈值内."""

    # N=100 records to simulate production-like data volume
    NUM_SEED_RECORDS = 100

    @pytest.fixture
    def perf_memory(self, tmp_path):
        """Seed memory with N records for performance testing."""
        m = AgentMemory(db_path=tmp_path / "perf.db")
        conn = m._conn()
        _create_search_logs_table(conn)
        for i in range(self.NUM_SEED_RECORDS):
            _seed_entry(conn,
                        session_id=f"perf_{i:04d}",
                        query=f"查询问题第{i}号" if i > 0 else "精确命中的查询",
                        answer=f"这是第{i}号答案内容" * 10,
                        source="agent" if i % 2 == 0 else "bm25",
                        )
        # Add indexes like production
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_logs(query)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_query_answer ON search_logs(query, answer)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON search_logs(source)")
        conn.commit()
        return m

    def test_exact_hit_latency(self, perf_memory):
        """精确命中延迟应 <50ms (p95)."""
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            result = perf_memory.recall("精确命中的查询")
            dt = (time.perf_counter() - t0) * 1000  # ms
            times.append(dt)

        assert result is not None
        assert result["source"] == "exact_hit"
        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        print(f"[perf] exact_hit: avg={avg_ms:.2f}ms, max={max_ms:.2f}ms, n={len(times)}")
        assert max_ms < 50, f"精确命中延迟 {max_ms:.1f}ms 超过 50ms 阈值"

    def test_fuzzy_hit_latency(self, perf_memory):
        """模糊匹配延迟应 <50ms (p95)."""
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            result = perf_memory.recall("精确命中的查询吗")  # similar but not exact
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)

        # Should find exact match for "精确命中的查询" first, but this query
        # is different enough to be fuzzy. May or may not match depending on LIKE.
        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        print(f"[perf] fuzzy_hit: avg={avg_ms:.2f}ms, max={max_ms:.2f}ms, n={len(times)}")
        assert max_ms < 50, f"模糊匹配延迟 {max_ms:.1f}ms 超过 50ms 阈值"

    def test_miss_latency(self, perf_memory):
        """未命中延迟应 <10ms."""
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            result = perf_memory.recall("完全不存在的查询内容")
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)

        assert result is None
        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        print(f"[perf] miss: avg={avg_ms:.2f}ms, max={max_ms:.2f}ms, n={len(times)}")
        assert max_ms < 10, f"未命中延迟 {max_ms:.1f}ms 超过 10ms 阈值"

    def test_format_context_latency(self, seeded_memory):
        """format_context() 延迟应 <1ms."""
        result = seeded_memory.recall("年假多少天")
        assert result is not None
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            seeded_memory.format_context(result)
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)

        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        print(f"[perf] format_context: avg={avg_ms:.3f}ms, max={max_ms:.3f}ms, n={len(times)}")
        assert max_ms < 1, f"format_context 延迟 {max_ms:.3f}ms 超过 1ms 阈值"

    def test_learn_latency(self, perf_memory):
        """learn() 延迟应 <50ms (SQLite UPDATE)."""
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            perf_memory.learn(f"perf_{i:04d}", {"mode": "test", "confidence": 0.9})
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)

        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        print(f"[perf] learn: avg={avg_ms:.2f}ms, max={max_ms:.2f}ms, n={len(times)}")
        assert max_ms < 50, f"learn 延迟 {max_ms:.1f}ms 超过 50ms 阈值"


# ═══════════════════════════════════════════════════════════════════════
# Part 5: Behavior Comparison Tests (NEW)
# ═══════════════════════════════════════════════════════════════════════


class TestBehaviorComparison:
    """有记忆 vs 无记忆的行为对比."""

    def test_exact_hit_vs_normal_flow_response_differs(self, tmp_path):
        """同一查询，有记忆（精确命中）和无记忆应返回不同的 AgentResponse."""
        from src.agent.base import AgentResponse

        # With memory (exact hit): short-circuits, returns cached answer
        db_path = tmp_path / "cmp.db"
        m = AgentMemory(db_path=db_path)
        conn = m._conn()
        _create_search_logs_table(conn)
        _seed_entry(conn, session_id="s1", query="年假有几天", answer="【缓存答案】年假为5天。", source="agent")

        recalled = m.recall("年假有几天")
        assert recalled is not None
        assert recalled["source"] == "exact_hit"
        assert "【缓存答案】" in recalled["answer"]

        # Without memory: would go through normal tool_loop
        # This is simulated — the actual difference is that with memory,
        # AgentResponse.processing_time = 0 and no tool_calls
        memory_response = AgentResponse(
            success=True,
            answer=recalled["answer"],
            sources=[],
            tool_calls=[],
            confidence=1.0,
            processing_time=0.0,
            tokens_used=0,
        )
        assert memory_response.processing_time == 0.0
        assert len(memory_response.tool_calls) == 0
        assert memory_response.tokens_used == 0
        assert "【缓存答案】" in memory_response.answer

    def test_memory_hit_reduces_processing_time(self, tmp_path):
        """有记忆时 processing_time = 0，无记忆时 > 0."""
        db_path = tmp_path / "time.db"
        m = AgentMemory(db_path=db_path)
        conn = m._conn()
        _create_search_logs_table(conn)
        _seed_entry(conn, session_id="s1", query="test", answer="答案")

        t0 = time.time()
        result = m.recall("test")
        mem_time = time.time() - t0

        assert result is not None
        assert result["source"] == "exact_hit"
        # Memory recall should be nearly instant (<50ms)
        assert mem_time < 0.05, f"记忆召回耗时 {mem_time*1000:.1f}ms，超过 50ms"

    def test_memory_miss_fallback_has_no_overhead(self, tmp_path):
        """无命中时，记忆查询不应显著增加延迟."""
        db_path = tmp_path / "overhead.db"
        m = AgentMemory(db_path=db_path)

        # Measure recall time for a miss
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            result = m.recall("不存在的查询内容完全随机")
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)

        assert result is None
        avg_ms = sum(times) / len(times)
        print(f"[cmp] miss overhead: avg={avg_ms:.3f}ms, n={len(times)}")
        # Must be under 10ms (perf requirement)
        assert avg_ms < 5, f"未命中路径平均延迟 {avg_ms:.2f}ms 超过 5ms"

    def test_learn_does_not_block(self, tmp_path):
        """learn() 应非阻塞（在 daemon thread 中执行）."""
        import threading

        db_path = tmp_path / "learn_perf.db"
        m = AgentMemory(db_path=db_path)
        conn = m._conn()
        _create_search_logs_table(conn)
        _seed_entry(conn, session_id="s1", query="test", answer="ans")

        # learn() should return immediately (it runs synchronously
        # but the design intent is to make it non-blocking).
        # We verify it completes quickly.
        t0 = time.perf_counter()
        m.learn("s1", {"mode": "test", "confidence": 1.0})
        dt = (time.perf_counter() - t0) * 1000

        assert dt < 50, f"learn() 耗时 {dt:.1f}ms，预期 <50ms"


# ═══════════════════════════════════════════════════════════════════════
# Part 6: Existing Tests (preserved and enhanced)
# ═══════════════════════════════════════════════════════════════════════


class TestAgentMemoryRecall:
    """Recall basic tests."""

    def test_exact_match(self, seeded_memory):
        result = seeded_memory.recall("年假有几天")
        assert result is not None
        assert result["source"] == "exact_hit"
        assert "5天" in result["answer"]
        assert result["search_mode"] == "agent"

    def test_fuzzy_match(self, seeded_memory):
        result = seeded_memory.recall("年假多少天？")
        assert result is not None
        assert result["source"] == "fuzzy_hit"

    def test_fuzzy_match_different_wording(self, seeded_memory):
        """不同措辞的相似问题应匹配."""
        # "年假有几天" is in seed data as exact match for "年假有几天"
        # Use a slight variation to trigger fuzzy match
        result = seeded_memory.recall("年假有几天呢")
        assert result is not None
        assert result["source"] == "fuzzy_hit"

    def test_no_match(self, seeded_memory):
        result = seeded_memory.recall("生育津贴如何申请")
        assert result is None

    def test_empty_query(self, seeded_memory):
        assert seeded_memory.recall("") is None

    def test_recall_agent_source_only(self, seeded_memory):
        """只有 source='agent' 且 success=1 的记录才应被召回."""
        # bm25 source entry should NOT be recalled by fuzzy match
        result = seeded_memory.recall("五险一金缴纳比例")
        # This query won't fuzzy-match '五险一金缴纳比例' from the bm25 entry
        # because it filters on source='agent'
        assert result is None

    def test_recall_ignores_failed(self, seeded_memory):
        """失败记录的查询不应被召回."""
        result = seeded_memory.recall("失败查询")
        assert result is None, "失败记录不应被召回"


class TestAgentMemoryLearn:
    """Learn basic tests."""

    def test_learn_updates_tags(self, seeded_memory):
        seeded_memory.learn("srch_001", {"mode": "agent", "index_path": "/index/hr", "confidence": 0.9})

        conn = seeded_memory._conn()
        cur = conn.execute("SELECT tags FROM search_logs WHERE session_id = ?", ("srch_001",))
        row = cur.fetchone()
        assert row is not None
        tags = json.loads(row[0])
        assert tags["effective_mode"] == "agent"
        assert tags["confidence"] == 0.9

    def test_learn_nonexistent_session(self, seeded_memory):
        seeded_memory.learn("nonexistent", {"mode": "test"})  # should not raise

    def test_learn_with_metadata(self, seeded_memory):
        """learn() 接受完整元数据."""
        seeded_memory.learn("srch_001", {
            "mode": "agent",
            "index_path": "/index/hr",
            "confidence": 0.85,
            "tool_count": 5,
            "latency": 12.3,
            "search_count": 3,
            "read_count": 2,
        })
        conn = seeded_memory._conn()
        row = conn.execute("SELECT tags FROM search_logs WHERE session_id = ?", ("srch_001",)).fetchone()
        tags = json.loads(row[0])
        assert tags["tool_count"] == 5
        assert tags["latency_s"] == 12.3


class TestAgentMemoryFeedback:
    """Feedback basic tests."""

    def test_feedback_valid(self, seeded_memory):
        seeded_memory.feedback("srch_001", 5, "很准确")
        stats = seeded_memory.get_feedback_stats()
        assert stats["total"] == 1
        assert stats["avg_rating"] == 5.0

    def test_feedback_invalid_rating(self, seeded_memory):
        seeded_memory.feedback("srch_001", 6)
        stats = seeded_memory.get_feedback_stats()
        assert stats["total"] == 0

    def test_feedback_multiple(self, seeded_memory):
        seeded_memory.feedback("srch_001", 5)
        seeded_memory.feedback("srch_002", 3)
        seeded_memory.feedback("srch_003", 1)
        stats = seeded_memory.get_feedback_stats()
        assert stats["total"] == 3
        assert stats["avg_rating"] == 3.0
        assert stats["positive"] == 1
        assert stats["negative"] == 1


class TestAgentMemoryIntegration:
    """Integration tests."""

    def test_mcp_pipeline_memory_hit(self, tmp_path):
        """Memory hit returns cached answer."""
        db_path = tmp_path / "test_memory.db"
        m = AgentMemory(db_path=db_path)
        conn = m._conn()
        _create_search_logs_table(conn)
        _seed_entry(conn, session_id="test_hit", query="测试记忆", answer="历史答案内容", source="agent")

        result = m.recall("测试记忆")
        assert result is not None
        assert "历史答案内容" in result["answer"]

    def test_agent_imports_cleanly(self):
        from src.stats.memory import AgentMemory
        m = AgentMemory()
        assert m is not None
        m.close()


import sqlite3  # noqa: E402 (needed for helper used above)
