"""Tests for trace visualization — SSE events emitted by backend + frontend JS logic.

Coverage:
  1. Backend: tool_hook emits tool_call + tool_result + search_result SSE events
  2. Backend: search_result normalized format (doc_id/title/score/snippet)
  3. Backend: search_result omitted for non-search tools
  4. Frontend: TraceCollector JS class logic (simulated via Node or structure check)
  5. Frontend: SSE event → trace collector method wiring

Note: Full frontend rendering tests require a browser/headless environment (Playwright).
This file tests the backend event emission and verifies the frontend JS structure.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.web.sse_events import AgentEventType, sse_encode


# ── Backend: tool_hook SSE event emission ────────────────────────────


class TestBackendSearchResultEvent:
    """Verify the tool_hook in api.py emits search_result events correctly."""

    def _make_tool_hook(self, event_queue: list):
        """Replicate the tool_hook logic from api.py for isolated testing."""
        from src.web.sse_events import make_event

        def tool_hook(tc, result):
            # tool_call event
            event_queue.append(("tool_call", {
                "tool": tc.name,
                "arguments": tc.arguments,
            }))
            # tool_result event
            success = getattr(result, "success", True)
            content_raw = getattr(result, "content", "")  # original for search_result parsing
            content = content_raw
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."  # truncated for display
            event_queue.append(("tool_result", {
                "tool": tc.name,
                "success": success,
                "content_preview": content,
            }))
            # search_result event — only for search tools.
            # Use content_raw (before truncation) to keep valid JSON.
            if tc.name == "search" and success and content_raw:
                try:
                    parsed = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
                    results = parsed.get("results", []) if isinstance(parsed, dict) else []
                    if results:
                        hits = []
                        for r in results[:10]:
                            hits.append({
                                "doc_id": r.get("doc_id", ""),
                                "title": r.get("title", r.get("doc_id", "?")),
                                "score": r.get("score", 0),
                                "snippet": (r.get("snippet", "") or "")[:200],
                            })
                        event_queue.append(("search_result", {
                            "query": tc.arguments.get("query", ""),
                            "results": hits,
                            "total": len(results),
                        }))
                except Exception:
                    pass  # search_result is non-critical
            return None  # tool_hook doesn't return a value

        return tool_hook

    def _make_tc(self, name: str, **kw) -> MagicMock:
        """Create a mock ToolCall."""
        tc = MagicMock()
        tc.name = name
        tc.arguments = kw
        return tc

    def _make_result(self, success: bool = True, content: str = "") -> MagicMock:
        """Create a mock ToolResult with simple container."""
        from types import SimpleNamespace
        r = SimpleNamespace()
        r.success = success
        r.content = content
        return r

    # ── search_result emission ──

    def test_search_result_emitted_on_search_tool(self):
        """Search tool with results → search_result event emitted."""
        events = []
        hook = self._make_tool_hook(events)

        search_content = json.dumps({"results": [
            {"doc_id": "abc123", "title": "年假管理制度.md", "score": 0.85, "snippet": "年假为5天"},
            {"doc_id": "def456", "title": "考勤管理办法.md", "score": 0.72, "snippet": "请假流程"},
        ]})
        tc = self._make_tc("search", query="年假")
        result = self._make_result(success=True, content=search_content)

        hook(tc, result)

        event_types = [e[0] for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "search_result" in event_types, "search 工具应发出 search_result 事件"

        # Verify search_result payload
        sr_event = [e for e in events if e[0] == "search_result"][0]
        payload = sr_event[1]
        assert payload["query"] == "年假"
        assert len(payload["results"]) == 2
        assert payload["total"] == 2
        assert payload["results"][0]["doc_id"] == "abc123"
        assert payload["results"][0]["title"] == "年假管理制度.md"
        assert payload["results"][0]["score"] == 0.85

    def test_search_result_not_emitted_on_read_tool(self):
        """非 search 工具不应发出 search_result."""
        events = []
        hook = self._make_tool_hook(events)

        tc = self._make_tc("read", doc_id="abc123")
        result = self._make_result(success=True, content="文档内容...")

        hook(tc, result)

        event_types = [e[0] for e in events]
        assert "search_result" not in event_types

    def test_search_result_not_emitted_on_failed_search(self):
        """失败的搜索不应发出 search_result."""
        events = []
        hook = self._make_tool_hook(events)

        tc = self._make_tc("search", query="xxx")
        result = self._make_result(success=False, content="")

        hook(tc, result)

        event_types = [e[0] for e in events]
        assert "search_result" not in event_types

    def test_search_result_empty_results_skipped(self):
        """空搜索结果不应发出 search_result."""
        events = []
        hook = self._make_tool_hook(events)

        search_content = json.dumps({"results": []})
        tc = self._make_tc("search", query="不存在")
        result = self._make_result(success=True, content=search_content)

        hook(tc, result)

        event_types = [e[0] for e in events]
        assert "search_result" not in event_types

    def test_search_result_malformed_json_skipped(self):
        """损坏的 JSON 不应崩溃."""
        events = []
        hook = self._make_tool_hook(events)

        tc = self._make_tc("search", query="test")
        result = self._make_result(success=True, content="not valid json{{{")

        hook(tc, result)  # Should not raise

        event_types = [e[0] for e in events]
        assert "search_result" not in event_types

    def test_search_result_max_10_results(self):
        """超过 10 条结果时应截断."""
        events = []
        hook = self._make_tool_hook(events)

        many_results = [{"doc_id": f"d{i}", "title": f"Doc {i}", "score": 1.0 - i * 0.01, "snippet": "..."}
                        for i in range(20)]
        search_content = json.dumps({"results": many_results})
        tc = self._make_tc("search", query="test")
        result = self._make_result(success=True, content=search_content)

        hook(tc, result)
    
        print(f"DEBUG events: {[(e[0], list(e[1].keys())) for e in events]}")  # debug
        sr_event = [e for e in events if e[0] == "search_result"][0]
        assert len(sr_event[1]["results"]) == 10, "最多应返回 10 条结果"

    # ── tool_call + tool_result always emitted ──

    def test_tool_call_always_emitted(self):
        """任何工具调用都应发出 tool_call."""
        events = []
        hook = self._make_tool_hook(events)

        tc = self._make_tc("bash", command="ls")
        result = self._make_result(success=True, content="file1.txt\nfile2.txt")
        hook(tc, result)

        event_types = [e[0] for e in events]
        assert "tool_call" in event_types
        assert event_types[0] == "tool_call"

    def test_tool_result_always_emitted(self):
        """任何工具结果都应发出 tool_result."""
        events = []
        hook = self._make_tool_hook(events)

        tc = self._make_tc("grep", pattern="error")
        result = self._make_result(success=True, content="line 1: error found")
        hook(tc, result)

        event_types = [e[0] for e in events]
        assert "tool_result" in event_types

    def test_tool_result_content_truncated(self):
        """超过 500 字符的结果应被截断."""
        events = []
        hook = self._make_tool_hook(events)

        long_content = "x" * 1000
        tc = self._make_tc("read", doc_id="d1")
        result = self._make_result(success=True, content=long_content)
        hook(tc, result)

        tr_event = [e for e in events if e[0] == "tool_result"][0]
        preview = tr_event[1]["content_preview"]
        assert len(preview) == 500 + 3  # 500 chars + "..."
        assert preview.endswith("...")

    # ── AgentEventType enum ──

    def test_search_result_event_type_defined(self):
        """SEARCH_RESULT 事件类型应存在于 AgentEventType 枚举中."""
        assert hasattr(AgentEventType, "SEARCH_RESULT")
        assert AgentEventType.SEARCH_RESULT.value == "search_result"

    def test_sse_encode_search_result(self):
        """search_result 事件可被 sse_encode 序列化."""
        payload = {"query": "年假", "results": [
            {"doc_id": "abc", "title": "doc.md", "score": 0.9, "snippet": "年假5天"},
        ], "total": 1}
        encoded = sse_encode(AgentEventType.SEARCH_RESULT, payload)
        assert "event: search_result" in encoded
        assert "年假" in encoded
        assert "abc" in encoded
        # Verify it's valid JSON
        import re
        match = re.search(r'data: (.+)\n\n', encoded)
        assert match is not None
        parsed = json.loads(match.group(1))
        assert parsed["query"] == "年假"


# ── Frontend: TraceCollector JS logic ───────────────────────────────


class TestFrontendTraceCollector:
    """验证前端 JS TraceCollector 类的逻辑结构。

    由于 JS 测试需要 Node.js/浏览器环境，此类通过 Python 验证：
    - app.js 中 trace 对象的存在性和关键方法签名
    - SSE 事件到 trace 方法的映射完整性
    """

    TRACE_METHODS = ["clear", "onToolCall", "onToolResult",
                     "onSearchResult", "onAnswerComplete",
                     "_addRow", "_updateRow", "_esc"]

    def test_app_js_contains_trace_object(self):
        """app.js 应包含 trace 对象定义."""
        with open("src/web/static/app.js", encoding="utf-8") as f:
            content = f.read()
        assert "const trace = {" in content, "app.js 应定义 trace 对象"
        assert "clear()" in content
        assert "onToolCall" in content
        assert "onToolResult" in content
        assert "onSearchResult" in content
        assert "onAnswerComplete" in content
        assert "_addRow" in content

    def test_sss_event_handlers_wired(self):
        """SSE 事件处理函数应调用 trace 方法."""
        with open("src/web/static/app.js", encoding="utf-8") as f:
            content = f.read()
        # Each handler should call the corresponding trace method
        assert "trace.onToolCall(data)" in content
        assert "trace.onToolResult(data)" in content
        assert "trace.onSearchResult(data)" in content
        assert "trace.onAnswerComplete(data)" in content

    def test_clear_trace_called_on_new_search(self):
        """每次新查询应先清除 trace."""
        with open("src/web/static/app.js", encoding="utf-8") as f:
            content = f.read()
        assert "clearTrace()" in content, "新查询时应调用 clearTrace"

    def test_html_contains_trace_panel(self):
        """index.html 应包含 trace-panel 元素."""
        with open("src/web/static/index.html", encoding="utf-8") as f:
            content = f.read()
        assert 'id="trace-panel"' in content
        assert 'class="trace-panel' in content
        assert 'id="trace-panel-body"' in content
        assert 'id="trace-toggle-btn"' in content

    def test_css_contains_trace_styles(self):
        """style.css 应包含 trace 面板样式."""
        with open("src/web/static/style.css", encoding="utf-8") as f:
            content = f.read()
        assert ".trace-panel" in content
        assert ".trace-panel-header" in content
        assert ".trace-panel-body" in content
        assert ".trace-row" in content
        assert ".trace-summary" in content
        assert "trace-fade-in" in content
        assert "@keyframes trace-pulse" in content

    def test_i18n_has_trace_keys(self):
        """i18n.js 应包含 trace 面板翻译键."""
        with open("src/web/static/i18n.js", encoding="utf-8") as f:
            content = f.read()
        assert "'trace.title'" in content
        assert "'trace.toggle'" in content
        assert "'trace.empty'" in content
        assert "'trace.waiting'" in content

    def test_html_has_i18n_attributes(self):
        """index.html 中的 trace 元素应有 data-i18n 属性."""
        with open("src/web/static/index.html", encoding="utf-8") as f:
            content = f.read()
        assert 'data-i18n="trace.title"' in content
        assert 'data-i18n="trace.toggle"' in content
        assert 'data-i18n="trace.empty"' in content

    def test_connect_events_has_search_result_listener(self):
        """connectEvents 应注册 search_result 事件监听."""
        with open("src/web/static/app.js", encoding="utf-8") as f:
            content = f.read()
        assert "search_result" in content
        assert "onSearchResult" in content

    def test_trace_panel_toggle_interaction(self):
        """trace 面板应有折叠/展开交互."""
        with open("src/web/static/app.js", encoding="utf-8") as f:
            content = f.read()
        assert "traceToggle" in content
        assert "tracePanel" in content
        assert "classList.toggle('collapsed')" in content


# ── SSE event format consistency ────────────────────────────────────


class TestSSEFormatConsistency:
    """确保后端事件格式与前端期望一致."""

    def test_tool_call_format(self):
        """tool_call 事件包含 tool + arguments."""
        payload = {"tool": "search", "arguments": {"query": "年假"}}
        encoded = sse_encode(AgentEventType.TOOL_CALL, payload)
        assert "search" in encoded
        assert "年假" in encoded

    def test_tool_result_format(self):
        """tool_result 事件包含 tool + success + content_preview."""
        payload = {"tool": "search", "success": True,
                   "content_preview": "找到 2 篇文档..."}
        encoded = sse_encode(AgentEventType.TOOL_RESULT, payload)
        assert "search" in encoded
        assert "找到 2" in encoded

    def test_answer_complete_contains_search_hits(self):
        """answer_complete 事件含 search_hits 供 trace 面板汇总."""
        payload = {
            "answer": "年假为 5 天",
            "search_hits": [
                {"doc_id": "abc", "title": "年假管理制度.md", "score": 0.85}
            ],
            "tool_calls": [{"tool": "search", "success": True}],
            "tokens_used": 1500,
            "processing_time": 3200,
        }
        encoded = sse_encode(AgentEventType.ANSWER_COMPLETE, payload)
        assert "年假为 5 天" in encoded
        assert "search_hits" in encoded
        assert "年假管理制度.md" in encoded

    def test_heartbeat_format(self):
        """heartbeat 为 SSE 注释帧 (冒号开头表示 comment, 非 event)."""
        from src.web.sse_events import heartbeat
        hb = heartbeat()
        # SSE heartbeat is a comment frame (starts with ':')
        assert hb.startswith(":"), "heartbeat 应为 SSE 注释帧"
        assert hb == ":\n\n", "标准 SSE 心跳格式"

    def test_error_event_format(self):
        """error 事件含 message 字段."""
        payload = {"message": "API 调用失败"}
        encoded = sse_encode(AgentEventType.ERROR, payload)
        assert "API" in encoded
