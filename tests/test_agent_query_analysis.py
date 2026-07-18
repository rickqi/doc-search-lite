"""P4 — Query Analysis Pre-stage tests.

Verifies that SearchAgent has a _analyze_query method and that the tool_loop
includes a query_analysis diagnostics step before the main loop. These are
structural/source-level assertions — no LLM calls required.
"""

import inspect

from src.agent.search_agent import SearchAgent


class TestQueryAnalysis:
    """Test P4 query analysis pre-stage."""

    def test_analyze_query_method_exists(self):
        """SearchAgent should have _analyze_query method."""
        assert hasattr(SearchAgent, "_analyze_query")

    def test_analyze_query_is_callable(self):
        """_analyze_query should be callable."""
        assert callable(getattr(SearchAgent, "_analyze_query", None))

    def test_query_analysis_step_in_tool_loop(self):
        """_run_tool_loop should reference query_analysis diagnostics step."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "query_analysis" in source or "_analyze_query" in source

    def test_query_analysis_uses_fast_tier(self):
        """_analyze_query should use model_tier='fast' for low latency."""
        source = inspect.getsource(SearchAgent._analyze_query)
        assert "model_tier" in source
        assert "fast" in source

    def test_analyze_query_returns_tuple(self):
        """_analyze_query should return a tuple of (action, search_query)."""
        source = inspect.getsource(SearchAgent._analyze_query)
        assert "return" in source

    def test_analyze_query_has_greeting_shortcut(self):
        """_analyze_query should short-circuit simple greetings to 'direct'."""
        source = inspect.getsource(SearchAgent._analyze_query)
        assert "你好" in source or "hello" in source.lower()

    def test_analyze_query_has_rule_based_routing(self):
        """_analyze_query should have rule-based search indicator matching."""
        source = inspect.getsource(SearchAgent._analyze_query)
        assert "制度" in source or "search_indicators" in source

    def test_query_analysis_injects_system_message(self):
        """_run_tool_loop should inject analysis result as a system message."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "查询分析" in source or "query_analysis" in source
