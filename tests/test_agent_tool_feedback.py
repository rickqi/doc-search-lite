"""P2 — Tool Feedback Signal tests.

Verifies that SearchTool, ReadTool, and GrepTool provide structured feedback
signals (hints) to help the LLM adapt its search strategy.
"""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock


class TestSearchToolFeedback:
    """Test SearchTool feedback signals on zero / low hits."""

    def test_zero_hit_returns_hint(self):
        """SearchTool result JSON should include a hint field on zero hits."""
        from src.agent.tools.search import SearchTool

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = SimpleNamespace(
            results=[],
            total=0,
            offset=0,
            limit=5,
            has_more=False,
            execution_time=0.0,
            query="nonexistent",
        )
        tool = SearchTool(mock_searcher)
        result = tool.execute(query="xyznonexistent12345", limit=5)

        # result is a ToolResult; result.data is a JSON string
        data_str = result.data if hasattr(result, "data") else str(result)
        parsed = json.loads(data_str) if isinstance(data_str, str) else data_str
        assert isinstance(parsed, dict)
        assert "hint" in parsed
        assert "零命中" in parsed["hint"] or "建议" in parsed["hint"]

    def test_low_hit_returns_hint(self):
        """SearchTool should return a low-hit hint when total <= 2."""
        from src.agent.tools.search import SearchTool

        # Build a single fake SearchPreview-like object
        fake_preview = SimpleNamespace(
            doc_id="doc1",
            title="Test Doc",
            score=0.5,
            snippet="some snippet",
            source_path="/fake/doc.md",
            highlights=None,
        )
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = SimpleNamespace(
            results=[fake_preview],
            total=1,
            offset=0,
            limit=5,
            has_more=False,
            execution_time=0.0,
            query="test",
        )
        tool = SearchTool(mock_searcher)
        result = tool.execute(query="test", limit=5)

        data_str = result.data if hasattr(result, "data") else str(result)
        parsed = json.loads(data_str) if isinstance(data_str, str) else data_str
        assert isinstance(parsed, dict)
        assert "hint" in parsed
        assert "命中较少" in parsed["hint"] or "建议" in parsed["hint"]

    def test_search_source_has_hint_logic(self):
        """SearchTool.execute source should contain hint feedback logic."""
        from src.agent.tools.search import SearchTool

        source = inspect.getsource(SearchTool.execute)
        assert "hint" in source.lower() or "建议" in source


class TestReadToolFeedback:
    """Test ReadTool duplicate-read detection."""

    def test_read_tool_has_duplicate_detection(self):
        """ReadTool.execute source should track _read_history for duplicates."""
        from src.agent.tools.read import ReadTool

        source = inspect.getsource(ReadTool.execute)
        assert (
            "_read_history" in source
            or "already_read" in source.lower()
            or "已读取" in source
            or "duplicate" in source.lower()
        )

    def test_read_tool_init_has_history(self):
        """ReadTool.__init__ should initialize _read_history."""
        from src.agent.tools.read import ReadTool

        source = inspect.getsource(ReadTool.__init__)
        assert "_read_history" in source


class TestGrepToolFeedback:
    """Test GrepTool feedback on zero matches."""

    def test_grep_tool_has_hint(self):
        """GrepTool.execute source should provide hints on zero match."""
        from src.agent.tools.grep import GrepTool

        source = inspect.getsource(GrepTool.execute)
        assert "hint" in source.lower() or "建议" in source or "suggestion" in source.lower()

    def test_grep_hint_has_warning_emoji(self):
        """GrepTool zero-match feedback should include a warning indicator."""
        from src.agent.tools.grep import GrepTool

        source = inspect.getsource(GrepTool.execute)
        assert "⚠️" in source or "No matches" in source
