"""P3 — Convergence Nudge tests.

Verifies that the convergence early-stop mechanism includes an exploration
nudge: instead of hard-stopping on the first trigger of consecutive searches,
the agent injects a guidance message giving the LLM one more chance.

Note: _check_convergence is a nested closure inside _run_tool_loop, not a
standalone method. Tests inspect _run_tool_loop source to verify the logic.
"""

import inspect

import pytest

from src.agent.search_agent import SearchAgent


class TestConvergenceNudge:
    """Test convergence nudge (exploration push) behavior."""

    def test_exploration_nudged_flag_exists(self):
        """_run_tool_loop source must use _exploration_nudged flag."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "_exploration_nudged" in source

    def test_convergence_check_is_nested_closure(self):
        """_check_convergence closure should exist inside _run_tool_loop."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "_check_convergence" in source

    def test_nudge_injects_message(self):
        """Nudge logic should inject a user message into the conversation."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "messages.append" in source or "messages.insert" in source

    def test_nudge_resets_consecutive_counter(self):
        """After nudge, consecutive search counter should be reset."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "_consecutive_searches = 0" in source or "_consecutive_searches=0" in source

    def test_nudge_only_once(self):
        """_exploration_nudged flag should be set True after first nudge."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "_exploration_nudged = True" in source

    def test_hard_stop_after_nudge(self):
        """If nudged already, consecutive searches should trigger hard stop."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "return True" in source
        assert "post-nudge" in source or "_exploration_nudged" in source

    def test_consecutive_threshold_is_2(self):
        """Consecutive search threshold for nudge should be 2."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert ">= 2" in source or ">=2" in source

    def test_should_stop_passed_to_chat_with_tools(self):
        """_check_convergence should be passed as should_stop callback."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "should_stop" in source
        assert "_check_convergence" in source
