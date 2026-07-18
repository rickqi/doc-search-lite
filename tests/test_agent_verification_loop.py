"""P1 — Draft Verification Recovery Loop tests.

Verifies that SearchAgent has the methods and recovery-loop logic needed for
P1 draft-verification recovery. These are structural/source-level assertions
— no LLM calls or instantiation required.
"""

import inspect

from src.agent.search_agent import SearchAgent


class TestVerificationLoop:
    """Test that SearchAgent supports P1 verification recovery."""

    def test_search_agent_has_recovery_methods(self):
        """SearchAgent must have _verify_draft_grounding and/or regeneration."""
        assert hasattr(SearchAgent, "_verify_draft_grounding") or hasattr(
            SearchAgent, "_regenerate_answer_from_docs"
        )

    def test_verify_draft_grounding_is_callable(self):
        """_verify_draft_grounding should be a method."""
        assert callable(getattr(SearchAgent, "_verify_draft_grounding", None))

    def test_regen_answer_from_docs_is_callable(self):
        """_regenerate_answer_from_docs should be a method."""
        assert callable(getattr(SearchAgent, "_regenerate_answer_from_docs", None))

    def test_recovery_flag_in_code(self):
        """_run_tool_loop source must reference verify_recovery or recovery logic."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "verify_recovery" in source or "verification_recovery" in source

    def test_verification_uses_suggested_queries(self):
        """Recovery loop should use suggested_queries from feedback."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        assert "suggested_queries" in source

    def test_recovery_capped_at_2(self):
        """Recovery supplementary searches should be capped at 2 queries."""
        source = inspect.getsource(SearchAgent._run_tool_loop)
        # The code slices suggested_queries[:2]
        assert "[:2]" in source or "[: 2]" in source
