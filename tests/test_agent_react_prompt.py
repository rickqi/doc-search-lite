"""P0 — ReAct Prompt tests.

Verifies that SYSTEM_PROMPT contains reasoning guidance (ReAct-style thought
process) that instructs the LLM to articulate 已知/缺失/计划 before each tool
call. These are static assertions against the prompt string — no LLM or
network calls are needed.
"""


from src.agent.search_agent import SYSTEM_PROMPT


class TestReActPrompt:
    """Test that the system prompt guides ReAct-style reasoning."""

    def test_prompt_contains_reasoning_keywords(self):
        """SYSTEM_PROMPT includes 已知/推理 and 缺失/计划 keywords."""
        assert "已知" in SYSTEM_PROMPT or "推理" in SYSTEM_PROMPT
        assert "缺失" in SYSTEM_PROMPT or "计划" in SYSTEM_PROMPT

    def test_prompt_contains_action_guidance(self):
        """SYSTEM_PROMPT references tools and search/read actions."""
        assert "工具" in SYSTEM_PROMPT
        assert "search" in SYSTEM_PROMPT.lower() or "搜索" in SYSTEM_PROMPT

    def test_prompt_not_empty(self):
        """SYSTEM_PROMPT should be substantial (> 200 chars)."""
        assert len(SYSTEM_PROMPT) > 200

    def test_prompt_contains_read_requirement(self):
        """SYSTEM_PROMPT must enforce reading documents (not snippet-only)."""
        assert "read" in SYSTEM_PROMPT.lower() or "读取" in SYSTEM_PROMPT

    def test_prompt_mentions_prohibited_behaviors(self):
        """SYSTEM_PROMPT lists forbidden behaviors (❌)."""
        assert "禁止" in SYSTEM_PROMPT or "❌" in SYSTEM_PROMPT
