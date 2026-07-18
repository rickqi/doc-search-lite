"""Tests for SufficientContextChecker and SearchFeedback."""

import json
from unittest.mock import MagicMock

from src.agent.llm_client import ChatResponse
from src.agent.sufficient_context import (
    SearchFeedback,
    SufficientContextChecker,
)

# ── SearchFeedback tests ──


class TestSearchFeedback:
    """Test SearchFeedback dataclass."""

    def test_defaults(self):
        fb = SearchFeedback(sufficient=True, coverage_score=0.8)
        assert fb.covered_aspects == []
        assert fb.missing_aspects == []
        assert fb.suggested_queries == []
        assert fb.suggested_tools == []
        assert fb.reason == ""

    def test_full_construction(self):
        fb = SearchFeedback(
            sufficient=False,
            coverage_score=0.3,
            covered_aspects=["报销标准"],
            missing_aspects=["审批流程"],
            suggested_queries=["审批流程 差旅"],
            suggested_tools=["search"],
            reason="缺少审批流程信息",
        )
        assert not fb.sufficient
        assert fb.coverage_score == 0.3
        assert len(fb.missing_aspects) == 1


# ── SufficientContextChecker tests ──


class TestSufficientContextChecker:
    """Test SufficientContextChecker.check() method."""

    def _make_checker(self, response_content: str) -> SufficientContextChecker:
        llm = MagicMock()
        llm.chat.return_value = ChatResponse(
            content=response_content,
            usage={"total_tokens": 100},
        )
        return SufficientContextChecker(llm)

    def test_no_snippets_returns_insufficient(self):
        """Empty snippets → insufficient with suggestion to search."""
        llm = MagicMock()
        checker = SufficientContextChecker(llm)
        result = checker.check("年假如何申请", [])
        assert not result.sufficient
        assert result.coverage_score == 0.0
        assert len(result.suggested_queries) > 0
        # LLM should NOT be called for empty snippets
        llm.chat.assert_not_called()

    def test_sufficient_response(self):
        """LLM says sufficient → SearchFeedback.sufficient=True."""
        response_json = json.dumps({
            "sufficient": True,
            "coverage_score": 0.9,
            "covered_aspects": ["年假天数", "申请流程"],
            "missing_aspects": [],
            "suggested_queries": [],
            "suggested_tools": [],
            "reason": "信息完整",
        })
        checker = self._make_checker(response_json)
        result = checker.check(
            "年假如何申请",
            [{"title": "年假制度", "snippet": "年假天数和申请流程...", "source": "hr/年假.md"}],
        )
        assert result.sufficient
        assert result.coverage_score >= 0.9

    def test_insufficient_response(self):
        """LLM says insufficient → returns missing aspects and suggestions."""
        response_json = json.dumps({
            "sufficient": False,
            "coverage_score": 0.4,
            "covered_aspects": ["差旅报销标准"],
            "missing_aspects": ["审批流程", "报销时限"],
            "suggested_queries": ["差旅审批流程", "差旅报销时限"],
            "suggested_tools": ["search"],
            "reason": "缺少审批流程和时限信息",
        })
        checker = self._make_checker(response_json)
        result = checker.check(
            "差旅报销标准和审批流程",
            [{"title": "差旅标准", "snippet": "差旅报销标准...", "source": "finance/差旅.md"}],
        )
        assert not result.sufficient
        assert "审批流程" in result.missing_aspects
        assert len(result.suggested_queries) > 0

    def test_llm_failure_defaults_sufficient(self):
        """LLM call failure → default to sufficient (don't block)."""
        llm = MagicMock()
        llm.chat.side_effect = Exception("API error")
        checker = SufficientContextChecker(llm)
        result = checker.check("测试", [{"title": "t", "snippet": "s", "source": "x"}])
        assert result.sufficient  # Fail-open
        assert "失败" in result.reason

    def test_malformed_json_defaults_sufficient(self):
        """Malformed LLM response → default to sufficient."""
        checker = self._make_checker("This is not JSON at all")
        result = checker.check("测试", [{"title": "t", "snippet": "s", "source": "x"}])
        assert result.sufficient  # Fail-open

    def test_json_in_markdown_code_block(self):
        """JSON wrapped in ```markdown``` is extracted correctly."""
        response = (
            '```json\n{"sufficient": false, "coverage_score": 0.3, '
            '"covered_aspects": [], "missing_aspects": ["all"], '
            '"suggested_queries": ["test"], "suggested_tools": ["search"], '
            '"reason": "test"}\n```'
        )
        checker = self._make_checker(response)
        result = checker.check("测试", [{"title": "t", "snippet": "s", "source": "x"}])
        assert not result.sufficient

    def test_snippet_truncation(self):
        """Long snippets are truncated to MAX_SNIPPET_LENGTH."""
        ok_json = json.dumps({
            "sufficient": True, "coverage_score": 0.5,
            "covered_aspects": [], "missing_aspects": [],
            "suggested_queries": [], "suggested_tools": [], "reason": "",
        })
        checker = self._make_checker(ok_json)
        long_snippet = "x" * 1000
        result = checker.check("test", [{"title": "t", "snippet": long_snippet, "source": "x"}])
        assert result.sufficient  # Just verify it doesn't crash

    def test_many_snippets_limited_to_10(self):
        """More than 10 snippets are truncated."""
        ok_json = json.dumps({
            "sufficient": True, "coverage_score": 0.5,
            "covered_aspects": [], "missing_aspects": [],
            "suggested_queries": [], "suggested_tools": [], "reason": "",
        })
        checker = self._make_checker(ok_json)
        snippets = [{"title": f"doc{i}", "snippet": f"content{i}", "source": f"s{i}"} for i in range(20)]
        result = checker.check("test", snippets)
        assert result.sufficient  # Just verify it doesn't crash

    def test_empty_llm_response(self):
        """Empty LLM response → default to sufficient."""
        checker = self._make_checker("")
        result = checker.check("test", [{"title": "t", "snippet": "s", "source": "x"}])
        assert result.sufficient

    def test_prompt_contains_query_and_snippets(self):
        """Verify the LLM prompt includes query and snippet content."""
        llm = MagicMock()
        llm.chat.return_value = ChatResponse(
            content=json.dumps({
                "sufficient": True, "coverage_score": 0.5,
                "covered_aspects": [], "missing_aspects": [],
                "suggested_queries": [], "suggested_tools": [], "reason": "",
            }),
            usage={"total_tokens": 100},
        )
        checker = SufficientContextChecker(llm)
        checker.check("年假", [{"title": "年假制度", "snippet": "年假5天", "source": "hr.md"}])

        # Verify the prompt sent to LLM contains our data
        call_args = llm.chat.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0].content if hasattr(messages[0], "content") else str(messages[0])
        assert "年假" in prompt_text


# ── Pydantic robustness tests ──


class TestSufficientContextPydanticRobustness:
    """Test Pydantic-based validation robustness for edge cases."""

    def _make_checker(self, response_content: str) -> SufficientContextChecker:
        llm = MagicMock()
        llm.chat.return_value = ChatResponse(
            content=response_content,
            usage={"total_tokens": 100},
        )
        return SufficientContextChecker(llm)

    def test_extra_json_fields_ignored(self):
        """Unknown fields in LLM JSON response should be ignored, not error."""
        response_json = json.dumps({
            "sufficient": False,
            "coverage_score": 0.3,
            "covered_aspects": [],
            "missing_aspects": ["all"],
            "suggested_queries": ["test"],
            "suggested_tools": ["search"],
            "reason": "test",
            "unknown_field": "should be ignored",
            "extra_meta": {"deep": "nested"},
        })
        checker = self._make_checker(response_json)
        result = checker.check("测试", [{"title": "t", "snippet": "s", "source": "x"}])
        assert not result.sufficient

    def test_missing_fields_use_defaults(self):
        """Missing optional fields should use Pydantic defaults, not error."""
        response_json = json.dumps({"sufficient": False, "coverage_score": 0.4})
        checker = self._make_checker(response_json)
        result = checker.check("测试", [{"title": "t", "snippet": "s", "source": "x"}])
        assert not result.sufficient
        assert result.covered_aspects == []
        assert result.suggested_tools == ["search"]
        assert result.reason == ""

    def test_completely_invalid_json_returns_fail_open(self):
        """Completely invalid JSON still returns fail-open sufficient default."""
        checker = self._make_checker("{{{totally not json")
        result = checker.check("测试", [{"title": "t", "snippet": "s", "source": "x"}])
        assert result.sufficient
        assert result.coverage_score == 0.5

    def test_empty_json_object_returns_defaults(self):
        """Empty JSON object {} should yield fail-open defaults."""
        checker = self._make_checker("{}")
        result = checker.check("测试", [{"title": "t", "snippet": "s", "source": "x"}])
        assert result.sufficient
        assert result.coverage_score == 0.5
        assert result.suggested_tools == ["search"]
