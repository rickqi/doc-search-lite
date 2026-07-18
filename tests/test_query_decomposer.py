"""Tests for QueryDecomposer — query decomposition into sub-tasks."""

import json
from unittest.mock import MagicMock

from src.agent.llm_client import ChatResponse
from src.agent.query_decomposer import (
    DecompositionResult,
    QueryDecomposer,
    SubQuery,
)

# ── SubQuery tests ──


class TestSubQuery:
    """Test SubQuery dataclass."""

    def test_defaults(self):
        sq = SubQuery(query="test")
        assert sq.aspect == ""
        assert sq.keywords == []

    def test_full(self):
        sq = SubQuery(query="年假申请", aspect="年假流程", keywords=["年假", "申请"])
        assert sq.aspect == "年假流程"
        assert len(sq.keywords) == 2


# ── DecompositionResult tests ──


class TestDecompositionResult:
    """Test DecompositionResult dataclass."""

    def test_no_decomposition(self):
        result = DecompositionResult(needs_decomposition=False, original_query="test")
        assert not result.needs_decomposition
        assert result.sub_queries == []

    def test_with_sub_queries(self):
        result = DecompositionResult(
            needs_decomposition=True,
            sub_queries=[
                SubQuery(query="报销标准", aspect="标准"),
                SubQuery(query="审批流程", aspect="流程"),
            ],
            original_query="报销标准和审批流程",
        )
        assert len(result.sub_queries) == 2


# ── QueryDecomposer tests ──


class TestQueryDecomposer:
    """Test QueryDecomposer.decompose() method."""

    def _make_decomposer(self, response_content: str) -> QueryDecomposer:
        llm = MagicMock()
        llm.chat.return_value = ChatResponse(
            content=response_content,
            usage={"total_tokens": 80},
        )
        return QueryDecomposer(llm)

    def test_short_query_no_decomposition(self):
        """Very short query skips decomposition entirely (no LLM call)."""
        llm = MagicMock()
        decomposer = QueryDecomposer(llm)
        result = decomposer.decompose("年假")
        assert not result.needs_decomposition
        llm.chat.assert_not_called()

    def test_single_aspect_no_decomposition(self):
        """Query with no compound signals skips decomposition."""
        llm = MagicMock()
        decomposer = QueryDecomposer(llm)
        result = decomposer.decompose("如何申请年假")
        assert not result.needs_decomposition
        llm.chat.assert_not_called()

    def test_compound_query_triggers_llm(self):
        """Query with compound signals triggers LLM decomposition."""
        response_json = json.dumps({
            "needs_decomposition": True,
            "sub_queries": [
                {"query": "差旅报销标准", "aspect": "报销标准", "keywords": ["差旅", "标准"]},
                {"query": "差旅审批流程", "aspect": "审批流程", "keywords": ["差旅", "审批"]},
            ],
            "cross_reference": False,
        })
        decomposer = self._make_decomposer(response_json)
        result = decomposer.decompose("差旅报销标准和审批流程的区别")
        assert result.needs_decomposition
        assert len(result.sub_queries) == 2
        assert result.sub_queries[0].query == "差旅报销标准"

    def test_llm_says_no_decomposition(self):
        """LLM determines query is single-aspect."""
        response_json = json.dumps({
            "needs_decomposition": False,
            "sub_queries": [],
            "cross_reference": False,
        })
        decomposer = self._make_decomposer(response_json)
        result = decomposer.decompose("差旅标准和审批流程的关系")
        assert not result.needs_decomposition

    def test_llm_failure_defaults_no_decomposition(self):
        """LLM failure → no decomposition (fail-safe)."""
        llm = MagicMock()
        llm.chat.side_effect = Exception("API error")
        decomposer = QueryDecomposer(llm)
        result = decomposer.decompose("差旅报销标准和审批流程的区别")
        assert not result.needs_decomposition

    def test_malformed_json_defaults_no_decomposition(self):
        """Malformed response → no decomposition."""
        decomposer = self._make_decomposer("not json")
        result = decomposer.decompose("差旅标准和审批流程")
        assert not result.needs_decomposition

    def test_only_one_subquery_means_no_decomposition(self):
        """Single sub-query is not meaningful decomposition."""
        response_json = json.dumps({
            "needs_decomposition": True,
            "sub_queries": [
                {"query": "差旅标准", "aspect": "标准", "keywords": ["差旅"]},
            ],
            "cross_reference": False,
        })
        decomposer = self._make_decomposer(response_json)
        result = decomposer.decompose("差旅标准和审批流程")
        assert not result.needs_decomposition  # Only 1 sub-query → treat as no decomposition

    def test_json_in_markdown_code_block(self):
        """JSON in markdown code block is parsed correctly."""
        response = (
            '```json\n{"needs_decomposition": true, '
            '"sub_queries": [{"query": "A", "aspect": "a", "keywords": []}, '
            '{"query": "B", "aspect": "b", "keywords": []}], '
            '"cross_reference": false}\n```'
        )
        decomposer = self._make_decomposer(response)
        result = decomposer.decompose("A和B的对比")
        assert result.needs_decomposition
        assert len(result.sub_queries) == 2

    def test_empty_llm_response(self):
        """Empty response → no decomposition."""
        decomposer = self._make_decomposer("")
        result = decomposer.decompose("差旅标准和审批流程")
        assert not result.needs_decomposition

    def test_comparison_query_triggers_decomposition(self):
        """Comparison-style query triggers LLM analysis."""
        response_json = json.dumps({
            "needs_decomposition": True,
            "sub_queries": [
                {"query": "A公司福利", "aspect": "A公司", "keywords": ["A公司"]},
                {"query": "B公司福利", "aspect": "B公司", "keywords": ["B公司"]},
            ],
            "cross_reference": True,
        })
        decomposer = self._make_decomposer(response_json)
        result = decomposer.decompose("A公司和B公司的福利对比分析")
        assert result.needs_decomposition
        assert result.cross_reference is True

    def test_original_query_preserved(self):
        """Original query is always preserved in result."""
        llm = MagicMock()
        decomposer = QueryDecomposer(llm)
        result = decomposer.decompose("短查询")
        assert result.original_query == "短查询"


# ── Pydantic robustness tests ──


class TestQueryDecomposerPydanticRobustness:
    """Test Pydantic-based validation robustness for edge cases."""

    def _make_decomposer(self, response_content: str) -> QueryDecomposer:
        llm = MagicMock()
        llm.chat.return_value = ChatResponse(
            content=response_content,
            usage={"total_tokens": 80},
        )
        return QueryDecomposer(llm)

    def test_extra_json_fields_ignored(self):
        """Unknown fields in LLM JSON response should be ignored, not error."""
        response_json = json.dumps({
            "needs_decomposition": True,
            "sub_queries": [
                {"query": "A", "aspect": "a", "keywords": [], "bogus": "x"},
                {"query": "B", "aspect": "b", "keywords": [], "junk": 42},
            ],
            "cross_reference": False,
            "unknown_field": "should be ignored",
            "extra_meta": {"deep": "nested"},
        })
        decomposer = self._make_decomposer(response_json)
        result = decomposer.decompose("A和B的对比")
        assert result.needs_decomposition
        assert len(result.sub_queries) == 2

    def test_missing_fields_use_defaults(self):
        """Missing optional fields should use Pydantic defaults, not error."""
        response_json = json.dumps({
            "needs_decomposition": True,
            "sub_queries": [
                {"query": "A"},
                {"query": "B"},
            ],
        })
        decomposer = self._make_decomposer(response_json)
        result = decomposer.decompose("A和B的对比")
        assert result.needs_decomposition
        assert len(result.sub_queries) == 2
        assert result.sub_queries[0].aspect == ""
        assert result.sub_queries[0].keywords == []

    def test_completely_invalid_json_returns_no_decomposition(self):
        """Completely invalid JSON still returns fail-safe default."""
        decomposer = self._make_decomposer("{{{not json at all")
        result = decomposer.decompose("A和B的对比")
        assert not result.needs_decomposition
        assert result.original_query == "A和B的对比"

    def test_subquery_with_empty_query_filtered_out(self):
        """Sub-queries with blank query strings are filtered out."""
        response_json = json.dumps({
            "needs_decomposition": True,
            "sub_queries": [
                {"query": "A", "aspect": "a"},
                {"query": "   ", "aspect": "blank"},
            ],
            "cross_reference": False,
        })
        decomposer = self._make_decomposer(response_json)
        result = decomposer.decompose("A和B的对比")
        assert not result.needs_decomposition
