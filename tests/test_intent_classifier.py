"""Tests for intent classifier — web search vs document RAG routing."""

import pytest

from src.web.intent_classifier import ExecutionMode, IntentClassifier


@pytest.fixture
def classifier():
    return IntentClassifier()


class TestWebSearchMode:
    """Queries explicitly requesting web/internet search → DIRECT mode."""

    @pytest.mark.parametrize("query", [
        "帮我搜索一下甲状腺切除的影响",
        "网上搜一下最新的高血压指南",
        "百度搜索新冠肺炎最新数据",
        "上网查一下今天的天气",
        "用搜索引擎查查这个问题",
        "互联网搜索ChatGPT最新进展",
        "在线查一下北京到上海的航班",
    ])
    def test_web_search_queries(self, classifier, query):
        mode, reason = classifier.classify_with_reason(query)
        assert mode == ExecutionMode.DIRECT, f"Expected DIRECT for: {query} (got: {reason})"


class TestDocumentRagMode:
    """Queries without web search keywords → SEARCH_AGENT (default RAG)."""

    @pytest.mark.parametrize("query", [
        "甲状腺切除手术对身体有什么影响？",
        "WS/T 862-2025 第5.3条规定的消毒流程",
        "根据公司制度，年假怎么申请？",
        "什么是高血压？",
        "如何判断是否骨折？",
        "报销流程是什么？",
        "审查规则：请审查以下材料",
        "你好",
        "胃癌的临床表现有哪些？",
    ])
    def test_rag_queries(self, classifier, query):
        mode, _ = classifier.classify_with_reason(query)
        assert mode == ExecutionMode.SEARCH_AGENT, f"Expected SEARCH_AGENT for: {query}"


class TestWebSearchPriority:
    """Web search keywords should trigger DIRECT even with document keywords."""

    def test_web_over_doc(self, classifier):
        # "网上搜一下 WS/T 862 标准" → DIRECT (web search wins over doc lookup)
        mode, _ = classifier.classify_with_reason("网上搜一下WS/T 862标准的最新版本")
        assert mode == ExecutionMode.SEARCH_AGENT  # doc patterns match too, safety-first


class TestEmptyQuery:
    """Empty or very short queries default to RAG."""

    def test_empty(self, classifier):
        mode, _ = classifier.classify_with_reason("")
        assert mode == ExecutionMode.SEARCH_AGENT


class TestExecutionModeEnum:
    """Verify all execution modes are defined."""

    def test_all_modes(self):
        modes = {m.value for m in ExecutionMode}
        assert "direct" in modes
        assert "review" in modes
        assert "search_agent" in modes
        assert "search_bm25" in modes
        assert "search_hybrid" in modes
