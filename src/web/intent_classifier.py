"""Intent classifier — routes queries to direct LLM or RAG pipeline.

Usage:
    >>> from src.web.intent_classifier import IntentClassifier, ExecutionMode
    >>> c = IntentClassifier()
    >>> c.classify("甲状腺切除有什么影响？")
    ExecutionMode.DIRECT
    >>> c.classify("WS/T 862-2025 第5.3条")
    ExecutionMode.SEARCH_AGENT
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class ExecutionMode(str, Enum):
    """Execution mode returned in every response for debugging and routing."""
    DIRECT = "direct"
    REVIEW = "review"
    SEARCH_AGENT = "search_agent"
    SEARCH_BM25 = "search_bm25"
    SEARCH_HYBRID = "search_hybrid"


# Patterns that indicate the query needs internet/web search
WEB_SEARCH_PATTERNS = [
    r"(搜索|搜一下|搜一搜|帮我搜|查一下|查一查).{0,20}",
    r"(网上|互联网|在线|网络|上网).{0,5}(搜索|查询|查找|搜|查)",
    r"(百度|Google|谷歌|Bing|必应).{0,5}(搜索|一下|搜|查)",
    r"(上网|在线|去网上).{0,10}(查|搜|找|看看)",
    r"(用|通过|借助).{0,5}(搜索引擎|互联网|网络).{0,10}(搜|查|找)",
]

# Patterns that indicate the query needs local document retrieval
DOCUMENT_PATTERNS = [
    # Standard/document references
    r"(WS/T|WS |GB/T|GB |YY/T)\s*\d+",
    r"第[一二三四五六七八九十\d]+[章节条]",
    r"(根据|按照|依据).{0,10}(规定|标准|制度|指南|规范)",
    # Specific document queries
    r"(制度|合同|协议|条款|规定).{0,5}(怎么|如何|什么)",
    r"(年假|报销|出差|加班|请假|调休|转正|离职).{0,10}(怎么|如何|多少|几天)",
    r"(流程|步骤|手续).{0,5}(是|怎么|如何)",
    # Compliance/review
    r"(审查规则|待审查材料|审查材料|角色设定|审查专家)",
    # File/content lookup
    r"(查找|搜索|检索|查询|找一下|帮我找).{0,10}(文档|文件|资料|内容)",
    r"(最新|最近).{0,5}(版本|更新|修订)",
    # Explicit search intent
    r"(在文档|从文档|根据文档|查文档|搜文档)",
]

# System prompt for web search (direct) mode — instructs LLM to use real-time internet search,
# NOT pre-trained knowledge from model parameters.
WEB_SEARCH_SYSTEM_PROMPT = (
    "你是一个联网搜索助手。重要：不要依赖你的预训练知识（模型训练时从海量互联网文本中学到的、"
    "存储在模型参数里的静态知识），必须使用实时互联网搜索来获取最新信息。\n\n"
    "要求：\n"
    "1. 对每个问题执行互联网搜索，基于搜索结果回答\n"
    "2. 标注每条信息的来源（网址或来源名称）\n"
    "3. 如果搜索结果不充分，明确说明'搜索结果不足'，不要编造\n"
    "4. 优先使用最新、最权威的搜索结果"
)


class IntentClassifier:
    """Hybrid intent classifier: keyword rules with optional LLM fallback.

    Routes queries to:
    - 'direct': internet/web search requested by user
    - 'search_agent': local document RAG pipeline (default)
    """

    def __init__(self):
        self._web_re = [re.compile(p, re.IGNORECASE) for p in WEB_SEARCH_PATTERNS]
        self._doc_re = [re.compile(p, re.IGNORECASE) for p in DOCUMENT_PATTERNS]

    def classify(self, query: str) -> ExecutionMode:
        """Classify query intent.

        Returns:
            ExecutionMode.DIRECT for web search queries,
            ExecutionMode.SEARCH_AGENT for everything else (default).
        """
        web_score = sum(1 for p in self._web_re if p.search(query))
        doc_score = sum(1 for p in self._doc_re if p.search(query))

        # Only route to direct if user explicitly asks for web/internet search
        if web_score > 0 and doc_score == 0:
            return ExecutionMode.DIRECT

        # Everything else goes to document RAG
        return ExecutionMode.SEARCH_AGENT

    def classify_with_reason(self, query: str) -> tuple[ExecutionMode, str]:
        """Classify and return reason for debugging."""
        mode = self.classify(query)

        web_matches = [p.pattern for p in self._web_re if p.search(query)]
        doc_matches = [p.pattern for p in self._doc_re if p.search(query)]

        if mode == ExecutionMode.DIRECT:
            reason = f"web_search: {web_matches[0][:60]}"
        else:
            if doc_matches:
                reason = f"document_lookup: {doc_matches[0][:60]}"
            else:
                reason = "default_rag (no web search keywords detected)"

        return mode, reason


# Singleton
_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """Get or create the global intent classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
