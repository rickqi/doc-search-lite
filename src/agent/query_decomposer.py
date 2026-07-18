"""Query Decomposer — Split complex queries into independent sub-tasks.

Inspired by Google's Agentic RAG Planner Agent + Query Rewriter.
For compound queries (e.g. "A的标准和B的流程"), decomposes into
independent sub-queries that can be searched and answered separately.

Usage:
    decomposer = QueryDecomposer(llm_client)
    result = decomposer.decompose("差旅报销标准和审批流程的区别")
    if result.needs_decomposition:
        for sq in result.sub_queries:
            agent.search(sq.query)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.agent.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SubQueryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = ""
    aspect: str = ""
    keywords: list[str] = Field(default_factory=list)


class DecompositionResultModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    needs_decomposition: bool = False
    sub_queries: list[SubQueryModel] = Field(default_factory=list)
    cross_reference: bool = False


@dataclass
class SubQuery:
    """A single sub-task from query decomposition.

    Attributes:
        query: Search query for this sub-task.
        aspect: Description of what this sub-query covers.
        keywords: Extracted keywords for routing.
    """

    query: str
    aspect: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """Result of query decomposition.

    Attributes:
        needs_decomposition: Whether the query should be split.
        sub_queries: List of decomposed sub-queries (empty if no decomposition needed).
        cross_reference: Whether sub-queries need cross-referencing (e.g. using IDs).
        original_query: The original user query.
    """

    needs_decomposition: bool
    sub_queries: list[SubQuery] = field(default_factory=list)
    cross_reference: bool = False
    original_query: str = ""


# Prompt for decomposition (lightweight, one-shot)
DECOMPOSITION_PROMPT = """分析用户查询，判断是否需要分解为多个独立搜索子任务。

查询: {query}

分解规则:
1. 如果查询包含多个独立问题（如"A的X和B的Y"、"X和Y的区别"），请拆分
2. 如果是对比/分析型查询（"X与Y的差异"、"X和Y分别是什么"），请拆分
3. 如果是单一问题，不需要分解
4. 每个子任务应该是独立可搜索的

输出严格JSON格式（不要markdown包裹）:
{{
  "needs_decomposition": true或false,
  "sub_queries": [
    {{"query": "子查询1", "aspect": "描述1", "keywords": ["关键词1", "关键词2"]}},
    {{"query": "子查询2", "aspect": "描述2", "keywords": ["关键词3"]}}
  ],
  "cross_reference": false
}}"""


class QueryDecomposer:
    """Decomposes complex queries into independent sub-tasks.

    Uses LLM to analyze query structure and split compound questions
    into individually searchable sub-queries.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._last_usage: dict[str, int] = {}

    def decompose(self, query: str) -> DecompositionResult:
        """Decompose a query into sub-tasks if needed.

        Args:
            query: User query string.

        Returns:
            DecompositionResult with sub_queries if decomposition is needed.
        """
        # Quick rule-based check: single-aspect queries skip LLM
        if not self._might_need_decomposition(query):
            return DecompositionResult(
                needs_decomposition=False,
                original_query=query,
            )

        prompt = DECOMPOSITION_PROMPT.format(query=query)

        try:
            from src.agent.llm_client import ChatMessage
            response = self._llm.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1,
                max_tokens=400,
                model_tier="fast",
            )
            self._last_usage = response.usage if response.usage else {}
            return self._parse_response(response.content, query)
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return DecompositionResult(
                needs_decomposition=False,
                original_query=query,
            )

    def _might_need_decomposition(self, query: str) -> bool:
        """Quick rule-based check if query might need decomposition.

        Avoids LLM call for obvious single-aspect queries.
        """
        # Compound signals that suggest multiple aspects
        compound_signals = [
            "和", "与", "以及", "同时", "并且",
            "对比", "比较", "区别", "差异", "分别",
            "哪些", "各",
        ]
        signal_count = sum(1 for s in compound_signals if s in query)

        # Need at least 2 compound signals OR one signal + long query
        if signal_count >= 2:
            return True
        if signal_count >= 1 and len(query) > 15:
            return True
        return False

    def _parse_response(self, content: str, original_query: str) -> DecompositionResult:
        """Parse LLM response into DecompositionResult."""
        if not content:
            return DecompositionResult(
                needs_decomposition=False,
                original_query=original_query,
            )

        text = content.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        try:
            raw = json.loads(text)
            result = DecompositionResultModel.model_validate(raw)
        except Exception as e:
            logger.warning(f"Failed to parse decomposition response: {e}")
            return DecompositionResult(
                needs_decomposition=False,
                original_query=original_query,
            )

        if not result.needs_decomposition:
            return DecompositionResult(
                needs_decomposition=False,
                original_query=original_query,
            )

        sub_queries: list[SubQuery] = [
            SubQuery(query=sq.query, aspect=sq.aspect, keywords=sq.keywords)
            for sq in result.sub_queries
            if sq.query.strip()
        ]

        # Validate: must have at least 2 sub-queries to be meaningful
        if len(sub_queries) < 2:
            return DecompositionResult(
                needs_decomposition=False,
                original_query=original_query,
            )

        return DecompositionResult(
            needs_decomposition=True,
            sub_queries=sub_queries,
            cross_reference=result.cross_reference,
            original_query=original_query,
        )
