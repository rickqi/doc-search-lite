"""Sufficient Context Checker — Evaluate whether collected information is sufficient.

Inspired by Google's Agentic RAG Sufficient Context Agent.
Checks: 1) Retrieved snippets coverage  2) Intermediate draft review  3) Gap analysis

Usage:
    checker = SufficientContextChecker(llm_client)
    feedback = checker.check(
        query="差旅报销标准和审批流程",
        collected_snippets=[{"title": "...", "snippet": "...", "source": "..."}],
    )
    if not feedback.sufficient:
        # Inject feedback and continue searching
        for suggestion in feedback.suggested_queries:
            agent.search(suggestion)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from src.agent.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SearchFeedbackModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sufficient: bool = True
    coverage_score: float = 0.5
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=lambda: ["search"])
    reason: str = ""


@dataclass
class SearchFeedback:
    """Structured feedback from sufficient context analysis.

    Attributes:
        sufficient: Whether collected information is sufficient to answer the query.
        coverage_score: Coverage score 0.0-1.0.
        covered_aspects: List of query aspects that are covered.
        missing_aspects: List of query aspects that are NOT covered.
        suggested_queries: Specific search queries to fill gaps.
        suggested_tools: Recommended tools ("search", "grep", "read").
        reason: Human-readable explanation.
    """

    sufficient: bool
    coverage_score: float
    covered_aspects: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)
    suggested_tools: List[str] = field(default_factory=list)
    reason: str = ""


# Prompt for sufficiency check (use lightweight model to save cost)
SUFFICIENCY_CHECK_PROMPT = """你是信息充足性评估专家。

原始查询: {query}

已收集的文档片段:
{snippets}

请评估已收集的信息是否足以完整回答原始查询。

评估标准:
1. 查询的每个方面是否都有对应文档支持
2. 文档片段是否包含足够细节（不是仅有标题或简单提及）
3. 是否有明显缺失的关键信息

输出严格JSON格式（不要markdown包裹）:
{{
  "sufficient": true或false,
  "coverage_score": 0.0到1.0的浮点数,
  "covered_aspects": ["已覆盖的方面1", "已覆盖的方面2"],
  "missing_aspects": ["缺失的方面1"],
  "suggested_queries": ["建议搜索查询1", "建议搜索查询2"],
  "suggested_tools": ["search"],
  "reason": "简短说明为什么充足或不足"
}}"""


class SufficientContextChecker:
    """Checks whether collected information is sufficient to answer a query.

    Uses an LLM call to evaluate coverage of the original query.
    Designed to be lightweight — uses short prompts and low max_tokens.
    """

    MAX_SNIPPET_LENGTH = 1500
    MAX_TOTAL_LENGTH = 5000   # Total context budget for snippets (search + read combined)

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def check(
        self,
        query: str,
        collected_snippets: List[Dict[str, Any]],
    ) -> SearchFeedback:
        """Check if collected information is sufficient for the query.

        Args:
            query: Original user query.
            collected_snippets: List of dicts with keys: title, snippet, source.

        Returns:
            SearchFeedback with sufficiency assessment and suggestions.
        """
        if not collected_snippets:
            return SearchFeedback(
                sufficient=False,
                coverage_score=0.0,
                missing_aspects=["全部 — 未收集到任何文档"],
                suggested_queries=[query],
                suggested_tools=["search"],
                reason="未收集到任何文档片段",
            )

        # Format snippets (truncated for token efficiency)
        formatted = self._format_snippets(collected_snippets)

        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query,
            snippets=formatted,
        )

        try:
            from src.agent.llm_client import ChatMessage
            response = self._llm.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1,
                max_tokens=500,
                model_tier="fast",
            )
            return self._parse_response(response.content)
        except Exception as e:
            logger.warning(f"Sufficiency check failed: {e}")
            # On failure, assume sufficient (don't block the answer)
            return SearchFeedback(
                sufficient=True,
                coverage_score=0.5,
                reason=f"充足性检查失败，默认为充足: {e}",
            )

    def _format_snippets(self, snippets: List[Dict[str, Any]]) -> str:
        """Format snippets into a condensed text for the LLM prompt."""
        parts: List[str] = []
        total_len = 0

        for i, s in enumerate(snippets[:10]):  # Max 10 snippets
            title = s.get("title", f"文档{i+1}")
            snippet = str(s.get("snippet", ""))
            if len(snippet) > self.MAX_SNIPPET_LENGTH:
                snippet = snippet[:self.MAX_SNIPPET_LENGTH] + "..."
            source = s.get("source", "")

            entry = f"[{title}] (来源: {source})\n{snippet}"
            if total_len + len(entry) > self.MAX_TOTAL_LENGTH:
                break
            parts.append(entry)
            total_len += len(entry)

        return "\n\n".join(parts) if parts else "(无文档片段)"

    def _parse_response(self, content: str) -> SearchFeedback:
        """Parse LLM response into SearchFeedback."""
        if not content:
            return SearchFeedback(
                sufficient=True,
                coverage_score=0.5,
                reason="LLM返回空内容，默认为充足",
            )

        # Extract JSON from possible markdown code block
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
            validated = SearchFeedbackModel.model_validate(raw)
            return SearchFeedback(
                sufficient=validated.sufficient,
                coverage_score=validated.coverage_score,
                covered_aspects=validated.covered_aspects,
                missing_aspects=validated.missing_aspects,
                suggested_queries=validated.suggested_queries,
                suggested_tools=validated.suggested_tools,
                reason=validated.reason,
            )
        except Exception as e:
            logger.warning(f"Failed to parse sufficiency response: {e}")
            return SearchFeedback(
                sufficient=True,
                coverage_score=0.5,
                reason="解析充足性响应失败，默认为充足",
            )
