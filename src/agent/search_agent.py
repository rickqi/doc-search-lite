"""Search Agent for semantic Q&A and precision document search.

This module provides the SearchAgent class that combines LLM capabilities
with document search and retrieval tools for intelligent document Q&A.

Architecture (DCI-Agent-Lite inspired):
    Tool Loop (default): LLM-driven tool calling with search/grep/read/rerank
    Pipeline (legacy): Hardcoded 6-step pipeline (analyze→search→read→answer)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from src.agent.base import Agent, AgentResponse, Tool
from src.agent.llm_client import ChatMessage, ChatResponse, LLMClient, ToolCall
from src.agent.query_decomposer import DecompositionResult, QueryDecomposer, SubQuery
from src.agent.sufficient_context import SearchFeedback, SufficientContextChecker
from src.agent.tools.read import ReadTool
from src.agent.tools.search import SearchTool
from src.utils.config import Config

if TYPE_CHECKING:
    from src.search.reranker import ZhipuAIReranker

logger = logging.getLogger(__name__)


# Optimized system prompt for tool-calling paradigm (v0.14: ReAct reasoning added)
SYSTEM_PROMPT = """你是文档搜索助手。你必须通过工具搜索并阅读文档来回答用户问题。

⚠️ 强制交替工作流（search → read 交替，不可连续搜索）:
1. 调用 search 搜索相关文档
2. **立即**调用 read 读取搜索结果中 doc_id 对应的文档内容 — 在读取之前**禁止**进行下一次 search
3. 如果读取后发现信息不足，再发起下一次 search，然后必须再次 read
4. 基于读取到的**文档原文**回答问题

这条规则没有例外：search 和 read 必须交替进行。连续两次 search 而不 read 是严重违规。

🧠 推理要求（每次工具调用前必须遵守）:
在调用每个工具之前，先用一两句话说明你的推理过程：
- 已知：从之前的搜索/阅读中获得了什么关键信息
- 缺失：还需要什么信息才能完整回答用户问题
- 计划：下一步为什么选择这个搜索词或这篇文档

例如：
已知：找到了《年假管理制度》搜索结果（score=0.85），但尚未阅读全文
缺失：年假申请的具体流程步骤和审批节点
计划：读取该文档，定位"申请流程"章节
→ 然后调用 read(doc_id="搜索结果中的doc_id", max_lines=200)

禁止行为:
- ❌ 仅基于搜索结果的 snippet/摘要就直接回答 — snippet 信息不完整，必须 read 全文
- ❌ 不调用 read 就给出最终回答 — 这是最常见的错误，绝对不允许
- ❌ 凭借自身知识回答 — 所有信息必须来自文档原文
- ❌ 连续多次 search 而不调用 read — search 后必须立即 read，禁止在 read 前做第二次 search
- ❌ 先搜齐所有维度再统一读 — 正确的顺序是 search→read→search→read，逐个维度深入

正确示例（多维度查询）:
  第1轮: [推理] 用户问了三个问题，先处理第一个：夫妻共同债务标准。
          search(query="民法典1064条 夫妻共同债务") → 获取结果含 doc_id
  第2轮: [推理] 搜索到民法典相关文档，需要阅读具体法条内容。
          read(doc_id="搜索结果中的doc_id", max_lines=200) → 读取文档原文
  第3轮: [推理] 已获得夫妻债务认定标准。现在处理第二个问题：子女名下房产执行。
          search(query="未成年子女 名下房产 强制执行 家庭共有财产") → 获取新结果
  第4轮: [推理] 需要阅读子女房产执行的具体判例。
          read(doc_id="新搜索结果的doc_id", max_lines=200) → 读取判例原文
  第5轮: 基于两个维度的文档原文，综合回答问题

可用工具:
- search: BM25全文检索。参数: query(查询), limit(数量)
- read: 读取文档完整内容。参数: doc_id或source_path, max_lines(建议200-300)
- summarize: 读取并总结文档要点（节省token）。参数: doc_id或source_path, focus(关注重点,可选)
- grep: 正则搜索原始文件。参数: pattern(正则), file_filter(文件过滤)
- rerank: 对搜索结果重排序。参数: query, documents(文本列表), top_n
- bash: 在原始文档目录执行只读命令(rg/find/head/tail/cat/wc)。参数: command

🔍 BM25 搜索策略（⚠️ 精确关键词优于自然语言！）:
- BM25 是关键词匹配引擎，不是语义搜索。用精炼关键词效果远优于长句子:
  * ✅ 正确: "法释2018年2号 第三条"  → 精确命中目标文档
  * ❌ 错误: "请搜索关于夫妻债务纠纷的那个司法解释"  → 关键词被稀释
- 从用户问题中提取具体编号、日期、文件名直接作为搜索词:
  * "法释〔2025〕10号" → 直接搜索: "法释 2025 10号"
  * "民法典第1064条" → 直接搜索: "民法典 1064 条 夫妻共同债务"
  * "查封扣押冻结规定" → 直接搜索: "查封 扣押 冻结 不得拍卖"
- 搜索不到结果时，尝试用更通用的术语替代用户口语化表述:
  * 把用户通俗说法转换为可能的官方制度名称（如"脱敏制度"→"数据脱敏管理办法" "数据安全分类分级"）
  * 尝试只搜索核心关键词，去掉限定词（如"脱敏制度2.0"→"数据脱敏" "数据安全"）
  * 尝试组合搜索: 关键词 + "管理办法" 或 "规范" 或 "制度"（如"脱敏 管理办法"）
- 每次搜索后评估结果质量：如果标题明显不相关，立即换不同关键词重新搜索
- 不要连续超过3次用相同或相似的词搜索 — 每次失败后必须显著改变搜索策略

回答规则:
- 引用具体内容时标注 [来源文档名]
- 不确定时说明"文档中未找到相关信息"
- 回答简洁，不超过500字"""

# Kept for pipeline mode's _analyze_query()
QUERY_ANALYSIS_PROMPT = """分析查询意图，提取关键搜索词。

查询: {query}

输出JSON格式:
{{"action": "search|direct", "search_query": "提取的关键词", "grep_pattern": "可选的精确匹配正则"}}

规则:
- 提取核心关键词，去掉虚词(的、了、吗、呢、什么、如何等)
- 如果查询包含具体名词/术语，保留原文
- 如果是复合查询，用空格分隔关键词
- 简单问候→direct，其他→search
- grep_pattern仅当查询包含需要精确匹配的内容时填写"""

# Built-in skill prompt variants (appended to SYSTEM_PROMPT when --skill is used)
SKILL_PROMPTS: Dict[str, str] = {
    "summarize": (
        "\n\n输出要求: 用200字以内概括关键要点。用项目符号列出，每个要点不超过30字。"
        "不要引用原文，用自己的话概括。"
    ),
    "compare": (
        "\n\n输出要求: 以对比表格形式呈现各文档的异同点。表格列: 条款 | 文档A | 文档B | 差异说明。"
        "突出关键变化。"
    ),
    "extract-table": (
        "\n\n输出要求: 提取所有涉及数据标准、金额、日期、数量的内容。用Markdown表格呈现，"
        "列: 项目 | 内容 | 来源。精确引用原文数字。"
    ),
    "detailed": (
        "\n\n输出要求: 详细分析文档内容，分章节总结。每个章节标注来源文档和段落。"
        "引用原文关键语句时用引用格式(>)。"
    ),
    "timeline": (
        "\n\n输出要求: 按时间线梳理文档中的版本变更历史。"
        "格式: YYYY-MM-DD | 事件 | 关键变更 | 来源文档。"
    ),
    "action-items": (
        "\n\n输出要求: 提取所有行动项。格式表格: 序号 | 行动项 | 责任人 | 截止日期 | 来源。"
        "缺失信息标注'未明确'。"
    ),
}


def _build_system_prompt(
    skill: Optional[str] = None,
    loaded_skill_content: Optional[str] = None,
) -> str:
    """Build system prompt with optional skill-specific instructions appended.

    Args:
        skill: Skill name (key into SKILL_PROMPTS), or None for default prompt.
        loaded_skill_content: External SKILL.md content to append, or None.

    Returns:
        Complete system prompt string.
    """
    prompt = SYSTEM_PROMPT
    if skill and skill in SKILL_PROMPTS:
        prompt = prompt + SKILL_PROMPTS[skill]
    if loaded_skill_content:
        prompt = prompt + "\n\n" + loaded_skill_content
    return prompt


@dataclass
class SearchResult:
    """Represents a processed search result."""

    doc_id: str
    title: str
    score: float
    snippet: str
    source_path: Optional[str] = None
    content: Optional[str] = None


class SearchAgent(Agent):
    """Agent for semantic Q&A and precision document search.

    This agent combines LLM capabilities with search and read tools
    to provide intelligent document-based question answering.

    Supports two execution modes:
    - "tool_loop" (default): LLM-driven tool calling loop (DCI-Agent-Lite paradigm)
    - "pipeline": Legacy hardcoded 6-step pipeline

    Attributes:
        _llm_client: LLMClient for chat completions
        _config: Configuration object
        _max_search_results: Maximum search results to consider
        _max_read_docs: Maximum documents to read in detail
        _max_context_tokens: Maximum context tokens for LLM

    Example:
        >>> from src.utils.config import Config
        >>> config = Config.from_env()
        >>> agent = SearchAgent(config, search_tool, read_tool)
        >>> response = agent.run("什么是绩效考核?")
        >>> print(response.answer)
    """

    MAX_TOOL_ITERATIONS = 8  # Reduced from 100 (ITER-RETGEN: 2nd iteration gives biggest boost)
    MAX_FEEDBACK_ROUNDS = 2  # Maximum sufficiency check iterations
    MAX_EXPANSION_VARIANTS = 5  # Aligned with Microsoft AgenticRAG multi-query search
    BEST_OF_K_THRESHOLD = 0.5  # P5: Reduced from 0.7 — avoid 3x rerun for borderline cases
    BEST_OF_K_RUNS = 3         # P5: Total runs in Best-of-K

    # Query complexity signals for adaptive strategy
    COMPLEX_SIGNALS = [
        "和", "与", "以及", "同时", "对比", "比较", "区别", "差异",
        "为什么", "原因", "分析", "影响", "趋势",
        "变更", "变化", "不同", "分别",
    ]
    MEDIUM_SIGNALS = [
        "如何", "怎么", "什么", "哪些", "多少",
        "制度", "流程", "规定", "政策", "标准", "办法",
        "申请", "办理", "审批", "操作",
    ]

    def __init__(
        self,
        config: Config,
        search_tool: SearchTool,
        read_tool: ReadTool,
        llm_client: Optional[LLMClient] = None,
        max_search_results: int = 10,
        max_read_docs: int = 3,
        max_context_tokens: int = 3000,
        use_rerank: bool = False,
        mode: str = "tool_loop",
        usage_tracker=None,
        diagnostics=None,
    ) -> None:
        """Initialize the SearchAgent.

        Args:
            config: Configuration object with API settings
            search_tool: SearchTool instance for document search
            read_tool: ReadTool instance for document reading
            llm_client: Optional pre-configured LLMClient
            max_search_results: Maximum search results to retrieve
            max_read_docs: Maximum documents to read in detail
            max_context_tokens: Maximum context tokens for LLM
            use_rerank: Whether to use ZhipuAI Rerank API for result reranking (pipeline mode only)
            mode: Execution mode - "tool_loop" (default) or "pipeline"
            usage_tracker: Optional UsageTracker for recording API usage
        """
        super().__init__()

        self._config = config
        self._llm_client = llm_client or LLMClient(config)
        self._max_search_results = max_search_results
        self._max_read_docs = max_read_docs
        self._max_context_tokens = max_context_tokens
        self._use_rerank = use_rerank
        self._mode = mode
        self._usage_tracker = usage_tracker
        self._session_id: Optional[str] = None
        self._diagnostics = diagnostics

        # Token budget tracking
        self._total_tokens_used = 0
        self._max_session_tokens = 50000

        # Register tools
        self.register_tool(search_tool)
        self.register_tool(read_tool)

        # Lazy-init reranker (only when needed in pipeline mode)
        self._reranker: ZhipuAIReranker | None = None

        logger.info(
            f"Initialized SearchAgent with {len(self.tools)} tools "
            f"(rerank={use_rerank}, mode={mode})"
        )

    @property
    def name(self) -> str:
        """Unique identifier for the agent.

        Returns:
            str: The agent name 'search_agent'
        """
        return "search_agent"

    @property
    def tokens_used(self) -> int:
        """Total tokens used across all LLM calls in this session.

        Returns:
            int: Total tokens consumed
        """
        return self._total_tokens_used

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        skill: Optional[str] = None,
        loaded_skill_content: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResponse:
        """Execute the search agent to answer a query.

        Dispatches to either tool_loop or pipeline mode based on _mode.

        Args:
            query: The user's question or search query
            context: Optional context (e.g., {"mode": "precision"})
            skill: Optional skill name for output formatting (e.g. "summarize")
            loaded_skill_content: Optional external SKILL.md content to inject

        Returns:
            AgentResponse containing:
                - success: Whether the query was answered
                - answer: The answer text
                - sources: List of source document paths
                - tool_calls: Record of tools used
                - confidence: Confidence score (0-1)
                - tokens_used: Total tokens consumed
                - processing_time: Time taken in seconds
        """
        if self._usage_tracker:
            self._session_id = self._usage_tracker.start_session(query=query, mode=self._mode)

        # === Agent Memory: 精确命中检查 ===
        # 如果历史中有完全相同的问题且成功，直接返回历史答案（零延迟）
        self._memory_recall = None
        self._memory_ctx_injected = False
        try:
            from src.stats.memory import AgentMemory as _AgentMemory
            _mem = _AgentMemory()
            _recalled = _mem.recall(query)
            if _recalled and _recalled.get("source") == "exact_hit":
                answer = _recalled.get("answer", "")
                if answer:
                    logger.info("AgentMemory: exact hit for '%s' — returning cached answer", query[:60])
                    # Reconstruct session_id for learn() later
                    _recalled_sid = _recalled.get("session_id", "") or f"mem_{abs(hash(query))}"
                    result = AgentResponse(
                        success=True,
                        answer=answer,
                        sources=[],
                        tool_calls=[],
                        reasoning="(来自历史记忆 — 之前已回答过该问题)",
                        confidence=1.0,
                        processing_time=0.0,
                        tokens_used=0,
                    )
                    # Still log the hit for tracking
                    if not getattr(self, "_no_log", False):
                        try:
                            from src.stats.search_logger import SearchLogger as _SL
                            _SL.log_async(session_id=_recalled_sid, query=query, response=result, source="agent", search_mode="agent_memory_hit", index_path=getattr(self, "_index_path", ""), skill=skill or "")
                        except Exception:
                            pass
                    return result
            if _recalled and _recalled.get("source") == "fuzzy_hit":
                logger.info("AgentMemory: fuzzy hit for '%s' — will inject context", query[:60])
                self._memory_recall = _recalled
        except Exception as e:
            logger.debug("AgentMemory recall skipped (non-fatal): %s", e)

        if self._mode == "pipeline":
            result = self._run_pipeline(query, context)
        else:
            result = self._run_tool_loop(query, context, skill=skill, loaded_skill_content=loaded_skill_content, history=history)

        # P5: Best-of-K — 对 complex 查询且低置信度时自动运行多次取最优
        if (self._mode != "pipeline"
                and getattr(self, "_complexity", "") == "complex"
                and (result.confidence or 0.0) < self.BEST_OF_K_THRESHOLD
                and not os.environ.get("DISABLE_BEST_OF_K")):
            try:
                _b_start = time.time()
                _results = [result]
                for _i in range(self.BEST_OF_K_RUNS - 1):  # 再跑 K-1 次
                    _extra = self._run_tool_loop(
                        query, context, skill=skill,
                        loaded_skill_content=loaded_skill_content,
                        history=history,
                    )
                    _results.append(_extra)
                # 选最优：confidence 优先，其次答案长度
                result = max(_results, key=lambda r: (
                    r.confidence or 0.0, len(r.answer or ""),
                ))
                result.metadata["best_of_k"] = len(_results)
                result.metadata["best_of_k_selected_index"] = _results.index(result)
                if self._diagnostics:
                    try:
                        self._diagnostics.record_step("best_of_k", (time.time() - _b_start) * 1000)
                    except Exception:
                        logger.warning("Failed to record best_of_k diagnostics")
            except Exception:
                pass  # Best-of-K 失败时使用原始 result

        # === 异步搜索记录 (fire-and-forget, 不影响性能) ===
        from src.stats.search_logger import SearchLogger as _SearchLogger
        _srch_sid = getattr(self, "_srch_session_id", "") or self._session_id or _SearchLogger.generate_session_id()
        if not getattr(self, "_no_log", False):
            try:
                _SearchLogger.log_async(
                    session_id=_srch_sid,
                    query=query,
                    response=result,
                    source=getattr(self, "_search_source", "agent"),
                    search_mode="agent",
                    index_path=getattr(self, "_index_path", ""),
                    raw_dir=str(getattr(self, "_raw_dirs", [""])[0]) if getattr(self, "_raw_dirs", None) else "",
                    model=getattr(self._llm_client, "model", "") if self._llm_client else "",
                    skill=skill or "",
                )
            except Exception:
                pass  # 记录失败不影响搜索

        # === Agent Memory: 学习 (异步, 不阻塞) ===
        try:
            from src.stats.memory import AgentMemory as _AMem2
            _mem2 = _AMem2()
            _mem2.learn(_srch_sid, {
                "mode": self._mode,
                "index_path": getattr(self, "_index_path", ""),
                "confidence": getattr(result, "confidence", 0),
                "tool_count": len(getattr(result, "tool_calls", [])),
                "latency": getattr(result, "processing_time", 0),
                "search_count": sum(1 for tc in getattr(result, "tool_calls", []) if tc.get("tool") == "search"),
                "read_count": sum(1 for tc in getattr(result, "tool_calls", []) if tc.get("tool") == "read"),
            })
        except Exception:
            pass  # 学习失败不影响搜索

        return result

    # ─────────────────────────────────────────────────────
    # Tool Loop Mode (default)
    # ─────────────────────────────────────────────────────

    def _run_tool_loop(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        skill: Optional[str] = None,
        loaded_skill_content: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResponse:
        """Execute search using LLM-driven tool calling loop (DCI-Agent-Lite paradigm).

        The LLM autonomously decides which tools to call, how many times,
        and when to produce a final answer.

        Args:
            query: The user's question or search query
            context: Optional context information
            skill: Optional skill name for output formatting

        Returns:
            AgentResponse with answer, sources, and tool call records
        """
        start_time = time.time()
        tool_calls: List[Dict[str, Any]] = []
        self._total_tokens_used = 0

        if self._diagnostics:
            self._diagnostics.start_query(query, complexity="unknown")

        try:
            messages = [
                ChatMessage(role="system", content=_build_system_prompt(skill, loaded_skill_content)),
            ]

            # Agent Memory: 模糊匹配 context 注入 (如果有)
            if getattr(self, "_memory_recall", None):
                try:
                    from src.stats.memory import AgentMemory as _AMem
                    _mem = _AMem()
                    ctx_text = _mem.format_context(self._memory_recall)
                    if ctx_text:
                        messages.append(ChatMessage(role="system", content=ctx_text))
                        self._memory_ctx_injected = True
                        logger.debug("AgentMemory: context injected into system prompt")
                except Exception:
                    pass

            if history:
                max_turns = int(os.environ.get("CONVERSATION_MEMORY_WINDOW", "5"))
                recent = history[-max_turns * 2:]  # each turn = user + assistant
                for msg in recent:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append(ChatMessage(role=role, content=content))
            messages.append(ChatMessage(role="user", content=query))

            # Adaptive strategy based on query complexity
            complexity = self._classify_query_complexity(query)
            self._complexity = complexity  # P5: Store for Best-of-K check in run()
            logger.info(f"Query complexity: {complexity} (query: {query[:50]}...)")
            if self._diagnostics:
                self._diagnostics._complexity = complexity

            if complexity == "simple":
                # Simple strategy: skip expansion, fewer iterations (2 = search + read)
                max_iter = 2
            elif complexity == "light":
                # Light strategy: moderate iterations (4), skip expansion
                # 4 iterations give LLM enough room to refine searches without expansion overhead
                max_iter = 4
            elif complexity == "complex":
                # Complex strategy: decomposition + expansion + more iterations
                max_iter = self.MAX_TOOL_ITERATIONS

                # Step 1: Try query decomposition for compound queries
                if self._diagnostics:
                    self._diagnostics.record_step_start("decompose")
                decomposer = QueryDecomposer(self._llm_client)
                decomposition = decomposer.decompose(query)
                if self._diagnostics:
                    self._diagnostics.record_step_end("decompose")
                self._total_tokens_used += decomposer._last_usage.get("total_tokens", 0)
                if decomposition.needs_decomposition and decomposition.sub_queries:
                    logger.info(
                        f"Query decomposed into {len(decomposition.sub_queries)} sub-queries"
                    )
                    # Execute sub-queries in parallel and inject merged context
                    if self._diagnostics:
                        self._diagnostics.record_step_start("sub_queries_parallel")
                    merged_context = self._execute_sub_queries_parallel(
                        decomposition.sub_queries, tool_calls,
                    )
                    if self._diagnostics:
                        self._diagnostics.record_step_end("sub_queries_parallel")
                    if merged_context:
                        messages.append(ChatMessage(
                            role="system",
                            content=merged_context,
                        ))
                    else:
                        # Fallback: inject hint-style message when parallel exec fails
                        sub_notes = "\n".join(
                            f"- [{sq.aspect}] {sq.query}"
                            for sq in decomposition.sub_queries
                        )
                        messages.append(ChatMessage(
                            role="system",
                            content=(
                                f"🔍 查询分解: 用户查询包含多个方面，请依次搜索:\n{sub_notes}\n"
                                "请确保每个方面都搜索到相关文档后再综合回答。"
                            ),
                        ))

                # Step 2: Query expansion — skip if decomposition already covered aspects
                if decomposition and decomposition.sub_queries and len(decomposition.sub_queries) >= 2:
                    # Decomposition already provides diverse queries — skip expansion
                    pass
                else:
                    if self._diagnostics:
                        self._diagnostics.record_step_start("expand")
                    expanded_queries = self._expand_query(query)
                    if self._diagnostics:
                        self._diagnostics.record_step_end("expand")
                    if len(expanded_queries) > 1:
                        expansion_note = "\n".join(f"- {q}" for q in expanded_queries[1:])
                        # Auto-execute expanded queries to find broader results
                        expanded_docs = set()
                        expanded_hits: List[str] = []
                        for eq in expanded_queries[1:4]:  # Try top 3 variants
                            try:
                                eq_result = self._search_tool.execute(query=eq, limit=3)
                                if eq_result.success and eq_result.data:
                                    import json as _json
                                    eq_data = _json.loads(eq_result.data) if isinstance(eq_result.data, str) else eq_result.data
                                    for r in eq_data.get("results", [])[:3]:
                                        did = r.get("doc_id", "")
                                        if did and did not in expanded_docs:
                                            expanded_docs.add(did)
                                            expanded_hits.append(f"- [{r.get('title', did)}] (query: {eq})")
                            except Exception:
                                logger.debug("Failed to expand query: %s", eq)
                        if expanded_hits:
                            expansion_note += "\n\n🔍 扩展搜索已找到以下可能相关的文档（请读取验证）:\n" + "\n".join(expanded_hits)
                        messages.append(ChatMessage(
                            role="system",
                            content=f"搜索建议: 以下是与用户查询相关的替代表达，可在搜索工具中使用:\n{expansion_note}"
                        ))
            else:
                # Medium strategy: standard behavior
                max_iter = self.MAX_TOOL_ITERATIONS
                if self._diagnostics:
                    self._diagnostics.record_step_start("expand")
                expanded_queries = self._expand_query(query)
                if self._diagnostics:
                    self._diagnostics.record_step_end("expand")
                if len(expanded_queries) > 1:
                    expansion_note = "\n".join(f"- {q}" for q in expanded_queries[1:])
                    messages.append(ChatMessage(
                        role="system",
                        content=f"搜索建议: 以下是与用户查询相关的替代表达，可在搜索工具中使用:\n{expansion_note}"
                    ))

            # ── P4: Query Analysis (complex only, fast tier) ──
            # P1-2: Skip for medium — analysis adds 2s latency for marginal
            # benefit on queries that already have strong keyword signals
            if complexity == "complex":
                if self._diagnostics:
                    self._diagnostics.record_step_start("query_analysis")
                try:
                    analysis_resp = self._llm_client.chat(
                        messages=[
                            ChatMessage(
                                role="system",
                                content=(
                                    "分析用户查询，输出结构化分析（不超过 200 字）：\n"
                                    "1. 核心问题：<一句话概括>\n"
                                    "2. 需要的信息维度：<列出 2-4 个维度>\n"
                                    "3. ⚠️ BM25关键词（精炼！）: <列出 3-5 个用空格分隔的关键词，优先提取文档编号/法条号/文件名作为关键词。不要用自然语言句子。>\n"
                                    "4. 可能的文档类型：<制度/流程/标准/通知/法律/司法解释/行政法规/其他>"
                                ),
                            ),
                            ChatMessage(role="user", content=query),
                        ],
                        model_tier="fast",
                        max_tokens=300,
                        temperature=0.1,
                    )
                    self._total_tokens_used += analysis_resp.usage.get("total_tokens", 0)
                    if analysis_resp.content and analysis_resp.content.strip():
                        messages.append(ChatMessage(
                            role="system",
                            content=f"📋 查询分析：\n{analysis_resp.content.strip()}",
                        ))
                        logger.info(f"Query analysis injected ({len(analysis_resp.content)} chars)")
                except Exception as e:
                    logger.warning(f"Query analysis failed (non-blocking): {e}")
                if self._diagnostics:
                    self._diagnostics.record_step_end("query_analysis")

            # Callback to track tool calls for AgentResponse
            _prev_search_doc_ids: list = []
            _search_converged = False
            _consecutive_searches = 0  # Track consecutive searches without read
            _exploration_nudged = False  # P3: track if we've already nudged once

            def on_tool_call(tc: ToolCall, result: Any) -> None:
                nonlocal _search_converged, _consecutive_searches
                entry = {
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "success": getattr(result, "success", True),
                }
                # Track consecutive searches for early-stop
                if tc.name == "search":
                    _consecutive_searches += 1
                elif tc.name == "read":
                    _consecutive_searches = 0
                # Capture tool metadata (execution_time, lines_read, etc.)
                result_meta = getattr(result, "metadata", None)
                if isinstance(result_meta, dict):
                    entry["metadata"] = result_meta
                entry["timestamp"] = time.time()

                # Early-stop: detect search result convergence
                if tc.name == "search" and entry["success"]:
                    try:
                        raw = getattr(result, "content", None) or getattr(result, "data", None)
                        if raw and isinstance(raw, str):
                            data = json.loads(raw)
                            current_ids = [r.get("doc_id") for r in data.get("results", [])[:5]]
                            if _prev_search_doc_ids and current_ids:
                                overlap = len(set(current_ids) & set(_prev_search_doc_ids))
                                if overlap >= min(3, len(current_ids)):
                                    _search_converged = True
                                    logger.info(f"Search results converged (overlap {overlap}/{len(current_ids)})")
                            _prev_search_doc_ids.clear()
                            _prev_search_doc_ids.extend(current_ids)
                    except (json.JSONDecodeError, AttributeError):
                        logger.debug("Failed to parse search result for convergence detection")
                # Preserve result content for sufficiency checks.
                # llm_client.ToolResult uses .content (str), base.ToolResult uses .data
                raw_content = getattr(result, "content", None) or getattr(result, "data", None)
                if raw_content:
                    # Guard against accidental object repr (e.g. "<src.agent.base.ToolResult ...>")
                    if isinstance(raw_content, str):
                        entry["content"] = raw_content[:2000] if len(raw_content) > 2000 else raw_content
                    elif hasattr(raw_content, "data"):
                        # Got a base.ToolResult instead of a string — extract .data
                        inner = raw_content.data
                        if isinstance(inner, str):
                            entry["content"] = inner[:2000] if len(inner) > 2000 else str(inner)
                        else:
                            entry["content"] = str(inner)[:2000]
                    else:
                        entry["content"] = str(raw_content)[:2000]
                tool_calls.append(entry)

            if self._diagnostics:
                self._diagnostics.record_step_start("tool_loop")

            # Convergence early-stop callback: when _search_converged is set,
            # when LLM makes consecutive searches without reading.
            # P3: On first consecutive search without read, inject a
            # nudge message instead of hard-stopping, giving the
            # LLM one more chance to read or change strategy.
            def _check_convergence() -> bool:
                nonlocal _exploration_nudged, _consecutive_searches
                if _search_converged:
                    return True
                if _consecutive_searches >= 1:
                    if not _exploration_nudged:
                        # P3: Nudge instead of hard stop — inject guidance
                        messages.append(ChatMessage(
                            role="user",
                            content=(
                                "你刚完成了一次搜索，但尚未读取任何文档。\n"
                                "⚠️ 请立即调用 read 读取搜索到的文档（使用搜索结果中的 doc_id）。\n"
                                "在读取之前不要进行下一次 search。"
                            ),
                        ))
                        _exploration_nudged = True
                        _consecutive_searches = 0  # Reset to give one more chance
                        logger.info("Convergence nudge: injected read-now prompt")
                        return False
                    logger.info(f"Early-stop: consecutive search without read (post-nudge)")
                    return True
                return False

            response = self._llm_client.chat_with_tools(
                messages=messages,
                tools=self.tools,
                max_iterations=max_iter,
                temperature=0.3,
                max_tokens=2000,
                max_total_tokens=self._max_session_tokens,
                context_management="level3",
                on_tool_call=on_tool_call,
                should_stop=_check_convergence,
            )
            if self._diagnostics:
                self._diagnostics.record_step_end("tool_loop")

            # Log convergence detection result
            if _search_converged:
                logger.info("Search converged — skipping feedback loop if applicable")

            # Post-loop guardrail: if no read was called, force-read top search results
            # and regenerate answer based on actual document content
            read_called = any(tc.get("tool") == "read" for tc in tool_calls)
            if not read_called and tool_calls:
                # P0-2: Log what happened to diagnose why LLM skipped reads
                tool_names = [tc.get("tool", "?") for tc in tool_calls]
                search_args = [tc.get("arguments", {}).get("query", "")[:60] for tc in tool_calls if tc.get("tool") == "search"]
                logger.warning(
                    f"Agent completed {len(tool_calls)} tool calls without reading: "
                    f"tools={tool_names}, searches={search_args}. "
                    f"Force-reading top results."
                )
                if self._diagnostics:
                    self._diagnostics.record_step_start("forced_read")
                # Re-search to get doc_ids (can't extract from tool_calls records)
                search_tool = self.get_tool("search")
                read_tool = self.get_tool("read")
                if search_tool and read_tool:
                    # Use original query for re-search
                    search_result = search_tool.execute(query=query, limit=3)
                    if search_result.success and search_result.data:
                        try:
                            data = json.loads(search_result.data)
                            doc_ids = [r.get("doc_id") for r in data.get("results", []) if r.get("doc_id")]
                        except json.JSONDecodeError:
                            doc_ids = []

                        doc_contents = []
                        for doc_id in doc_ids[:3]:
                            read_result = read_tool.execute(doc_id=doc_id, max_lines=200)
                            tool_calls.append({
                                "tool": "read",
                                "arguments": {"doc_id": doc_id, "max_lines": 200},
                                "success": read_result.success,
                                "forced": True,
                            })
                            if read_result.success and read_result.data:
                                source = read_result.metadata.get("source_path", doc_id) if read_result.metadata else doc_id
                                doc_contents.append(f"[文档: {source}]\n{read_result.data[:3000]}")

                        if doc_contents:
                            # Regenerate answer with actual document content
                            context = "\n\n".join(doc_contents)
                            forced_messages = [
                                ChatMessage(role="system", content="你是文档搜索助手。基于以下文档内容回答问题。必须引用文档原文，标注来源文档名。"),
                                ChatMessage(role="user", content=f"问题: {query}\n\n参考文档:\n{context}"),
                            ]
                            forced_response = self._llm_client.chat(
                                messages=forced_messages,
                                temperature=0.3,
                                max_tokens=2000,
                            )
                            self._total_tokens_used += forced_response.usage.get("total_tokens", 0)
                            # Override the original response with the document-based one
                            response = forced_response

                if self._diagnostics:
                    self._diagnostics.record_step_end("forced_read")

            self._total_tokens_used += response.usage.get("total_tokens", 0)

            # ── Sufficient Context Feedback Loop ──
            # For medium/complex queries, check if collected info is sufficient
            feedback_rounds = 0
            if complexity in ("medium", "complex") and tool_calls:
                while feedback_rounds < self.MAX_FEEDBACK_ROUNDS:
                    if self._diagnostics:
                        self._diagnostics.record_step_start(f"sufficiency_{feedback_rounds + 1}")
                    feedback = self._check_sufficient_context(query, tool_calls)
                    if self._diagnostics:
                        self._diagnostics.record_step_end(f"sufficiency_{feedback_rounds + 1}")
                    logger.info(
                        f"Sufficiency check (round {feedback_rounds + 1}): "
                        f"sufficient={feedback.sufficient}, score={feedback.coverage_score:.2f}"
                    )

                    # Record feedback in tool_calls for observability
                    tool_calls.append({
                        "tool": "_sufficiency_check",
                        "arguments": {"round": feedback_rounds + 1},
                        "success": True,
                        "sufficient": feedback.sufficient,
                        "coverage_score": feedback.coverage_score,
                        "missing_aspects": feedback.missing_aspects,
                        "suggested_queries": feedback.suggested_queries,
                    })

                    if feedback.sufficient or not feedback.suggested_queries:
                        break

                    # Inject feedback and continue searching
                    self._inject_feedback(messages, feedback)
                    if self._diagnostics:
                        self._diagnostics.record_step_start(f"feedback_{feedback_rounds + 1}")
                    feedback_response = self._llm_client.chat_with_tools(
                        messages=messages,
                        tools=self.tools,
                        max_iterations=min(2, self.MAX_TOOL_ITERATIONS),
                        temperature=0.3,
                        max_tokens=2000,
                        max_total_tokens=self._max_session_tokens,
                        context_management="level3",
                        on_tool_call=on_tool_call,
                    )
                    if self._diagnostics:
                        self._diagnostics.record_step_end(f"feedback_{feedback_rounds + 1}")
                    self._total_tokens_used += feedback_response.usage.get("total_tokens", 0)

                    # Update response if new one has content
                    if feedback_response.content and feedback_response.content.strip():
                        response = feedback_response

                    feedback_rounds += 1

            # ── Final Answer Quality Guardrail ──
            # If the final response looks like an intermediate step (too short,
            # or contains tool-call-like patterns), regenerate from documents.
            answer_text = (response.content or "").strip()
            needs_regeneration = (
                not answer_text
                or len(answer_text) < 80  # Too short to be a real answer
                or "我需要先" in answer_text[:20]  # Starts with "I need to first..."
                or "让我先" in answer_text[:20]  # Starts with "Let me first..."
                or answer_text.startswith("search(")  # Raw tool call text
                or answer_text.startswith("read(")   # Raw tool call text
                or "<tool_call>" in answer_text[:200]  # XML-format tool call leaked into answer (liteLLM)
                or "</tool_call>" in answer_text[:200]
                or "文档内容未直接显示" in answer_text  # LLM didn't use tool results
                or "无法直接引用原文" in answer_text   # LLM claims it can't cite
                or "请分享相关文档" in answer_text      # LLM asks user for docs
                or "请提供" in answer_text[:50]         # Asks user for input
            )

            if needs_regeneration and tool_calls:
                logger.info("Final answer looks incomplete, regenerating from collected documents")
                if self._diagnostics:
                    self._diagnostics.record_step_start("regenerate")
                regenerated = self._regenerate_answer_from_docs(query, tool_calls)
                if self._diagnostics:
                    self._diagnostics.record_step_end("regenerate")
                if regenerated and regenerated.content and regenerated.content.strip():
                    response = regenerated
                else:
                    # Regeneration failed — use clean fallback, never leak LLM's intermediate think-text
                    from src.agent.llm_client import ChatResponse
                    response = ChatResponse(
                        content="无法找到相关信息。",
                        usage={"total_tokens": 0},
                    )
                    logger.warning("Answer regeneration returned empty — using fallback")

            # ── Draft Grounding Verification (complex queries only) ──
            # Verify the final answer is grounded in collected documents
            if complexity == "complex" and response.content and tool_calls:
                draft = response.content.strip()
                if len(draft) > 50:  # Only verify substantive answers
                    if self._diagnostics:
                        self._diagnostics.record_step_start("verify_draft")
                    draft_feedback = self._verify_draft_grounding(query, draft, tool_calls)
                    if self._diagnostics:
                        self._diagnostics.record_step_end("verify_draft")
                    logger.info(
                        f"Draft grounding: sufficient={draft_feedback.sufficient}, "
                        f"score={draft_feedback.coverage_score:.2f}"
                    )

                    # ── P1: Verification Recovery Loop ──
                    # If verification fails with actionable suggestions, search
                    # the missing aspects and regenerate the answer.
                    if (
                        not draft_feedback.sufficient
                        and draft_feedback.suggested_queries
                    ):
                        if self._diagnostics:
                            self._diagnostics.record_step_start("verify_recovery")
                        logger.info(
                            f"Draft verification failed — recovering with "
                            f"{len(draft_feedback.suggested_queries[:2])} supplementary searches"
                        )

                        recovery_tools = self._build_tool_dict()
                        for sq in draft_feedback.suggested_queries[:2]:
                            search_tool = recovery_tools.get("search")
                            read_tool = recovery_tools.get("read")
                            if not search_tool or not read_tool:
                                continue
                            try:
                                sr = search_tool.execute(query=sq, limit=3)
                                sr_data = sr.data if hasattr(sr, "data") else str(sr)
                                tool_calls.append({
                                    "tool": "search",
                                    "arguments": {"query": sq},
                                    "success": True,
                                    "content": str(sr_data)[:2000],
                                    "_verification_recovery": True,
                                })
                                # Read the top document from recovery search
                                if isinstance(sr_data, str):
                                    try:
                                        sr_parsed = json.loads(sr_data)
                                        top_doc_id = (
                                            sr_parsed.get("results", [{}])[0]
                                            .get("doc_id", "")
                                        )
                                    except (json.JSONDecodeError, IndexError):
                                        top_doc_id = ""
                                else:
                                    results = getattr(sr_data, "results", [])
                                    top_doc_id = results[0].get("doc_id", "") if results else ""

                                if top_doc_id:
                                    rr = read_tool.execute(doc_id=top_doc_id, max_lines=200)
                                    rr_content = (
                                        rr.data if hasattr(rr, "data") else str(rr)
                                    )
                                    tool_calls.append({
                                        "tool": "read",
                                        "arguments": {"doc_id": top_doc_id},
                                        "success": True,
                                        "content": str(rr_content)[:2000],
                                        "_verification_recovery": True,
                                    })
                            except Exception as e:
                                logger.warning(f"Verification recovery search failed for '{sq}': {e}")

                        # Regenerate answer with recovered documents
                        regenerated = self._regenerate_answer_from_docs(query, tool_calls)
                        if regenerated and regenerated.content and regenerated.content.strip():
                            response = regenerated
                            logger.info("Draft verification recovery — answer regenerated")
                        if self._diagnostics:
                            self._diagnostics.record_step_end("verify_recovery")

                    tool_calls.append({
                        "tool": "_draft_verification",
                        "arguments": {},
                        "success": True,
                        "sufficient": draft_feedback.sufficient,
                        "coverage_score": draft_feedback.coverage_score,
                        "missing_aspects": draft_feedback.missing_aspects,
                    })

            # Extract sources from tool call arguments
            sources = self._extract_sources_from_tool_calls(tool_calls)
            search_hits = self._extract_search_hits_from_tool_calls(tool_calls)

            answer_text = response.content.strip() if response.content else "无法找到相关信息。"
            result = self._build_response(
                success=True,
                answer=answer_text,
                sources=sources,
                search_hits=search_hits,
                tool_calls=tool_calls,
                confidence=self._calculate_tool_loop_confidence(
                    tool_calls, query=query, answer=answer_text
                ),
                tokens_used=self._total_tokens_used,
                start_time=start_time,
            )

            # Attach diagnostics data and finish collection
            if self._diagnostics and self._diagnostics.is_active:
                llm_stats = self._diagnostics.get_llm_stats()
                result.step_timings = self._diagnostics.get_step_timings()
                result.llm_call_count = llm_stats["call_count"]
                result.llm_latency_total = llm_stats["latency_total_ms"] / 1000
                result.retry_count = llm_stats["retry_count"]
                result.cache_hits = llm_stats["cache_hits"]
                self._diagnostics.finish(
                    success=True,
                    model=getattr(self._llm_client, 'model', ''),
                    provider=os.getenv('LLM_PROVIDER', 'glm'),
                    search_count=sum(1 for tc in tool_calls if tc.get("tool") == "search"),
                    read_count=sum(1 for tc in tool_calls if tc.get("tool") == "read"),
                    result_count=len(result.sources),
                    feedback_rounds=feedback_rounds,
                    search_mode="agent",
                )

            return result

        except Exception as e:
            logger.error(f"Tool loop error: {e}", exc_info=True)
            err_result = self._build_response(
                success=False,
                answer="",
                sources=[],
                tool_calls=tool_calls,
                confidence=0.0,
                tokens_used=self._total_tokens_used,
                start_time=start_time,
                error=str(e),
            )

            if self._diagnostics and self._diagnostics.is_active:
                llm_stats = self._diagnostics.get_llm_stats()
                err_result.step_timings = self._diagnostics.get_step_timings()
                err_result.llm_call_count = llm_stats["call_count"]
                err_result.llm_latency_total = llm_stats["latency_total_ms"] / 1000
                err_result.retry_count = llm_stats["retry_count"]
                err_result.cache_hits = llm_stats["cache_hits"]
                self._diagnostics.finish(
                    success=False,
                    error_type=type(e).__name__,
                    model=getattr(self._llm_client, 'model', ''),
                    provider=os.getenv('LLM_PROVIDER', 'glm'),
                    search_count=sum(1 for tc in tool_calls if tc.get("tool") == "search"),
                    read_count=sum(1 for tc in tool_calls if tc.get("tool") == "read"),
                    search_mode="agent",
                )

            return err_result

    def _build_tool_dict(self) -> Dict[str, Any]:
        """Build a {name: tool_instance} dict from registered tools.

        Used by verification recovery to directly invoke search/read
        outside the LLM tool_loop.
        """
        return {t.name: t for t in self.tools}

    def _extract_sources_from_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract source paths from tool call records.

        Args:
            tool_calls: List of tool call records from the loop

        Returns:
            List of unique source paths / doc_ids used
        """
        sources: Set[str] = set()
        for tc in tool_calls:
            if tc.get("tool") == "read":
                args = tc.get("arguments", {})
                if "source_path" in args:
                    sources.add(args["source_path"])
                if "doc_id" in args:
                    sources.add(args["doc_id"])
        return list(sources)

    def _extract_search_hits_from_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract structured search hit details from tool call results.

        Parses search tool results to recover doc_id, title, score,
        snippet, and source_path for display in the answer_complete event.

        Args:
            tool_calls: List of tool call records from the loop

        Returns:
            List of search hit dicts with keys: doc_id, title, score,
            snippet, source_path, search_type
        """
        import json as _json

        hits: List[Dict[str, Any]] = []
        seen: set = set()

        for tc in tool_calls:
            if tc.get("tool") != "search" or not tc.get("success"):
                continue

            try:
                content = tc.get("content", "")
                if not content:
                    continue
                parsed = _json.loads(content) if isinstance(content, str) else content
                if not isinstance(parsed, dict):
                    continue

                results = parsed.get("results", [])
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    doc_id = str(r.get("doc_id", ""))
                    sp = str(r.get("source_path", ""))
                    key = doc_id or sp
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    hits.append({
                        "doc_id": doc_id,
                        "title": str(r.get("title", "")),
                        "score": round(float(r.get("score", 0)), 4),
                        "snippet": str(r.get("snippet", "")),
                        "source_path": sp,
                        "search_type": "BM25",
                    })
            except Exception:
                pass

        return hits

    def _expand_query(self, query: str) -> List[str]:
        """Use LLM to generate synonym/related queries for broader search.

        Args:
            query: Original user query

        Returns:
            List containing the original query followed by up to MAX_EXPANSION_VARIANTS
            variant queries. Falls back to [query] alone if LLM call fails.
        """
        prompt = (
            '用户查询: "{query}"\n'
            "请生成5个语义相同但用词不同的查询变体，用于扩大搜索范围。\n"
            "策略:\n"
            "- 将用户口语化/简称转换为可能的官方制度全称\n"
            '- 尝试不同的关键词组合（如"A B"→"A管理办法" "A规范" "A制度" "B分类"）\n'
            "- 同时生成精确匹配和模糊泛化的版本\n"
            "每行一个，不要编号，不要解释，不要重复原始查询。"
        ).format(query=query)
        try:
            response = self._llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
                model_tier="fast",
            )
            self._total_tokens_used += response.usage.get("total_tokens", 0)
            variants = [
                line.strip()
                for line in (response.content or "").strip().split("\n")
                if line.strip()
            ]
            return [query] + variants[:self.MAX_EXPANSION_VARIANTS]  # Original + max 5 variants
        except Exception:
            logger.warning("Query expansion failed, using original query only")
            return [query]

    def _execute_sub_queries_parallel(
        self,
        sub_queries: List[SubQuery],
        tool_calls: List[Dict[str, Any]],
    ) -> str:
        """Execute sub-queries in parallel using ThreadPoolExecutor.

        For each sub-query, runs BM25 search (limit=5) then reads the top 2
        documents. Results from all sub-queries are merged into a single
        formatted context string for injection into the main conversation.

        Args:
            sub_queries: List of SubQuery from QueryDecomposer.
            tool_calls: List to record tool call entries for observability.

        Returns:
            Formatted string with merged context from all sub-queries.
            Returns empty string if execution fails entirely.
        """
        search_tool = self.get_tool("search")
        read_tool = self.get_tool("read")

        if not search_tool or not read_tool:
            logger.warning("Search/read tools not available for parallel sub-query execution")
            return ""

        logger.info(f"Sub-query parallel execution: {len(sub_queries)} sub-queries")
        max_workers = min(len(sub_queries), 5)

        # Per-sub-query result container
        sub_results: Dict[int, List[Dict[str, str]]] = {}

        def _process_single_query(idx: int, sq: SubQuery) -> None:
            """Search + read for one sub-query; stores results in sub_results."""
            docs: List[Dict[str, str]] = []
            try:
                search_result = search_tool.execute(query=sq.query, limit=5)

                if not search_result.success or not search_result.data:
                    logger.info(f"Sub-query [{sq.aspect}] search returned no results")
                    sub_results[idx] = docs
                    return

                data = json.loads(search_result.data)
                results = data.get("results", [])

                # Record search tool call for observability
                tool_calls.append({
                    "tool": "search",
                    "arguments": {"query": sq.query, "limit": 5},
                    "success": True,
                    "_sub_query": sq.aspect,
                    "result_count": len(results),
                })

                # Read top 2 results
                for r in results[:2]:
                    doc_id = r.get("doc_id", "")
                    if not doc_id:
                        continue
                    try:
                        read_result = read_tool.execute(doc_id=doc_id)
                        tool_calls.append({
                            "tool": "read",
                            "arguments": {"doc_id": doc_id},
                            "success": read_result.success,
                            "_sub_query": sq.aspect,
                        })
                        if read_result.success and read_result.data:
                            title = r.get("title", doc_id)
                            snippet = r.get("snippet", "")
                            content = str(read_result.data)[:1500]
                            docs.append({
                                "title": title,
                                "snippet": snippet,
                                "content": content,
                            })
                    except Exception as e:
                        logger.warning(f"Sub-query [{sq.aspect}] read failed for {doc_id}: {e}")

            except Exception as e:
                logger.warning(f"Sub-query [{sq.aspect}] execution failed: {e}")
            finally:
                sub_results[idx] = docs

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_single_query, i, sq): i
                for i, sq in enumerate(sub_queries)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"Sub-query {idx} raised: {e}")
                    sub_results.setdefault(idx, [])

        # Build merged context string
        sections: List[str] = []
        for i, sq in enumerate(sub_queries):
            docs = sub_results.get(i, [])
            section = f"[方面{i + 1}: {sq.aspect}]\n"
            if docs:
                section += f"搜索到 {len(docs)} 篇相关文档:\n"
                for doc in docs:
                    title = doc["title"]
                    content = doc["content"]
                    section += f"- 《{title}》: \"...{content[:300]}...\"\n"
            else:
                section += "未找到相关文档。\n"
            sections.append(section)

        merged = (
            f"🔍 查询分解: 已并行搜索 {len(sub_queries)} 个方面:\n\n"
            + "\n".join(sections)
            + "\n请基于以上已收集的信息综合回答用户的问题。如有遗漏，可继续搜索补充。"
        )
        return merged

    def _classify_query_complexity(self, query: str) -> str:
        """Classify query complexity for adaptive strategy selection.

        Args:
            query: User query string.

        Returns:
            "simple", "light", "medium", or "complex"
        """
        import re as _re
        query = query.strip()

        # Simple: very short queries (single keyword)
        if len(query) <= 4:
            return "simple"

        # ── P0-2: Multi-question detection — count numbered sub-questions ──
        # Queries with 3+ numbered sub-questions (1), 2), 3)) are capped
        # at "medium" to avoid triggering full decomposition+expansion+8-round loop
        # which can cause >30s latency for what are essentially simple sub-queries.
        sub_question_patterns = [
            r'\b\d+[\)、）]',   # 1) 2) 3) or 1、2、3、
            r'[一二三四五六七八九十]+[、）\)]',  # 一、二、三、
            r'第[一二三四五六七八九十\d]+[个点问]',  # 第一个问题、第二点
        ]
        sub_q_count = sum(
            len(_re.findall(p, query))
            for p in sub_question_patterns
        )
        # Cap: if 3+ sub-questions AND query > 80 chars, treat as medium (4 iter)
        # not complex (8 iter) — the sub-questions are individually simple
        if sub_q_count >= 3 and len(query) > 80:
            logger.info(
                f"Query capped to 'medium': {sub_q_count} sub-questions detected "
                f"(len={len(query)}). Avoiding expensive decomposition+expansion+8-round loop."
            )
            return "medium"

        # Count signal matches
        complex_count = sum(1 for s in self.COMPLEX_SIGNALS if s in query)
        medium_count = sum(1 for s in self.MEDIUM_SIGNALS if s in query)

        # Long queries with multiple sub-questions: bump but don't auto-complex
        if len(query) > 80 and sub_q_count >= 2:
            complex_count += 1
        elif len(query) > 50:
            complex_count += 1  # Reduced from +2 — single long query isn't always complex

        # Scoring — "light" tier (4 iterations) prevents most single-signal
        # Chinese queries from burning 8 iterations unnecessarily
        if complex_count >= 2 or (complex_count >= 1 and len(query) > 80):
            return "complex"
        elif medium_count >= 2:
            return "medium"
        elif medium_count >= 1 or complex_count >= 1:
            return "light"
        else:
            return "simple"

    def _check_sufficient_context(
        self,
        query: str,
        tool_calls: List[Dict[str, Any]],
    ) -> SearchFeedback:
        """Check if information collected so far is sufficient.

        Extracts snippets from search tool calls and evaluates coverage.

        Args:
            query: Original user query.
            tool_calls: List of tool call records.

        Returns:
            SearchFeedback with sufficiency assessment.
        """
        # Collect snippets from search tool results
        snippets: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if tc.get("tool") != "search" or not tc.get("success"):
                continue
            try:
                content = tc.get("content", "")
                if not content:
                    continue
                parsed = json.loads(content) if isinstance(content, str) else content
                if not isinstance(parsed, dict):
                    continue
                for r in parsed.get("results", []):
                    if isinstance(r, dict):
                        snippets.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("snippet", ""),
                            "source": r.get("source_path", ""),
                        })
            except (json.JSONDecodeError, AttributeError):
                logger.debug("Failed to collect search result snippets for verification")

        # Also collect from read results (document content snippets)
        # Use longer truncation for read content since it's full document text
        READ_SNIPPET_LENGTH = 1500
        for tc in tool_calls:
            if tc.get("tool") != "read" or not tc.get("success"):
                continue
            content = tc.get("content", "")
            if content:
                source = tc.get("arguments", {}).get("doc_id", "") or tc.get("arguments", {}).get("source_path", "unknown")
                snippets.append({
                    "title": f"文档内容 ({source})",
                    "snippet": str(content)[:READ_SNIPPET_LENGTH],
                    "source": source,
                })

        checker = SufficientContextChecker(self._llm_client)
        return checker.check(query=query, collected_snippets=snippets)

    def _inject_feedback(
        self,
        messages: List[Any],
        feedback: SearchFeedback,
    ) -> None:
        """Inject structured feedback into messages to drive additional search.

        Args:
            messages: Chat messages list (modified in-place).
            feedback: SearchFeedback from sufficiency check.
        """
        feedback_prompt = (
            "⚠️ 搜索充足性检查结果:\n"
            f"- ✅ 已覆盖: {', '.join(feedback.covered_aspects) if feedback.covered_aspects else '无'}\n"
            f"- ❌ 缺失: {', '.join(feedback.missing_aspects) if feedback.missing_aspects else '无'}\n"
            f"- 建议搜索: {', '.join(feedback.suggested_queries) if feedback.suggested_queries else '无'}\n"
            "\n请继续搜索缺失的信息。"
            f"{'建议使用工具: ' + ', '.join(feedback.suggested_tools) + '。' if feedback.suggested_tools else ''}"
            "\n不要直接给出不完整的答案。"
        )
        messages.append(ChatMessage(role="system", content=feedback_prompt))

    def _verify_draft_grounding(self, query: str, answer: str, tool_calls: List[Dict[str, Any]]) -> SearchFeedback:
        """Verify that the draft answer is grounded in collected documents.

        Generates a quick draft verification: checks which claims have document
        support and which are unverified, returning suggestions for additional
        search if needed.

        Args:
            query: Original user query.
            answer: Draft answer from the agent.
            tool_calls: Tool call records with collected documents.

        Returns:
            SearchFeedback indicating whether the draft is well-grounded.
        """
        # Collect available document content from read calls only
        # (filter out metadata entries like _sufficiency_check, _draft_verification)
        doc_contents: List[str] = []
        for tc in tool_calls:
            if tc.get("tool") != "read" or not tc.get("success"):
                continue
            content = tc.get("content", "")
            if content:
                source = tc.get("arguments", {}).get("doc_id", "unknown")
                # Use full content (up to 2000 chars) for better verification
                doc_contents.append(f"[{source}]\n{str(content)[:2000]}")

        if not doc_contents:
            # No documents read — can't verify grounding
            return SearchFeedback(
                sufficient=False,
                coverage_score=0.0,
                missing_aspects=["未读取任何文档内容，无法验证回答"],
                suggested_queries=[query],
                suggested_tools=["search", "read"],
                reason="草稿验证失败: 未读取文档",
            )

        docs_text = "\n\n".join(doc_contents[:5])[:6000]

        verify_prompt = (
            f"验证回答是否有文档支撑。\n\n"
            f"原始查询: {query}\n\n"
            f"草稿回答:\n{answer[:1000]}\n\n"
            f"可用文档:\n{docs_text}\n\n"
            f"请检查草稿回答中的每个关键声明是否有文档原文支撑。\n"
            f"输出JSON:\n"
            f'{{"sufficient": true或false, "coverage_score": 0.0-1.0, '
            f'"covered_aspects": ["有支撑的"], "missing_aspects": ["缺乏支撑的"], '
            f'"suggested_queries": ["需要补充搜索的"], "reason": "说明"}}'
        )

        try:
            verify_response = self._llm_client.chat(
                messages=[ChatMessage(role="user", content=verify_prompt)],
                temperature=0.1,
                max_tokens=400,
                model_tier="fast",
            )
            self._total_tokens_used += verify_response.usage.get("total_tokens", 0)

            # Reuse SufficientContextChecker's parser
            checker = SufficientContextChecker(self._llm_client)
            return checker._parse_response(verify_response.content)
        except Exception as e:
            logger.warning(f"Draft verification failed: {e}")
            return SearchFeedback(
                sufficient=True,
                coverage_score=0.5,
                reason=f"草稿验证失败，默认通过: {e}",
            )

    def _regenerate_answer_from_docs(
        self,
        query: str,
        tool_calls: List[Dict[str, Any]],
    ) -> Optional[ChatResponse]:
        """Regenerate answer from collected document content.

        Used when the LLM's final response appears to be an intermediate
        step (tool-call text, too short, etc.) rather than a proper answer.

        Args:
            query: Original user query.
            tool_calls: Tool call records with collected documents.

        Returns:
            ChatResponse with regenerated answer, or None if no documents available.
        """
        # Collect document contents from read calls only
        doc_contents: List[str] = []
        for tc in tool_calls:
            if tc.get("tool") != "read" or not tc.get("success"):
                continue
            content = tc.get("content", "")
            if content:
                source = tc.get("arguments", {}).get("doc_id", "unknown")
                doc_contents.append(f"[文档: {source}]\n{str(content)[:3000]}")

        if not doc_contents:
            # Return clean fallback instead of None — caller may need a valid ChatResponse
            return ChatResponse(
                content="无法找到相关信息。",
                usage={"total_tokens": 0},
            )

        context = "\n\n".join(doc_contents[:5])[:8000]
        forced_messages = [
            ChatMessage(
                role="system",
                content=(
                    "你是文档搜索助手。基于以下文档内容直接回答用户问题。\n"
                    "必须引用文档原文，标注来源文档名。回答简洁，不超过500字。"
                ),
            ),
            ChatMessage(role="user", content=f"问题: {query}\n\n参考文档:\n{context}"),
        ]

        try:
            forced_response = self._llm_client.chat(
                messages=forced_messages,
                temperature=0.3,
                max_tokens=2000,
            )
            self._total_tokens_used += forced_response.usage.get("total_tokens", 0)
            return forced_response
        except Exception as e:
            logger.warning(f"Answer regeneration failed: {e}")
            return None

    def _llm_self_assess_confidence(self, query: str, answer: str) -> float:
        """P6: Ask fast-tier LLM to self-assess answer confidence (0.0-1.0).

        Returns neutral 0.5 on any failure.
        """
        if not answer or not query or not self._llm_client:
            return 0.5
        try:
            response = self._llm_client.chat(
                messages=[
                    {"role": "system", "content": "评估你对回答的置信度（0.0-1.0）。只输出一个数字。"},
                    {"role": "user", "content": f"问题：{query}\n回答：{answer[:500]}\n置信度："},
                ],
                model_tier="default",  # GLM-4 — DeepSeek Flash 推理模型消耗 token, max_tokens=10 会返回空
                max_tokens=10,
                temperature=0.0,
            )
            llm_confidence = float(response.content.strip())
            return max(0.0, min(1.0, llm_confidence))
        except Exception:
            return 0.5

    def _calculate_tool_loop_confidence(
        self, tool_calls: List[Dict[str, Any]], query: str = "", answer: str = ""
    ) -> float:
        """Calculate confidence based on tool call results.

        Args:
            tool_calls: List of tool call records from the loop
            query: Optional query for P6 LLM self-assessment calibration
            answer: Optional answer text for P6 LLM self-assessment calibration

        Returns:
            Confidence score between 0.0 and 0.95
        """
        if not tool_calls:
            return 0.3  # No tools used = low confidence

        # Check if search tools found results
        search_calls = [
            tc for tc in tool_calls if tc.get("tool") in ("search", "grep")
        ]
        read_calls = [tc for tc in tool_calls if tc.get("tool") == "read"]
        if not search_calls:
            return 0.4  # No search = guessing

        # Base confidence from search + read pattern
        confidence = 0.5
        if read_calls:
            confidence += 0.2  # Agent read documents = higher confidence
        if len(search_calls) >= 2:
            confidence += 0.1  # Multiple searches = more thorough
        if any(tc.get("tool") == "rerank" for tc in tool_calls):
            confidence += 0.1  # Rerank used = higher quality

        # P6: 混合 LLM 自评置信度 (启发式 60% + LLM 40%)
        # P1-1: Skip for light/medium — LLM self-assessment adds 1s latency
        # with minimal calibration value for non-complex queries
        if query and answer and getattr(self, "_complexity", "") == "complex":
            try:
                llm_confidence = self._llm_self_assess_confidence(query, answer)
                confidence = confidence * 0.6 + llm_confidence * 0.4
                if self._diagnostics:
                    try:
                        self._diagnostics.record_step("confidence_calibration", 0.0)
                    except Exception:
                        logger.debug("Failed to record confidence_calibration diagnostics")
            except Exception:
                logger.debug("LLM self-assessment failed, using heuristic confidence")

        return min(0.95, confidence)

    # ─────────────────────────────────────────────────────
    # Pipeline Mode (legacy, exact copy of original run())
    # ─────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Execute the hardcoded 6-step pipeline (legacy mode).

        Steps:
        1. _analyze_query() → action, search_query
        2. If "direct" → _direct_response() → return
        3. _execute_search() → BM25 search → optional rerank
        4. _read_top_documents() → ReadTool for top N
        5. Token budget check
        6. _generate_answer() → LLM answer

        Args:
            query: The user's question or search query
            context: Optional context (e.g., {"mode": "precision"})

        Returns:
            AgentResponse containing answer and metadata
        """
        start_time = time.time()
        tool_calls: List[Dict[str, Any]] = []

        # Reset per-query token budget
        self._total_tokens_used = 0

        try:
            # Step 1: Analyze query to determine action
            action, search_query = self._analyze_query(query)
            logger.info(f"Query analysis: action={action}, search_query={search_query}")

            # Step 2: Handle direct responses (greetings, etc.)
            if action == "direct":
                response = self._direct_response(query)
                return self._build_response(
                    success=True,
                    answer=response,
                    sources=[],
                    tool_calls=tool_calls,
                    confidence=1.0,
                    tokens_used=self._total_tokens_used,
                    start_time=start_time,
                )

            # Step 3: Execute search
            search_results = self._execute_search(
                query=search_query,
                limit=self._max_search_results,
                tool_calls=tool_calls,
            )

            if not search_results:
                return self._build_response(
                    success=True,
                    answer="未找到相关文档。请尝试使用不同的关键词搜索。",
                    sources=[],
                    tool_calls=tool_calls,
                    confidence=0.0,
                    tokens_used=self._total_tokens_used,
                    start_time=start_time,
                )

            # Step 4: Read top documents
            context_docs = self._read_top_documents(
                search_results=search_results,
                max_docs=self._max_read_docs,
                tool_calls=tool_calls,
            )

            # Step 5: Check token budget before expensive LLM call
            if self._total_tokens_used >= self._max_session_tokens:
                return self._build_response(
                    success=False,
                    answer="Token预算已用完，无法生成回答。",
                    sources=[],
                    tool_calls=tool_calls,
                    confidence=0.0,
                    tokens_used=self._total_tokens_used,
                    start_time=start_time,
                    error="Token budget exceeded",
                )

            # Step 6: Generate answer with context
            answer, confidence, tokens = self._generate_answer(
                query=query,
                search_results=search_results,
                context_docs=context_docs,
                mode=context.get("mode", "semantic") if context else "semantic",
            )
            self._total_tokens_used += tokens

            # Collect sources
            sources = self._collect_sources(search_results, context_docs)

            return self._build_response(
                success=True,
                answer=answer,
                sources=sources,
                tool_calls=tool_calls,
                confidence=confidence,
                tokens_used=self._total_tokens_used,
                start_time=start_time,
            )

        except Exception as e:
            logger.error(f"SearchAgent error: {e}", exc_info=True)
            return self._build_response(
                success=False,
                answer="",
                sources=[],
                tool_calls=tool_calls,
                confidence=0.0,
                tokens_used=self._total_tokens_used,
                start_time=start_time,
                error=str(e),
            )

    # ─────────────────────────────────────────────────────
    # Shared helper methods (used by both modes)
    # ─────────────────────────────────────────────────────

    def _analyze_query(self, query: str) -> tuple[str, str]:
        """Analyze query to determine action and extract search terms.

        Used by pipeline mode only.

        Args:
            query: The user query

        Returns:
            Tuple of (action, search_query)
            - action: "search" or "direct"
            - search_query: Optimized search query
        """
        # 1. Quick check for simple greetings
        query_lower = query.strip().lower()
        greetings = ["你好", "hello", "hi", "嗨", "您好", "早上好", "下午好"]
        if any(query_lower.startswith(g) for g in greetings) and len(query) < 20:
            return "direct", query

        # 2. Rule-based analysis for obvious search queries
        search_indicators = [
            "如何", "怎么", "什么", "哪些", "多少", "是否",
            "制度", "流程", "规定", "政策", "标准", "办法",
            "年假", "请假", "加班", "报销", "出差",
            "采购", "合同", "审批", "绩效", "考核",
            "工资", "薪资", "福利", "培训",
        ]
        if any(kw in query for kw in search_indicators):
            search_query = query.rstrip("？?!！。").strip()
            logger.info("Rule-based routing: query matched search indicators, skipping LLM")
            return "search", search_query

        # 3. For ambiguous queries, use LLM analysis
        try:
            messages = [
                ChatMessage(role="system", content="你是查询分析助手，输出JSON。"),
                ChatMessage(
                    role="user", content=QUERY_ANALYSIS_PROMPT.format(query=query)
                ),
            ]

            response = self._llm_client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=150,
                model_tier="fast",
            )

            # Track tokens from LLM call
            self._total_tokens_used += response.usage.get("total_tokens", 0)

            # Parse response
            content = response.content.strip()
            # Extract JSON from markdown code block if present
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            action = result.get("action", "search")
            search_query = result.get("search_query", query)

            return action, search_query

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Query analysis failed: {e}, defaulting to search")
            return "search", query

    def _direct_response(self, query: str) -> str:
        """Generate a direct response for non-search queries.

        Args:
            query: The user query

        Returns:
            Direct response string
        """
        query_lower = query.strip().lower()

        if any(g in query_lower for g in ["你好", "hello", "hi", "嗨", "您好"]):
            return (
                "你好！我是文档搜索助手。请告诉我你想了解什么，我会帮你搜索相关文档。"
            )
        elif any(g in query_lower for g in ["早上好", "下午好", "晚上好"]):
            return "你好！有什么我可以帮你的吗？你可以问我关于文档的问题。"
        else:
            return "我理解你的问题，但我主要是帮助搜索和回答文档相关的问题。请尝试提问关于文档内容的问题。"

    def _execute_search(
        self,
        query: str,
        limit: int,
        tool_calls: List[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Execute search and return formatted results.

        When use_rerank is enabled, retrieves extra BM25 candidates and
        reranks them via the ZhipuAI Rerank API for better precision.

        Args:
            query: Search query
            limit: Maximum results
            tool_calls: List to record tool call

        Returns:
            List of SearchResult objects
        """
        search_tool = self.get_tool("search")
        if not search_tool:
            logger.error("SearchTool not registered")
            return []

        # Fetch more candidates if reranking is enabled (cap at 20 for API efficiency)
        search_limit = min(limit * 3, 20) if self._use_rerank else limit
        result = search_tool.execute(query=query, limit=search_limit)

        # Record tool call
        tool_calls.append(
            {
                "tool": "search",
                "arguments": {"query": query, "limit": search_limit},
                "success": result.success,
                "result_count": result.metadata.get("total_results", 0)
                if result.metadata
                else 0,
            }
        )

        if not result.success or not result.data:
            return []

        # Parse results
        try:
            data = json.loads(result.data)
            results = []
            for item in data.get("results", []):
                results.append(
                    SearchResult(
                        doc_id=item.get("doc_id", ""),
                        title=item.get("title", ""),
                        score=item.get("score", 0.0),
                        snippet=item.get("snippet", ""),
                        source_path=item.get("source_path"),
                    )
                )

            # Apply cloud reranking if enabled
            if self._use_rerank and results:
                results = self._rerank_results(query, results, limit, tool_calls)

            return results[:limit]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse search results: {e}")
            return []

    def _get_reranker(self):
        """Lazy-initialize the ZhipuAI reranker."""
        if self._reranker is None:
            from src.search.reranker import ZhipuAIReranker
            self._reranker = ZhipuAIReranker(api_key=self._config.glm_api_key)
        return self._reranker

    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int,
        tool_calls: List[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Rerank search results using ZhipuAI Rerank API.

        Args:
            query: Original search query
            results: BM25 search results
            top_n: Number of top results to keep
            tool_calls: List to record tool call

        Returns:
            Reranked list of SearchResult objects
        """
        reranker = self._get_reranker()
        if reranker is None or not reranker.available:
            return results

        # Build document texts for reranking
        documents = []
        for r in results:
            # Combine title and snippet for richer context
            text = f"{r.title}\n{r.snippet}" if r.title else r.snippet
            documents.append(text)

        try:
            reranked = reranker.rerank(query=query, documents=documents, top_n=top_n)
            rerank_tokens = reranker.tokens_used

            # Track token usage
            self._total_tokens_used += rerank_tokens

            # Reorder results based on reranker output
            reordered = []
            for rr in reranked:
                if 0 <= rr.index < len(results):
                    result = results[rr.index]
                    # Update score to reranker's relevance score
                    result.score = rr.relevance_score
                    reordered.append(result)

            tool_calls.append({
                "tool": "rerank",
                "arguments": {"query": query, "num_documents": len(documents), "top_n": top_n},
                "success": True,
                "result_count": len(reordered),
                "tokens_used": rerank_tokens,
            })

            logger.info(f"Reranked {len(documents)} candidates → {len(reordered)} results")
            return reordered

        except Exception as e:
            logger.warning(f"Reranking failed, using BM25 order: {e}")
            tool_calls.append({
                "tool": "rerank",
                "arguments": {"query": query},
                "success": False,
                "error": str(e),
            })
            return results

    def _read_top_documents(
        self,
        search_results: List[SearchResult],
        max_docs: int,
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Read top documents and return their content.

        Args:
            search_results: List of search results
            max_docs: Maximum documents to read
            tool_calls: List to record tool calls

        Returns:
            Dict mapping doc_id to content
        """
        read_tool = self.get_tool("read")
        if not read_tool:
            logger.error("ReadTool not registered")
            return {}

        contents: Dict[str, str] = {}
        total_lines = 0
        max_total_lines = 1000  # Limit total context size

        for result in search_results[:max_docs]:
            if total_lines >= max_total_lines:
                break

            # Calculate max lines for this document
            remaining_lines = max_total_lines - total_lines
            max_lines = min(300, remaining_lines)

            # Execute read
            read_result = read_tool.execute(
                doc_id=result.doc_id,
                max_lines=max_lines,
            )

            # Record tool call
            tool_calls.append(
                {
                    "tool": "read",
                    "arguments": {"doc_id": result.doc_id, "max_lines": max_lines},
                    "success": read_result.success,
                    "lines_read": read_result.metadata.get("lines_read", 0)
                    if read_result.metadata
                    else 0,
                }
            )

            if read_result.success and read_result.data:
                contents[result.doc_id] = read_result.data
                total_lines += read_result.metadata.get("lines_read", 0)
                # Also update content in search result
                result.content = read_result.data

        return contents

    def _generate_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        context_docs: Dict[str, str],
        mode: str = "semantic",
    ) -> tuple[str, float, int]:
        """Generate answer using LLM with context.

        Args:
            query: User query
            search_results: Search results with snippets
            context_docs: Dict of doc_id to full content
            mode: "semantic" or "precision"

        Returns:
            Tuple of (answer, confidence, tokens_used)
        """
        # Build context
        context_parts = []
        for i, result in enumerate(search_results[:5]):
            # Use snippet or full content if available
            content = context_docs.get(result.doc_id, result.snippet)
            if content:
                context_parts.append(f"[文档{i + 1}] {result.title}\n{content}")

        context = "\n\n".join(context_parts)

        # Build prompt based on mode
        if mode == "precision":
            prompt = f"""基于以下文档内容，精确查找与查询相关的信息。

文档:
{context}

查询: {query}

要求:
1. 只回答文档中明确提到的内容
2. 标注信息来源 [文档N]
3. 如无相关内容，说明"未找到"
4. 回答简洁准确"""
        else:
            prompt = f"""基于以下文档内容回答问题。

文档:
{context}

问题: {query}

要求:
1. 基于文档内容回答
2. 标注来源 [文档N]
3. 回答简洁，不超过500字
4. 评估答案可信度(0-1)"""

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        response = self._llm_client.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )

        answer = response.content.strip()
        tokens_used = response.usage.get("total_tokens", 0)

        # Calculate confidence based on:
        # 1. Number of relevant documents found
        # 2. Search scores
        # 3. Content match quality
        if search_results:
            avg_score = sum(r.score for r in search_results[:3]) / min(
                len(search_results), 3
            )
            confidence = min(1.0, avg_score / 3.0)  # Normalize to 0-1
        else:
            confidence = 0.0

        return answer, confidence, tokens_used

    def _collect_sources(
        self,
        search_results: List[SearchResult],
        context_docs: Dict[str, str],
    ) -> List[str]:
        """Collect unique source paths from used documents.

        Args:
            search_results: List of search results
            context_docs: Dict of doc_id to content (indicates usage)

        Returns:
            List of unique source paths
        """
        sources: Set[str] = set()

        for result in search_results:
            # Only include if document was actually read
            if result.doc_id in context_docs and result.source_path:
                sources.add(result.source_path)

        return list(sources)

    def _build_response(
        self,
        success: bool,
        answer: str,
        sources: List[str],
        tool_calls: List[Dict[str, Any]],
        confidence: float,
        tokens_used: int,
        start_time: float,
        error: Optional[str] = None,
        search_hits: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResponse:
        """Build AgentResponse from components.

        Args:
            success: Whether the operation succeeded
            answer: The answer text
            sources: List of source paths
            tool_calls: Record of tool calls made
            confidence: Confidence score (0-1)
            tokens_used: Total tokens consumed
            start_time: Start time for timing calculation
            error: Optional error message
            search_hits: Structured search hit details for display

        Returns:
            AgentResponse object
        """
        processing_time = time.time() - start_time

        clean_answer = answer or ""
        import re
        _DSML = chr(0xFF5C) * 2 + "DSML" + chr(0xFF5C) * 2
        tool_call_patterns = [
            r'<tool_call>.*?</tool_call>',
            f'<{_DSML}.*',
            f'{_DSML}.*',
        ]
        stripped = False
        for pattern in tool_call_patterns:
            if re.search(pattern, clean_answer, flags=re.DOTALL):
                clean_answer = re.sub(pattern, '', clean_answer, flags=re.DOTALL).strip()
                stripped = True
        if stripped:
            if not clean_answer:
                clean_answer = "无法找到相关信息。"
            logger.warning(f"Stripped tool_call markup from answer, cleaned to {len(clean_answer)} chars")

        return AgentResponse(
            success=success,
            answer=clean_answer,
            sources=sources,
            search_hits=search_hits or [],
            tool_calls=tool_calls,
            reasoning=f"Confidence: {confidence:.2f}",
            confidence=confidence,
            tokens_used=tokens_used,
            processing_time=processing_time,
            error=error,
        )


def _discover_index_tags(raw_dir: Path, sample_size: int = 50) -> List[str]:
    """Discover common tags from .md.json files for QueryRouter IndexMeta.

    Scans a sample of .md.json files in the raw directory, collects unique
    tags, and returns the most common ones. This improves multi-index routing
    accuracy without requiring manual index metadata configuration.

    Args:
        raw_dir: Raw directory containing .md and .md.json files.
        sample_size: Max number of .md.json files to scan (performance bound).

    Returns:
        List of unique tags discovered (max 20).
    """
    import json
    from collections import Counter

    if not raw_dir or not raw_dir.is_dir():
        return []

    tag_counter: Counter = Counter()
    scanned = 0

    try:
        for mdjson_path in raw_dir.rglob("*.md.json"):
            if scanned >= sample_size:
                break
            try:
                with open(mdjson_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for tag in meta.get("tags", []):
                    if isinstance(tag, str) and tag.strip():
                        tag_counter[tag.strip()] += 1
                scanned += 1
            except (json.JSONDecodeError, OSError, KeyError):
                continue
    except OSError:
        return []

    # Return top-20 tags by frequency
    return [tag for tag, _ in tag_counter.most_common(20)]


def create_search_agent(
    config: Config,
    index_path: Optional[Path] = None,
    input_base: Optional[Path] = None,
    output_base: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    use_rerank: bool = False,
    mode: str = "tool_loop",
    usage_tracker=None,
    diagnostics=None,
    budget_guard=None,
) -> SearchAgent:
    """Factory function to create a fully configured SearchAgent.

    Args:
        config: Configuration object
        index_path: Path to search index (optional)
        input_base: Base directory for source documents (optional)
        output_base: Base directory for output Markdown files (optional)
        raw_dir: Directory with raw markdown files for GrepTool (optional)
        use_rerank: Whether to enable ZhipuAI Rerank API in pipeline mode (default: False)
        mode: Execution mode - "tool_loop" (default) or "pipeline"
        usage_tracker: Optional UsageTracker for recording API usage

    Returns:
        Configured SearchAgent instance
    """
    from src.search.bm25_search import create_searcher
    from src.storage.markdown_store import MarkdownStore

    # Support multi-index via comma-separated paths
    index_paths = [Path(p.strip()) for p in str(index_path).split(",") if p.strip()] if index_path else []

    if len(index_paths) > 1:
        from src.search.multi_index import MultiIndexSearcher
        from src.search.query_router import QueryRouter, IndexMeta

        # Auto-build QueryRouter from index paths with tag discovery
        indexes_meta: Dict[str, IndexMeta] = {}
        for p in index_paths:
            idx_name = p.parent.name if p.name == "index" else p.name
            # Discover tags from .md.json files in the raw directory
            tags = _discover_index_tags(p.parent)
            indexes_meta[idx_name] = IndexMeta(
                path=str(p),
                name=idx_name,
                source_dir=idx_name,
                tags=tags,
            )
        router = QueryRouter(indexes=indexes_meta)
        searcher = MultiIndexSearcher(index_paths=index_paths, query_router=router)
    else:
        searcher = create_searcher(
            index_path=index_paths[0] if index_paths else None,
            use_jieba=True,
            readonly=True,
        )

    # Create search tool with TTL cache for repeated queries
    search_tool = SearchTool(searcher)
    from src.agent.tool_types import ToolCache
    search_tool.set_cache(ToolCache(ttl=300, max_size=128))

    # Create read tool — support multiple raw dirs for source_path resolution
    input_dir = input_base or Path(".")
    output_dir = output_base or Path(".")
    raw_dirs = []
    if raw_dir:
        raw_dirs = [d.strip() for d in str(raw_dir).split(",") if d.strip()]
        if raw_dirs:
            output_dir = Path(raw_dirs[0])  # primary raw dir
    markdown_store = MarkdownStore(input_base=input_dir, output_base=output_dir)
    read_tool = ReadTool(markdown_store, raw_dirs=raw_dirs, searcher=searcher)

    # Create LLM client with optional usage tracker
    llm_client = LLMClient(
        config,
        usage_tracker=usage_tracker,
        diagnostics=diagnostics,
        budget_guard=budget_guard,
    )

    # Create agent
    agent = SearchAgent(
        config=config,
        search_tool=search_tool,
        read_tool=read_tool,
        llm_client=llm_client,
        use_rerank=use_rerank,
        mode=mode,
        usage_tracker=usage_tracker,
        diagnostics=diagnostics,
    )
    agent._index_path = str(index_path) if index_path else ""

    # Register GrepTool + BashTool for raw file search (DCI paradigm)
    if raw_dir and Path(raw_dir).is_dir():
        from src.agent.tools.grep import GrepTool
        from src.agent.tools.bash import BashTool
        grep_tool = GrepTool(raw_dir=Path(raw_dir))
        grep_tool.set_cache(ToolCache(ttl=300, max_size=128))
        bash_tool = BashTool(raw_dir=Path(raw_dir))
        agent.register_tool(grep_tool)
        agent.register_tool(bash_tool)
        logger.info(f"Registered GrepTool + BashTool for raw dir: {raw_dir}")

    # Register RerankTool if reranker is available (LLM decides when to use it)
    # Supports both cloud (ZhipuAI) and local (bge-reranker) via RERANKER_TYPE env
    from src.search.local_reranker import create_reranker
    reranker = create_reranker(config=config, usage_tracker=usage_tracker)
    if reranker.available:
        from src.agent.tools.rerank import RerankTool
        rerank_tool = RerankTool(reranker)
        agent.register_tool(rerank_tool)
        is_local = os.environ.get("RERANKER_TYPE") == "local"
        logger.info(f"Registered {'LocalReranker' if is_local else 'RerankTool'}")

    # Always register SummarizeTool (LLM decides when to summarize long docs)
    from src.agent.tools.summarize import SummarizeTool
    summarize_tool = SummarizeTool(
        markdown_store=markdown_store,
        raw_dirs=raw_dirs,
        searcher=searcher,
        llm_client=llm_client,
    )
    agent.register_tool(summarize_tool)
    logger.info("Registered SummarizeTool")

    return agent
