"""MCP Server — exposes doc-search tools via Model Context Protocol.

Provides typed tools for OpenCode / Claude / other MCP clients to search the
document knowledge base without manual curl/CLI commands.

Tools:
    - doc_search:   BM25 / Hybrid / Grep keyword search
    - doc_agent:    Agentic RAG with LLM-powered answer generation
    - doc_read:     Read full document content by doc_id
    - doc_analyze:  Document deep analysis (compare/extract/summarize)

Usage::

    # Install mcp package first:
    #   pip install -e ".[mcp]"

    # Run as standalone (stdio transport):
    python -m src.mcp_server

    # Or configure in opencode.json:
    #   "mcp": { "doc_search": { "type": "local",
    #     "command": [".venv\\Scripts\\python.exe", "-m", "src.mcp_server"] }}

Environment variables:
    DOC_SEARCH_INDEX  Default index path (comma-separated for multi-index)
    DOC_SEARCH_RAW    Default raw directory (for hybrid/grep/agent modes)
    DOC_SEARCH_RAW_ROOT  Root dir for auto-discovery (default: platform-specific)
    GLM_API_KEY       Required for doc_agent and doc_analyze (LLM access)

When DOC_SEARCH_INDEX is not set, the server auto-discovers all indexes under
the raw root (mirrors pi-doc.py behavior).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Lazy MCP import (package is optional) ──────────────────────
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load .env from project root so API keys are available for MCP tools.

    OpenCode launches MCP processes with only explicitly declared env vars
    (from the MCP config), not the full shell environment. Running inside an
    editor's process tree, .env files are not auto-loaded. This function
    ensures GLM_API_KEY, DEEPSEEK_API_KEY, etc. are available without
    requiring every opencode.json to enumerate all secrets.
    """
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).resolve().parent.parent
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)
    except ImportError:
        pass


# ── Auto-discover indexes (mirrors pi-doc.py discover_indexes) ─

_DEFAULT_RAW_ROOTS: Dict[str, List[str]] = {
    "win32": [r"./raw"],
    "linux": [
        str(Path.home() / "docs" / "raw"),
        str(Path(__file__).resolve().parent.parent.parent / "raw"),
        "/data/docs/raw",
    ],
    "darwin": [
        str(Path.home() / "docs" / "raw"),
        str(Path(__file__).resolve().parent.parent.parent / "raw"),
    ],
}


def _get_default_raw_roots() -> List[str]:
    """Return platform-appropriate default raw root paths."""
    return _DEFAULT_RAW_ROOTS.get(sys.platform, [
        str(Path.home() / "docs" / "raw"),
    ])


def discover_indexes(raw_root: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Auto-discover all index directories under raw_root.

    Mirrors pi-doc.py's discover_indexes(): each subdirectory containing
    both ``convert.db`` and ``index/`` is treated as a knowledge base.

    Only directories that have BOTH ``convert.db`` AND ``index/`` are included,
    ensuring the returned lists are always 1:1 aligned (same length, same order).

    Returns:
        (index_paths, raw_dirs) — parallel lists, index_paths[i] is the
        index directory for raw_dirs[i]. Comma-joinable.
    """
    indexes: List[str] = []
    raw_dirs: List[str] = []

    roots = [raw_root] if raw_root else _get_default_raw_roots()
    existing = next((r for r in roots if Path(r).exists()), None)
    if not existing:
        logger.warning("No raw root found. Tried: %s", ", ".join(roots))
        return indexes, raw_dirs

    root_path = Path(existing)
    seen: set[str] = set()

    def _add_candidate(d: Path) -> None:
        """Add d to both lists only if it has a usable index/ directory."""
        key = str(d.resolve())
        if key in seen:
            return
        seen.add(key)
        idx = d / "index"
        if idx.exists() and idx.is_dir():
            raw_dirs.append(str(d))
            indexes.append(str(idx))

    for d in sorted(root_path.iterdir()):
        if not d.is_dir():
            continue
        # Accept directories that have either convert.db (pipeline output)
        # or index/ (directly built index without batch-convert)
        if (d / "convert.db").exists() or (d / "index").is_dir():
            _add_candidate(d)
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and ((sub / "convert.db").exists() or (sub / "index").is_dir()):
                _add_candidate(sub)

    return indexes, raw_dirs


def _resolve_defaults() -> Tuple[str, str]:
    """Resolve default index_path and raw_dir.

    Priority: env var > auto-discover.
    Logs discovered indexes at startup.
    """
    idx = os.environ.get("DOC_SEARCH_INDEX", "")
    raw = os.environ.get("DOC_SEARCH_RAW", "")

    if not idx:
        raw_root = os.environ.get("DOC_SEARCH_RAW_ROOT", "") or None
        idx_list, raw_list = discover_indexes(raw_root)
        if idx_list:
            idx = ",".join(idx_list)
            logger.info("Auto-discovered %d indexes: %s", len(idx_list), idx)
        if not raw and raw_list:
            raw = ",".join(raw_list)

    return idx, raw


# ── Defaults resolved at import (env > auto-discover) ──────────
DEFAULT_INDEX, DEFAULT_RAW = _resolve_defaults()

# ── P1-1: Startup smoke test — verify each auto-discovered index ──
_SMOKE_TEST_QUERIES = ["制度", "管理", "规定", "标准", "流程", "指南"]


def _smoke_test_indexes() -> None:
    """Quick BM25 search against each discovered index.

    Logs a warning if any index returns zero results for ALL smoke-test queries,
    indicating the index may be empty or contain no Chinese documents.
    """
    if not DEFAULT_INDEX:
        return
    idx_list = [i.strip() for i in DEFAULT_INDEX.split(",") if i.strip()]
    for idx_path in idx_list:
        try:
            searcher = _get_searcher(idx_path)
            for q in _SMOKE_TEST_QUERIES:
                results = searcher.search(q, limit=1)
                items = getattr(results, "results", []) or []
                if items:
                    break  # At least one query found docs — index is healthy
            else:
                # All smoke-test queries returned empty
                logger.warning(
                    "SMOKE TEST FAILED for index '%s': all %d test queries "
                    "returned zero results. Index may be empty or not contain "
                    "Chinese documents. Queries tried: %s",
                    idx_path, len(_SMOKE_TEST_QUERIES),
                    ", ".join(_SMOKE_TEST_QUERIES),
                )
        except Exception as e:
            logger.warning(
                "SMOKE TEST ERROR for index '%s': %s", idx_path, e
            )


# ── Cache: avoid cold restart of searcher/agent per index_path ─
_bm25_cache: Dict[str, Any] = {}     # index_path -> BM25Searcher
_hybrid_cache: Dict[str, Any] = {}   # index_path -> HybridSearcher
_agent_cache: Dict[str, Any] = {}    # index_path -> SearchAgent
_config_cache: Optional[Any] = None
_doc_content_cache: Dict[str, str] = {}  # f"{idx}::{doc_id}" -> stripped content[:3000]
_DOC_CACHE_MAX = 256
_agent_memory: Any = None                   # Max cached documents (LRU eviction by size)


def _get_config():
    """Load Config.from_env() (cached)."""
    global _config_cache
    if _config_cache is None:
        from src.utils.config import Config
        _config_cache = Config.from_env()
    return _config_cache


def _resolve_index(index_path: Optional[str]) -> str:
    """Resolve index path: parameter > env var > error."""
    idx = index_path or DEFAULT_INDEX
    if not idx:
        raise ValueError(
            "No index_path provided. Set DOC_SEARCH_INDEX env var or pass index_path parameter."
        )
    return idx


def _resolve_raw(raw_dir: Optional[str]) -> str:
    """Resolve raw directory: parameter > env var > derive from index."""
    raw = raw_dir or DEFAULT_RAW
    if not raw:
        raise ValueError(
            "No raw_dir provided. Set DOC_SEARCH_RAW env var or pass raw_dir parameter."
        )
    return raw


def _grep_fallback(query: str, raw_dir: str, limit: int = 10) -> str:
    """Fall back to GrepTool when BM25 index is unavailable.

    Auto-converts multi-word natural language queries to OR regex,
    mirroring the CLI _query_with_grep logic.
    """
    import re as _re

    # Auto-convert multi-word queries (same logic as CLI _query_with_grep)
    _REGEX_META = set(".+*?[](){}|^$\\<>=!")
    pattern = query
    if query.strip() and not any(c in _REGEX_META for c in query):
        words = [w.strip() for w in query.split() if w.strip()]
        if len(words) >= 2:
            pattern = "|".join(words)

    raw_path = Path(raw_dir.split(",")[0].strip())
    if not raw_path.is_dir():
        return f"Grep fallback failed: raw_dir does not exist: {raw_path}"

    from src.agent.tools.grep import GrepTool

    grep_tool = GrepTool(raw_dir=raw_path, max_results=limit)
    result = grep_tool.execute(pattern=pattern, case_sensitive=False, file_filter="*.md")

    if not result.success:
        return f"Grep fallback failed: {result.error}"

    total = result.metadata.get("total_matches", 0)
    files_searched = result.metadata.get("files_searched", 0)
    if total == 0 or result.data == "No matches found.":
        return (
            f"Grep fallback: 0 matches in {files_searched} files "
            f"(pattern: {pattern})"
        )

    # Format results similar to _format_search_results
    lines: List[str] = [f"Found {total} results via grep fallback ({files_searched} files searched)\n"]
    output_lines = (result.data or "").split("\n") if isinstance(result.data, str) else []
    for line in output_lines[:limit * 3]:  # Each match has ~3 lines (match + 2 context)
        lines.append(line)
    return "\n".join(lines)


def _get_searcher(index_path: str):
    """Get or create a readonly BM25Searcher for the given index.

    For multi-index (comma-separated), uses the first index for single-index
    BM25 search. Use _get_agent for multi-index search.
    """
    if index_path not in _bm25_cache:
        from src.search.bm25_search import create_searcher
        first_idx = index_path.split(",")[0].strip()
        _bm25_cache[index_path] = create_searcher(
            index_path=Path(first_idx),
            readonly=True,
        )
    return _bm25_cache[index_path]


def _get_all_searchers(index_path: str) -> List[Any]:
    """Get or create readonly BM25Searchers for ALL indexes (multi-index).

    For comma-separated paths, returns one searcher per index.
    """
    paths = [p.strip() for p in index_path.split(",") if p.strip()]
    searchers = []
    for idx in paths:
        if idx not in _bm25_cache:
            from src.search.bm25_search import create_searcher
            _bm25_cache[idx] = create_searcher(index_path=Path(idx), readonly=True)
        searchers.append(_bm25_cache[idx])
    return searchers


# ── Run smoke test now that _get_searcher is defined ─
_smoke_test_indexes()


def _get_agent(index_path: str, raw_dir: Optional[str] = None, use_rerank: bool = True):
    """Get or create a SearchAgent for the given index.

    Supports comma-separated multi-index paths — create_search_agent
    internally uses MultiIndexSearcher for fan-out search.
    """
    cache_key = f"{index_path}::{raw_dir or ''}::{use_rerank}"
    if cache_key not in _agent_cache:
        from src.agent.search_agent import create_search_agent
        config = _get_config()
        # Pass full comma-separated path — create_search_agent handles multi-index
        idx_param = Path(index_path) if "," not in index_path else index_path
        agent = create_search_agent(
            config=config,
            index_path=idx_param,
            raw_dir=Path(raw_dir) if raw_dir else None,
            use_rerank=use_rerank,
        )
        # Disable SearchLogger for MCP (avoid side-effect file writes)
        setattr(agent, "_no_log", True)
        _agent_cache[cache_key] = agent
    return _agent_cache[cache_key]


# ── P0-1 helpers: quick search + agent timeout wrapper ──────────

_AGENT_TIMEOUT_SECONDS = 45


def _quick_search(
    query: str,
    index_path: str,
    limit: int = 3,
) -> Optional[List[Any]]:
    """Fast BM25 pre-check — verify index has relevant documents.

    Searches ALL indexes (multi-index) and merges results.
    Returns list of result items, None on error, empty list on no results.
    """
    try:
        idx_list = [i.strip() for i in index_path.split(",") if i.strip()]
        if not idx_list:
            return None
        all_items: List[Any] = []
        for idx in idx_list:
            try:
                searcher = _get_searcher(idx)
                results = searcher.search(query, limit=limit)
                items = getattr(results, "results", []) or []
                all_items.extend(items)
            except Exception:
                continue  # Skip failed indexes
        return all_items if all_items else []
    except Exception:
        return None


async def _run_agent_with_timeout(
    agent: Any,
    query: str,
    skill: Optional[str],
    timeout_seconds: int = _AGENT_TIMEOUT_SECONDS,
) -> Any:
    """Run agent.run() with a hard timeout via asyncio.

    Uses asyncio.to_thread() to avoid blocking the event loop,
    so FastMCP's protocol layer stays responsive during the wait.
    Raises TimeoutError if the agent takes longer than timeout_seconds.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(agent.run, query=query, skill=skill),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Agent timed out after {timeout_seconds}s"
        )


async def _fast_agent_pipeline(
    query: str,
    index_path: str,
    raw_dir: Optional[str],
    skill: Optional[str],
) -> str:
    """MCP-optimized fast pipeline with query rewriting + multi-query search.

    Avoids the full tool_loop (4-8 rounds of LLM calls) which can exceed
    MCP SDK's 60s DEFAULT_REQUEST_TIMEOUT_MSEC. Instead does:
    0. Query → BM25 keywords via fast-tier LLM (~1s)
    1. Multi-query BM25 search with original + rewritten keywords (~0.5s)
    2. Read top N documents (score > 0.3 threshold) (~1s)
    3. Single LLM call to synthesize answer (~8-15s)
    Total: ~12-18s — safely within 60s MCP limit.

    P0 optimizations applied:
    - Query rewriting bridges ITER-RETGEN's semantic gap (biggest quality lift)
    - Multi-query BM25 increases recall (Microsoft AgenticRAG pattern)
    - Dynamic doc selection (score threshold, not fixed top-3)
    - Frontmatter stripping (no YAML noise in LLM context)
    - Increased per-doc context (3000 → 2500 to LLM)
    """
    t0 = time.time()
    config = _get_config()
    from src.agent.llm_client import LLMClient, ChatMessage
    import concurrent.futures

    # ── Step 0: Agent Memory recall — check if this query was answered before ──
    try:
        from src.stats.memory import AgentMemory
        global _agent_memory
        if _agent_memory is None:
            _agent_memory = AgentMemory()
        cached = _agent_memory.recall(query)
        if cached:
            answer = cached.get("answer", "")
            if answer:
                answer += (
                    f"\n\n--- Sources ---\n"
                    f"[记忆命中: 历史答案, 来源 {cached.get('session_id', 'unknown')}]\n"
                    f"_Tokens: 0 | Time: ~0s | Pipeline: memory_hit_"
                )
                logger.warning("Fast pipeline: MEMORY HIT for [%s] — returning cached answer", query[:60])
                return answer
    except Exception as e:
        logger.debug("Fast pipeline: memory check skipped (%s)", e)

    # ── Step 0: Query rewriting — extract BM25 keywords ──
    # ITER-RETGEN shows 2nd iteration fixes semantic gap (e.g., "脱敏制度"
    # won't find "数据脱敏管理办法"). A single fast-tier LLM call bridges this.
    # P0-fix: Domain-specific keywords — detects legal queries and injects
    # legal corpus terms ("民法典", "司法解释") to avoid matching noise docs.
    rewritten_queries = [query]

    # Domain detection — inject corpus-specific terms
    _LEGAL_SIGNALS = ["法定代表人", "法人", "债务", "民法典", "法释", "司法解释",
                      "查封", "执行", "判决", "债权", "担保", "合同", "婚姻",
                      "继承", "侵权", "诉讼", "仲裁", "房产"]
    _MEDICAL_SIGNALS = ["诊断", "治疗", "手术", "药物", "病", "患者",
                         "临床", "检验", "体检", "指南", "护理", "康复"]
    is_legal = any(s in query for s in _LEGAL_SIGNALS)
    is_medical = any(s in query for s in _MEDICAL_SIGNALS)
    domain_hint = ""
    if is_legal and not is_medical:
        domain_hint = "查询涉及法律问题，关键词中必须包含'民法典'或'司法解释'或'法释'或'合同法'等法律特有术语。"
    elif is_medical and not is_legal:
        domain_hint = "查询涉及医学问题，关键词中必须包含'指南'或'规范'或'标准'或'诊断'等医学特有术语。"

    try:
        temp_llm = LLMClient(config)
        kw_resp = await asyncio.wait_for(
            asyncio.to_thread(
                temp_llm.chat,
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "将用户问题转换为 2-3 个 BM25 搜索关键词，每个用空格分隔，"
                            "不要用自然语言句子。优先提取法条号、编号、文件名为关键词。\n"
                            "重要: 如果问题包含保险产品名称, 必须将完整产品名作为第一个关键词, "
                            "不要拆分产品名。\n"
                            f"{domain_hint}\n"
                            "只输出关键词，每行一个，不要其他内容。\n\n"
                            "示例:\n输入: 民法典第1064条关于夫妻共同债务怎么规定\n"
                            "输出:\n民法典 1064 夫妻共同债务\n民法典 夫妻 共同 债务 认定\n法释 2018 2号 共债共签\n\n"
                            "示例:\n输入: 和瑞长盈终出生一周可以投保吗\n"
                            "输出:\n和瑞长盈终 投保年龄\n和瑞长盈终 出生 投保"
                        ),
                    ),
                    ChatMessage(role="user", content=query),
                ],
                max_tokens=120,  # Reduced from 200 — keywords only
                temperature=0.1,
                model_tier="fast",  # Use fast tier (DeepSeek V4 Flash) for query rewriting
            ),
            timeout=8,  # Increased from 3 — DeepSeek V4 Flash needs ~1-3s for longer prompts
        )
        if kw_resp and kw_resp.content:
            extracted = [
                kw.strip() for kw in kw_resp.content.strip().split("\n")
                if kw.strip() and kw.strip() != query
            ]
            rewritten_queries.extend(extracted[:2])  # Top 2 rewritten variants
            t_kw = time.time()
            logger.warning(
                "Fast pipeline: query rewriting → %d keywords (%.1fs): %s",
                len(rewritten_queries), t_kw - t0,
                " | ".join(rewritten_queries[1:]),
            )
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Fast pipeline: query rewriting skipped (%s)", e)
        # Continue with original query only — best-effort

    # ── Step 1: Multi-query BM25 search ──
    all_results: List[Any] = []
    seen_ids: set = set()

    # P0-fix: Index domain filtering — for domain-specific queries, only search
    # relevant indexes. Legal queries → legal_kb/*, medical → medica/*.
    idx_list = [i.strip() for i in index_path.split(",") if i.strip()]
    if is_legal and not is_medical:
        idx_list = [i for i in idx_list
                    if any(k in i.lower() for k in ("legal", "law", "法律", "司法", "行政", "法", "民法"))]
        if not idx_list:
            idx_list = [i.strip() for i in index_path.split(",") if i.strip()]  # fallback
        logger.warning("Fast pipeline: legal query → filtering to %d/%d indexes", len(idx_list),
                       len(index_path.split(",")))
    elif is_medical and not is_legal:
        idx_list = [i for i in idx_list
                    if any(k in i.lower() for k in ("medica", "临床", "医学"))]
        if not idx_list:
            idx_list = [i.strip() for i in index_path.split(",") if i.strip()]

    filtered_index_path = ",".join(idx_list)

    for q in rewritten_queries:
        results = _quick_search(q, filtered_index_path, limit=5)
        if results:
            for r in results:
                doc_id = getattr(r, "doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(r)
    if not all_results:
        return (
            "索引中未找到与查询相关的文档。\n\n"
            f"查询: {query[:100]}\n"
            "建议: 使用 doc_search 确认索引内容，或用更精确的关键词重试。"
        )

    # Sort by score descending, apply threshold
    all_results.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
    SCORE_THRESHOLD = 0.3
    top_results = [r for r in all_results if getattr(r, "score", 0.0) >= SCORE_THRESHOLD]
    if not top_results:
        top_results = all_results[:3]  # Fallback: top 3 regardless of score
    top_results = top_results[:5]  # Max 5 docs

    t1 = time.time()
    logger.warning(
        "Fast pipeline: BM25 %d queries → %d unique results, %d above threshold (score>%.1f) (%.1fs)",
        len(rewritten_queries), len(all_results), len(top_results),
        SCORE_THRESHOLD, t1 - t0,
    )

    # ── Step 2: Speculative pre-read top documents ──
    def _read_doc(r):
        doc_id = getattr(r, "doc_id", "")
        title = getattr(r, "title", "")
        score = getattr(r, "score", 0.0)
        snippet = getattr(r, "snippet", "")
        source = getattr(r, "source_path", "")
        content = snippet

        if doc_id:
            # P2: Check document content cache first
            cache_key = f"{filtered_index_path}::{doc_id}"
            cached = _doc_content_cache.get(cache_key)
            if cached is not None:
                content = cached
            else:
                try:
                    f_idx_list = [i.strip() for i in filtered_index_path.split(",") if i.strip()]
                    for idx_i in f_idx_list:
                        searcher = _get_searcher(idx_i)
                        full = searcher.get_full_content(doc_id)
                        if full is not None:
                            fc = getattr(full, "full_content", "")
                            if fc:
                                # Strip frontmatter (P0 fix)
                                try:
                                    from src.converter.frontmatter import strip_frontmatter
                                    _, fc = strip_frontmatter(fc)
                                except Exception:
                                    logger.warning("Failed to strip frontmatter in doc_search")
                                content = fc[:3000]
                                # P2: Cache the result (simple LRU by size)
                                if len(_doc_content_cache) >= _DOC_CACHE_MAX:
                                    _doc_content_cache.pop(next(iter(_doc_content_cache)))
                                _doc_content_cache[cache_key] = content
                                break
                except Exception:
                    logger.warning("Failed to read doc in fast pipeline: %s", doc_id)

        return {
            "title": title,
            "score": score,
            "source": source,
            "content": content,
        }

    # Speculative parallel read (all docs simultaneously)
    top_docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(top_results), 5)) as pool:
        futures = {pool.submit(_read_doc, r): i for i, r in enumerate(top_results)}
        doc_map = {}
        for f in concurrent.futures.as_completed(futures):
            idx = futures[f]
            doc_map[idx] = f.result()
        top_docs = [doc_map[i] for i in sorted(doc_map.keys())]

    t2 = time.time()
    logger.warning("Fast pipeline: read %d docs (%.1fs)", len(top_docs), t2 - t1)

    # ── Step 3: Single LLM call to synthesize answer ──
    llm = LLMClient(config)

    # Build context from documents
    doc_context = ""
    for i, d in enumerate(top_docs):
        doc_context += (
            f"\n---\n文档{i+1}: {d['title']} (score: {d['score']:.3f})\n"
            f"{d['content'][:2500]}\n"  # Increased from 1500
        )

    skill_instruction = ""
    if skill == "summarize":
        skill_instruction = "\n输出要求: 用200字以内概括关键要点，用项目符号列出。"
    elif skill == "compare":
        skill_instruction = "\n输出要求: 以对比表格形式呈现各文档的异同点。"
    elif skill == "detailed":
        skill_instruction = "\n输出要求: 详细分析，分章节总结，引用原文关键语句。"

    messages = [
        ChatMessage(
            role="system",
            content=(
                "你是文档搜索助手。基于以下检索到的文档原文回答用户问题。\n"
                "要求：\n"
                "1. 引用具体内容时标注 [来源文档名]\n"
                "2. 如果文档中没有相关信息，明确说明\n"
                "3. 回答简洁，不超过500字\n"
                f"4. 不要凭借自身知识，所有信息必须来自文档原文{skill_instruction}"
            ),
        ),
        ChatMessage(
            role="user",
            content=f"{query}\n\n以下是检索到的文档：\n{doc_context}",
        ),
    ]

    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(llm.chat, messages=messages, max_tokens=2000, temperature=0.3,
                              model_tier="power"),  # Use power tier (DeepSeek V4 Pro) for synthesis
            timeout=30,
        )
        t3 = time.time()
        answer = resp.content or "未能生成回答。"
        tokens = resp.usage.get("total_tokens", 0)
        logger.warning(
            "Fast pipeline: LLM synthesis done (%.1fs, %d tokens, total %.1fs)",
            t3 - t2, tokens, t3 - t0,
        )

        # Format response with sources
        text = answer
        text += "\n\n--- Sources ---"
        for i, d in enumerate(top_docs[:5]):
            text += f"\n{i+1}. {d['title']} (score: {d['score']:.3f})"
            if d["source"]:
                text += f"\n   {d['source']}"
        text += f"\n\n_Tokens: {tokens:,} | Time: {t3-t0:.1f}s | Pipeline: fast_"

        # Agent Memory: 记录本次搜索策略 (异步, 不阻塞)
        try:
            if _agent_memory is not None:
                _agent_memory.learn(query, {
                    "mode": "fast_pipeline",
                    "index_path": index_path,
                    "confidence": 0.8,
                    "tool_count": len(rewritten_queries),
                    "latency": t3 - t0,
                    "search_count": len(all_results),
                    "read_count": len(top_docs),
                })
        except Exception:
            pass

        return text

    except asyncio.TimeoutError:
        t3 = time.time()
        logger.warning("Fast pipeline: LLM timed out (%.1fs), returning BM25 results", t3 - t2)
        # Fallback: return BM25 results without LLM synthesis
        result_count = len(all_results)
        text = f"LLM 综合超时，以下为 BM25 检索结果（{result_count} 条）：\n\n"
        text += _format_search_results_from_list(all_results[:10], 10)
        text += f"\n\n_Tokens: 0 | Time: {t3-t0:.1f}s | Pipeline: bm25_fallback_"
        return text


def _format_search_results_from_list(items: List[Any], limit: int) -> str:
    """Format a list of search result items (from _quick_search) into text."""
    lines: List[str] = [f"Found {len(items)} results\n"]
    for i, r in enumerate(items[:limit]):
        title = getattr(r, "title", "") or ""
        score = getattr(r, "score", 0.0)
        doc_id = getattr(r, "doc_id", "") or ""
        snippet = getattr(r, "snippet", "") or ""
        source = getattr(r, "source_path", "") or ""

        lines.append(f"{i + 1}. {title} (score: {score:.3f})")
        if doc_id:
            lines.append(f"   doc_id: {doc_id}")
        if source:
            lines.append(f"   source: {source}")
        if snippet:
            clean = snippet.replace("\n", " ").strip()
            if len(clean) > 200:
                clean = clean[:200] + "..."
            lines.append(f"   snippet: {clean}")
        lines.append("")
    return "\n".join(lines)


# ── Result formatting helpers ──────────────────────────────────

def _format_search_results(results: Any, limit: int) -> str:
    """Format PaginatedResults or UnifiedSearchResults into readable text."""
    lines: List[str] = []

    # Handle PaginatedResults (from BM25Searcher)
    items = getattr(results, "results", None) or getattr(results, "items", None) or []
    if not items and isinstance(results, list):
        items = results

    total = getattr(results, "total", len(items))
    exec_time = getattr(results, "execution_time", 0.0)

    lines.append(f"Found {total} results ({exec_time:.1f}ms)\n")

    for i, r in enumerate(items[:limit]):
        title = getattr(r, "title", "") or (r.get("title", "") if isinstance(r, dict) else "")
        score = getattr(r, "score", 0.0) or (r.get("score", 0.0) if isinstance(r, dict) else 0.0)
        doc_id = getattr(r, "doc_id", "") or (r.get("doc_id", "") if isinstance(r, dict) else "")
        snippet = getattr(r, "snippet", "") or (r.get("snippet", "") if isinstance(r, dict) else "")
        source = getattr(r, "source_path", "") or (r.get("source_path", "") if isinstance(r, dict) else "")

        lines.append(f"{i + 1}. {title} (score: {score:.3f})")
        if doc_id:
            lines.append(f"   doc_id: {doc_id}")
        if source:
            lines.append(f"   source: {source}")
        if snippet:
            # Clean snippet for display
            clean = snippet.replace("\n", " ").strip()
            if len(clean) > 200:
                clean = clean[:200] + "..."
            lines.append(f"   snippet: {clean}")
        lines.append("")

    return "\n".join(lines) if lines else "No results found."


def _format_agent_response(resp: Any) -> str:
    """Format AgentResponse into readable text."""
    lines: List[str] = []

    if not getattr(resp, "success", True):
        error = getattr(resp, "error", "Unknown error")
        return f"Search failed: {error}"

    # Main answer
    answer = getattr(resp, "answer", "")
    if answer:
        lines.append(answer)

    # Search hits (source citations)
    hits = getattr(resp, "search_hits", [])
    if hits:
        lines.append("\n--- Sources ---")
        for i, h in enumerate(hits[:8]):
            title = h.get("title", "Untitled") if isinstance(h, dict) else str(h)
            score = h.get("score", 0.0) if isinstance(h, dict) else 0.0
            source = h.get("source_path", "") if isinstance(h, dict) else ""
            lines.append(f"{i + 1}. {title} (score: {score:.3f})")
            if source:
                lines.append(f"   {source}")

    # Stats footer
    tokens = getattr(resp, "tokens_used", 0)
    proc_time = getattr(resp, "processing_time", 0.0)
    if tokens or proc_time:
        lines.append(f"\n_Tokens: {tokens:,} | Time: {proc_time:.1f}s_")

    return "\n".join(lines)


# ── MCP Server definition ─────────────────────────────────────

def create_server(host: str = "127.0.0.1", port: int = 8000) -> "FastMCP":  # type: ignore[valid-type]
    """Create and configure the FastMCP server with all tools.
    
    Args:
        host: Bind address for SSE/HTTP transports (ignored for stdio).
        port: Listen port for SSE/HTTP transports (ignored for stdio).
    """
    if not _MCP_AVAILABLE:
        raise ImportError(
            "mcp package not installed. Run: pip install -e \".[mcp]\""
        )

    mcp = FastMCP("doc_search", host=host, port=port)  # type: ignore[abstract]

    # ── Tool: doc_search ───────────────────────────────────────
    @mcp.tool()
    async def doc_search(
        query: str,
        mode: str = "bm25",
        index_path: str = "",
        limit: int = 10,
        raw_dir: str = "",
    ) -> str:
        """Search the document knowledge base by keyword.

        Args:
            query: Search query (natural language or keywords).
            mode: Search mode — "bm25" (fast keyword), "hybrid" (BM25+regex fusion),
                  "grep" (exact pattern match). Default: "bm25".
            index_path: Index directory path (comma-separated for multi-index).
                        If empty, uses DOC_SEARCH_INDEX env var.
            limit: Max results to return. Default: 10.
            raw_dir: Raw .md directory (required for hybrid/grep modes).
                     If empty, uses DOC_SEARCH_RAW env var.

        Returns:
            Formatted search results with doc_id, title, score, snippet.
        """
        idx = _resolve_index(index_path)
        try:
            if mode == "bm25":
                # Multi-index: search all indexes, merge, re-sort
                if "," in idx:
                    all_items: List[Any] = []
                    total_exec = 0.0
                    for searcher in _get_all_searchers(idx):
                        res = searcher.search(query, limit=limit)
                        all_items.extend(getattr(res, "results", []))
                        total_exec += getattr(res, "execution_time", 0.0)
                    # Sort by score descending, take top N
                    all_items.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
                    merged = type("Merged", (), {
                        "results": all_items[:limit],
                        "total": len(all_items),
                        "execution_time": total_exec,
                    })()
                    return _format_search_results(merged, limit)
                else:
                    searcher = _get_searcher(idx)
                    results = searcher.search(query, limit=limit)
                    return _format_search_results(results, limit)

            elif mode in ("hybrid", "grep"):
                raw = _resolve_raw(raw_dir)
                from src.search.hybrid import HybridSearcher

                idx_list = [i.strip() for i in idx.split(",") if i.strip()]
                raw_list = [r.strip() for r in raw.split(",") if r.strip()]

                # Multi-index: search each index+raw pair, merge results
                if len(idx_list) > 1:
                    all_items: List[Any] = []
                    total_exec = 0.0
                    for i, index_dir in enumerate(idx_list):
                        raw_dir_i = raw_list[i] if i < len(raw_list) else raw_list[0]
                        bm25 = _get_searcher(index_dir)
                        cache_key = f"{index_dir}::{raw_dir_i}"
                        if cache_key not in _hybrid_cache:
                            _hybrid_cache[cache_key] = HybridSearcher(
                                bm25_searcher=bm25,
                                grep_raw_dir=Path(raw_dir_i),
                            )
                        hybrid = _hybrid_cache[cache_key]
                        res = hybrid.search(query, limit=limit)
                        all_items.extend(getattr(res, "results", []))
                        total_exec += getattr(res, "execution_time", 0.0)

                    all_items.sort(key=lambda r: getattr(r, "score", getattr(r, "rrf_score", 0.0)), reverse=True)
                    merged = type("Merged", (), {
                        "results": all_items[:limit],
                        "total": len(all_items),
                        "execution_time": total_exec,
                    })()
                    return _format_search_results(merged, limit)
                else:
                    # Single index
                    bm25 = _get_searcher(idx)
                    cache_key = f"{idx}::{raw}"
                    if cache_key not in _hybrid_cache:
                        _hybrid_cache[cache_key] = HybridSearcher(
                            bm25_searcher=bm25,
                            grep_raw_dir=Path(raw),
                        )
                    hybrid = _hybrid_cache[cache_key]
                    results = hybrid.search(query, limit=limit)
                    return _format_search_results(results, limit)

            else:
                return f"Unknown mode '{mode}'. Use: bm25, hybrid, or grep."

        except Exception as e:
            logger.exception("doc_search failed")
            # Graceful fallback: try grep on raw directory if BM25/hybrid fails
            raw_fallback = raw_dir or DEFAULT_RAW
            if raw_fallback:
                logger.info("Falling back to grep mode on: %s", raw_fallback)
                try:
                    return _grep_fallback(query, raw_fallback, limit)
                except Exception as fallback_err:
                    logger.error("Grep fallback also failed: %s", fallback_err)
            return f"Search error: {e}\n\nTip: If the MCP server is running with the wrong Python, ensure opencode.json uses the project venv (e.g. '.venv\\\\Scripts\\\\python.exe') instead of bare 'python'."

    # ── Tool: doc_agent ────────────────────────────────────────
    @mcp.tool()
    async def doc_agent(
        query: str,
        index_path: str = "",
        skill: str = "",
        raw_dir: str = "",
        use_rerank: bool = True,
    ) -> str:
        """Agentic RAG search — LLM autonomously searches, reads, and answers.

        Best for complex questions requiring multi-step reasoning across documents.
        The LLM uses search/read/rerank tools in a loop (up to 8 rounds) with
        dynamic confidence scoring and sufficiency checks.

        Args:
            query: Natural language question.
            index_path: Index directory (comma-separated for multi-index).
                        If empty, uses DOC_SEARCH_INDEX env var.
            skill: Optional output formatting skill: summarize, compare,
                   extract-table, detailed, timeline, action-items.
            raw_dir: Raw .md directory (enables Grep/Bash tools).
                     If empty, uses DOC_SEARCH_RAW env var.
            use_rerank: Whether to enable ZhipuAI rerank. Default: True.

        Returns:
            LLM-generated answer with source citations.
        """
        idx = _resolve_index(index_path)

        # ── MCP 快速管线：BM25 → read top docs → LLM 综合 ──
        # MCP SDK 硬编码 60s 请求超时 (DEFAULT_REQUEST_TIMEOUT_MSEC=60000)，
        # full tool_loop (4-8 轮 LLM 调用) 容易超时。
        # 改用简化管线：1 次 BM25 + 1 次 LLM 综合，总耗时 ~15-25s。
        try:
            return await _fast_agent_pipeline(query, idx, raw_dir or None, skill)
        except Exception as e:
            logger.exception("doc_agent fast pipeline failed")
            return f"Agent error: {e}"

    # ── Tool: doc_read ─────────────────────────────────────────
    @mcp.tool()
    async def doc_read(
        doc_id: str = "",
        source_path: str = "",
        index_path: str = "",
        max_lines: int = 500,
    ) -> str:
        """Read full document content by doc_id or source_path.

        Use doc_search first to get doc_id values, then call this tool
        to read the complete document. Alternatively, provide a source_path
        to read the .md file directly (no index needed).

        Args:
            doc_id: Document ID from search results. Preferred lookup method.
            source_path: Relative path from search results, resolved against
                         raw_dir. Used when doc_id is empty or not found.
            index_path: Index directory. If empty, uses DOC_SEARCH_INDEX env var.
            max_lines: Max lines to return. Default: 500.

        Returns:
            Full document content with title and metadata.
        """
        try:
            tried_indexes: List[str] = []

            # Method 1: by doc_id (from index)
            if doc_id:
                idx = _resolve_index(index_path)
                # Try each index in sequence (doc_id may live in any of them)
                idx_list = [i.strip() for i in idx.split(",") if i.strip()]
                for idx_i in idx_list:
                    searcher = _get_searcher(idx_i)
                    result = searcher.get_full_content(doc_id)
                    tried_indexes.append(idx_i)
                    if result is not None:
                        title = getattr(result, "title", "Untitled")
                        source = getattr(result, "source_path", "")
                        content = getattr(result, "full_content", "")

                        lines = content.split("\n")
                        total = len(lines)
                        display = lines[:max_lines]

                        text = f"Title: {title}\n"
                        if source:
                            text += f"Source: {source}\n"
                        text += f"Lines: {total} (showing {min(max_lines, total)})\n"
                        text += "---\n"
                        text += "\n".join(display)

                        if total > max_lines:
                            text += f"\n\n---\n... {total - max_lines} more lines."

                        return text

                # doc_id was provided but not found in any index
                logger.warning(
                    "doc_read: doc_id '%s' not found in %d indexes: %s",
                    doc_id, len(tried_indexes), ", ".join(tried_indexes),
                )
                # Fall through to source_path if available — don't error yet

            # Method 2: by source_path (direct file read)
            if source_path:
                # Build raw_dirs from both DEFAULT_RAW and index_path parent dirs
                raw_dirs: List[str] = []
                try:
                    raw = _resolve_raw("")
                    raw_dirs.extend(r.strip() for r in raw.split(",") if r.strip())
                except ValueError:
                    logger.debug("Failed to resolve raw dirs for doc_read")
                # Derive raw_dirs from index_path (parent of each index dir)
                if index_path:
                    for ip in index_path.split(","):
                        ip = ip.strip()
                        if ip:
                            parent = str(Path(ip).parent)
                            if parent not in raw_dirs:
                                raw_dirs.append(parent)
                for rd in raw_dirs:
                    file_path = Path(rd) / source_path
                    if file_path.exists() and file_path.suffix == ".md":
                        from src.converter.frontmatter import strip_frontmatter
                        _, content = strip_frontmatter(file_path.read_text(encoding="utf-8"))
                        lines = content.split("\n")
                        total = len(lines)
                        display = lines[:max_lines]

                        text = f"Title: {file_path.stem}\n"
                        text += f"Source: {source_path}\n"
                        text += f"Lines: {total} (showing {min(max_lines, total)})\n"
                        text += "---\n"
                        text += "\n".join(display)

                        if total > max_lines:
                            text += f"\n\n---\n... {total - max_lines} more lines."

                        return text

                return f"File '{source_path}' not found in any raw directory. Tried: {raw_dirs}"

            # Neither method found the document
            if doc_id:
                idx_info = ", ".join(tried_indexes) if tried_indexes else _resolve_index(index_path)
                return (
                    f"❌ Document '{doc_id}' not found in any index.\n\n"
                    f"Tried indexes: {idx_info}\n"
                    f"Tip: Verify the doc_id is exactly as shown in doc_search results.\n"
                    f"Also try doc_read(source_path=\"...\") with the source_path from search results."
                )
            return "No doc_id or source_path provided."

        except Exception as e:
            logger.exception("doc_read failed")
            return f"Read error: {e}"

    # ── Tool: doc_analyze ──────────────────────────────────────
    @mcp.tool()
    async def doc_analyze(
        query: str,
        mode: str = "summarize",
        index_path: str = "",
        raw_dir: str = "",
        top_k: int = 3,
    ) -> str:
        """Deep document analysis — search + LLM analysis in one step.

        Automatically searches the index, retrieves top documents, and applies
        the specified analysis mode.

        Args:
            query: Analysis question or topic.
            mode: Analysis mode — "compare" (multi-doc comparison),
                  "extract" (structured info extraction), "summarize" (document summary),
                  "table" (table data extraction). Default: "summarize".
            index_path: Index directory. If empty, uses DOC_SEARCH_INDEX env var.
            raw_dir: Raw .md directory. If empty, uses DOC_SEARCH_RAW env var.
            top_k: Number of documents to retrieve for analysis. Default: 3.

        Returns:
            LLM-generated analysis result.
        """
        idx = _resolve_index(index_path)
        try:
            from src.agent.analysis_agent import search_and_analyze
            config = _get_config()
            resp = search_and_analyze(
                query=query,
                index_path=idx,
                config=config,
                mode=mode,
                raw_dir=raw_dir or None,
                top_k=top_k,
            )
            return _format_agent_response(resp)
        except Exception as e:
            logger.exception("doc_analyze failed")
            return f"Analysis error: {e}"

    return mcp


# ── Entry point ────────────────────────────────────────────────

# PID file for duplicate detection & health monitoring
_MCP_PID_FILE = Path(os.environ.get("TEMP", "/tmp")) / "doc_search_mcp.pid"
_MCP_STATUS_FILE = Path(os.environ.get("TEMP", "/tmp")) / "doc_search_mcp.status"


def _startup_health_check() -> None:
    """Validate startup conditions and prevent duplicate instances.

    Checks:
    1. No duplicate MCP server already running
    2. At least one index is accessible
    3. GLM_API_KEY is configured (required for agent/analyze)
    4. Writes PID file for OpenCode to detect process health

    Exits with code 1 and clear message on failure.
    """
    # ── Check 1: Duplicate detection ──
    if _MCP_PID_FILE.exists():
        try:
            old_pid = int(_MCP_PID_FILE.read_text().strip())
            # Check if the old process is still running
            import ctypes
            kernel32 = ctypes.windll.kernel32 if sys.platform == "win32" else None
            if kernel32:
                handle = kernel32.OpenProcess(0x0400, False, old_pid)  # PROCESS_QUERY_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    logger.warning(
                        "Another MCP server instance is already running (PID %d). "
                        "This process will replace it. Old PID file will be overwritten.",
                        old_pid,
                    )
        except (ValueError, OSError):
            pass  # Stale PID file — OK to overwrite

    # Write current PID
    _MCP_PID_FILE.write_text(str(os.getpid()))

    # ── Check 2: Index accessibility ──
    if not DEFAULT_INDEX:
        print(
            "WARNING: No indexes configured.\n"
            "Set DOC_SEARCH_INDEX env var or ensure the raw directory has index directories.\n"
            "MCP server will start but search tools will return errors.",
            file=sys.stderr,
        )
    else:
        idx_count = len(DEFAULT_INDEX.split(","))
        print(f"doc-search MCP: {idx_count} indexes discovered", file=sys.stderr)

    # ── Check 3: LLM API key ──
    glm_key = os.environ.get("GLM_API_KEY", "")
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not glm_key and not ds_key:
        print(
            "WARNING: No LLM API key configured.\n"
            "Set GLM_API_KEY or DEEPSEEK_API_KEY in .env file.\n"
            "doc_agent and doc_analyze tools will not work.",
            file=sys.stderr,
        )

    # ── Write status file ──
    _write_status("starting")


def _pre_warm_agent() -> None:
    """Pre-warm agent cache to avoid 15s cold-start on first request."""
    # ── P1: Pre-warm all BM25 searchers first (jieba + Tantivy loading) ──
    if DEFAULT_INDEX:
        t_sw = time.time()
        for idx_path in DEFAULT_INDEX.split(","):
            idx_path = idx_path.strip()
            if idx_path and Path(idx_path).exists():
                try:
                    _get_searcher(idx_path)
                except Exception as e:
                    logger.warning("Searcher pre-warm failed for %s: %s", idx_path, e)
        sw_elapsed = time.time() - t_sw
        logger.warning("Searchers pre-warmed (%d indexes, %.1fs)",
                       len([i for i in DEFAULT_INDEX.split(",") if i.strip()]), sw_elapsed)

    # ── Original agent pre-warm ──
    if not DEFAULT_INDEX:
        logger.warning("No indexes to pre-warm — skipping")
        return
    try:
        t0 = time.time()
        raw_first = DEFAULT_RAW.split(",")[0].strip() if DEFAULT_RAW else None
        _get_agent(DEFAULT_INDEX, raw_first)
        elapsed = time.time() - t0
        logger.warning("Agent pre-warmed in %.1fs", elapsed)
        _write_status("ready", {"pre_warm_s": f"{elapsed:.1f}", "indexes": len(DEFAULT_INDEX.split(","))})
    except Exception as e:
        logger.warning("Agent pre-warm failed (non-blocking): %s", e)
        _write_status("ready", {"pre_warm": "failed", "error": str(e)[:200]})


def _register_shutdown_handlers() -> None:
    """Register cleanup handlers for graceful shutdown."""
    import atexit
    import signal as _signal

    def _cleanup():
        _write_status("stopped")
        try:
            _MCP_PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup)

    # Handle termination signals
    def _signal_handler(signum, frame):
        _cleanup()
        sys.exit(0)

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass  # Not available on all platforms


def _write_status(state: str, extra: dict | None = None) -> None:
    """Write a status file for external monitoring."""
    try:
        status = {
            "pid": os.getpid(),
            "state": state,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "indexes": len(DEFAULT_INDEX.split(",")) if DEFAULT_INDEX else 0,
        }
        if extra:
            status.update(extra)
        _MCP_STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False))
    except Exception:
        pass  # Best-effort — don't crash if we can't write status


def main():
    """Run the MCP server.

    Transport modes:
        stdio (default): Standard I/O — for local OpenCode/Claude Desktop
        sse: Server-Sent Events — HTTP endpoint for remote clients
        streamable-http: Streamable HTTP — alternative HTTP transport

    Examples:
        python -m src.mcp_server                                    # stdio (local)
        python -m src.mcp_server --transport sse --port 9000        # SSE on port 9000
        python -m src.mcp_server --transport sse --host 0.0.0.0    # SSE, all interfaces
    """
    import argparse

    parser = argparse.ArgumentParser(description="doc-search MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"],
                        default="stdio", help="Transport protocol (default: stdio)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address for SSE/HTTP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Listen port for SSE/HTTP (default: 8000)")
    args = parser.parse_args()

    # ── Load .env for API keys (GLM_API_KEY, DEEPSEEK_API_KEY, etc.) ──
    _load_dotenv()

    if not _MCP_AVAILABLE:
        print("ERROR: mcp package not installed.", file=sys.stderr)
        sys.exit(1)

    # ── Startup validation & duplicate detection ──
    _startup_health_check()

    # Minimal logging — MCP uses stdout for protocol, stderr must stay clean
    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(levelname)s: %(message)s")
    logger.warning("doc-search MCP server starting (transport=%s, indexes=%d)",
                   args.transport, len(DEFAULT_INDEX.split(",")) if DEFAULT_INDEX else 0)

    # ── Pre-warm agent cache to avoid 15s cold-start on first request ──
    _pre_warm_agent()

    # ── Register cleanup handlers ──
    _register_shutdown_handlers()

    server = create_server(host=args.host, port=args.port)

    if args.transport == "sse":
        print(f"Starting SSE server on {args.host}:{args.port}", file=sys.stderr)
        server.run(transport="sse")
    elif args.transport == "streamable-http":
        print(f"Starting HTTP server on {args.host}:{args.port}", file=sys.stderr)
        server.run(transport="streamable-http")
    else:
        server.run()  # stdio (default)


if __name__ == "__main__":
    main()
