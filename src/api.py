"""doc-search HTTP API — FastAPI server for search functionality.

Provides REST endpoints for BM25 keyword search, AI agent search,
document retrieval, and system status. Designed for consumption by
Pi TUI via curl. Also implements the Dify External Knowledge Base
API via POST /retrieval.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

    Or programmatically:
        import uvicorn
        from src.api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import sys

# Fix Windows console encoding for Chinese and emoji output.
# Wrapped in try/except for headless environments (Task Scheduler, CI, etc.)
if sys.platform == "win32":
    try:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr is not None:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from src import __version__
from src.stats.search_logger import SearchLogger


def _get_diagnostics_db(index_path: str | None, raw_dir: str | None = None) -> Any | None:
    """Get ConvertDB for diagnostics persistence, or None if unavailable.

    Searches index_path parent dirs and raw_dir for existing convert.db.
    Returns ConvertDB instance or None. Does NOT create new databases.
    """
    if not index_path:
        return None
    try:
        idx = Path(str(index_path).split(",")[0].strip())
        candidates = [idx.parent, idx.parent.parent]
        if raw_dir:
            candidates.insert(0, Path(str(raw_dir)))
        for candidate in candidates:
            db_path = candidate / "convert.db"
            if db_path.exists():
                from src.storage.convert_db import ConvertDB
                return ConvertDB(candidate)
    except Exception:
        logger.warning("Failed to get diagnostics DB for index=%s, raw=%s", index_path, raw_dir)
    return None


app = FastAPI(title="doc-search API", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication — supports legacy WEB_API_KEY + multi-token + open mode
from src.web.auth import AuthMiddleware, TokenStore, get_auth_mode, get_web_api_key

_auth_mode = get_auth_mode()
_token_store = TokenStore() if _auth_mode == "token" else None
app.add_middleware(
    AuthMiddleware,
    legacy_key=get_web_api_key() if _auth_mode == "legacy" else "",
    token_store=_token_store,
)
if _auth_mode == "token":
    logger.info("Token-based authentication enabled (%d tokens)", _token_store.count)
elif _auth_mode == "legacy":
    logger.info("Legacy API key authentication enabled (WEB_API_KEY set)")
else:
    logger.info("Authentication disabled (no WEB_API_KEY or tokens.json)")

# ─── Pydantic Models ─────────────────────────────────────────


class QueryRequest(BaseModel):
    """BM25 keyword search request."""

    query: str
    index_path: str
    limit: int = 10
    log: bool = True


class SearchPreviewItem(BaseModel):
    """Single search result preview."""

    doc_id: str
    title: str
    score: float
    snippet: str
    source_path: str | None = None
    highlights: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """BM25 search response."""

    results: list[SearchPreviewItem]
    total: int
    limit: int
    has_more: bool
    query: str
    execution_time: float


class AgentQueryRequest(BaseModel):
    """AI agent search request."""

    query: str
    index_path: str
    raw_dir: str | None = None
    mode: str = "tool_loop"
    limit: int = 10
    use_rerank: bool = False
    skill: str | None = None
    load_skill: str | None = None
    log: bool = True
    session_id: str | None = None


class AnalyzeRequest(BaseModel):
    """Document analysis request (compare/extract/summarize/table)."""

    query: str = ""
    index_path: str
    raw_dir: str | None = None
    mode: str = "extract"  # compare | extract | summarize | table
    doc_ids: list[str] | None = None      # for compare (≥2)
    doc_id: str | None = None             # for extract/summarize/table
    aspect: str | None = None             # compare focus
    log: bool = True
    session_id: str | None = None


class DocumentResponse(BaseModel):
    """Full document content response."""

    doc_id: str
    title: str
    full_content: str
    source_path: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = __version__


class StatusResponse(BaseModel):
    """System status response."""

    status: str = "ok"
    index: dict[str, Any] | None = None


# ─── BM25Searcher Cache ─────────────────────────────────────

_searchers: dict[str, Any] = {}


def _log_search_api(query: str, response: Any, search_mode: str, index_path: str = "", enabled: bool = True, session_id: str = ""):
    """Fire-and-forget search logging — never raises, never blocks."""
    if not enabled:
        return ""
    sid = session_id or SearchLogger.generate_session_id()
    try:
        SearchLogger.log_async(
            session_id=sid,
            query=query,
            response=response,
            source="api",
            search_mode=search_mode,
            index_path=index_path,
        )
    except Exception:
        logger.warning("Failed to persist API session creation, session_id=%s", sid)
    return sid  # Return the session_id for cross-reference with Session


def _persist_api_session(
    session_id: str | None,
    query: str,
    answer: str,
    index_path: str,
    raw_dir: str = "",
    sources: list[str] | None = None,
    srch_id: str = "",
):
    """Persist an API call to a session, creating it if session_id is provided.

    Links the SearchLogger srch_session_id to session messages for cross-reference.
    Never raises — session persistence failure must not affect API response.
    """
    if not session_id:
        return
    try:
        from src.web.session_manager import get_session_manager
        mgr = get_session_manager()
        ctx = mgr.get_or_create(
            session_id=session_id,
            index_path=Path(index_path),
            raw_dir=Path(raw_dir) if raw_dir else None,
        )
        ctx.add_message("user", query, srch_id=srch_id)
        ctx.add_message("assistant", answer, srch_id=srch_id)
        if sources:
            ctx.sources = list(set(ctx.sources) | set(sources))
        mgr.save(ctx)
    except Exception:
        pass  # session persistence is best-effort


def _get_searcher(index_path: str):
    """Get or create a cached BM25Searcher for the given index path.

    Caches per index_path string to avoid re-creating Tantivy connections
    and hitting LockBusy errors from multiple IndexWriter instances.

    Args:
        index_path: Path to the Tantivy index directory.

    Returns:
        BM25Searcher instance.

    Raises:
        HTTPException: If the index path does not exist.
    """
    if index_path in _searchers:
        return _searchers[index_path]

    path = Path(index_path)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Index not found: {index_path}",
        )

    from src.search.bm25_search import create_searcher

    searcher = create_searcher(index_path=path, use_jieba=True, readonly=True)
    _searchers[index_path] = searcher
    logger.info("Created BM25Searcher for index: %s", index_path)
    return searcher


def _get_config():
    """Load Config from environment (.env).

    Returns:
        Config instance.

    Raises:
        HTTPException: If required env vars are missing.
    """
    try:
        from src.utils.config import Config

        return Config.from_env()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── Endpoints ───────────────────────────────────────────────

# Session manager (global singleton, initialized lazily)
_session_manager = None


def _get_session_manager():
    global _session_manager
    if _session_manager is None:
        from src.web.session_manager import get_session_manager
        _session_manager = get_session_manager()
    return _session_manager


async def _run_direct_answer_with_events(ctx, config, prompt: str):
    """直接 LLM 回答（无需文档检索），SSE 流式推送。

    类似 _run_direct_review_with_events，但使用通用 system prompt
    而非合规审查专用 prompt。
    """
    import litellm

    from src.web.intent_classifier import WEB_SEARCH_SYSTEM_PROMPT
    from src.web.sse_events import AgentEventType, make_event

    q = ctx.event_queue
    abort = ctx.abort_event
    ctx.prompt = prompt
    ctx.add_message("user", prompt)

    await q.put(make_event(AgentEventType.SESSION_START,
        session_id=ctx.session_id, prompt=prompt[:200],
        execution_mode="direct"))

    try:
        await q.put(make_event(AgentEventType.THINKING,
            message="正在回答...", execution_mode="direct"))

        full_content = ""
        total_tokens = 0

        response = await litellm.acompletion(
            model=config.litellm_model,
            messages=[
                {"role": "system", "content": WEB_SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            api_key=config.active_api_key,
            api_base=config.active_base_url,
            max_tokens=4096,
            temperature=0.3,
            timeout=120,
            stream=True,
        )

        async for chunk in response:
            if abort and abort.is_set():
                await q.put(make_event(AgentEventType.ABORTED, message="用户中止"))
                return

            if hasattr(chunk, "usage") and chunk.usage:
                total_tokens = getattr(chunk.usage, "total_tokens", 0)

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_content += delta.content
                await q.put(make_event(AgentEventType.ANSWER_CHUNK,
                    content=delta.content))

        await q.put(make_event(AgentEventType.ANSWER_COMPLETE,
            answer=full_content,
            success=True,
            sources=[],
            tokens_used=total_tokens,
            processing_time=0,
            execution_mode="direct",
        ))

        ctx.add_message("assistant", full_content)

    except asyncio.CancelledError:
        await q.put(make_event(AgentEventType.ABORTED, message="用户中止"))
    except RuntimeError:
        await q.put(make_event(AgentEventType.ABORTED, message="用户中止"))
    except Exception as e:
        logger.exception("Direct answer failed for session %s", ctx.session_id)
        await q.put(make_event(AgentEventType.ERROR, message=str(e)[:500]))


async def _run_agent_with_events(
    ctx, config, prompt: str, mode: str = "tool_loop", skill: str = "", loaded_skill_content: str | None = None
):
    """Run SearchAgent and push events to the SSE queue."""
    from src.agent.search_agent import create_search_agent
    from src.stats.diagnostics import DiagnosticsCollector
    from src.web.sse_events import AgentEventType, make_event

    q = ctx.event_queue
    abort = ctx.abort_event
    ctx.prompt = prompt
    ctx.add_message("user", prompt)

    # session_start
    await q.put(make_event(AgentEventType.SESSION_START,
        session_id=ctx.session_id, prompt=prompt))

    try:
        # thinking
        await q.put(make_event(AgentEventType.THINKING,
            message="正在分析查询意图..."))

        # Capture the running event loop BEFORE entering thread pool
        # (asyncio.get_event_loop() returns None in threads)
        loop = asyncio.get_running_loop()

        # Create agent with event hooks — support multi-index
        index_path = getattr(ctx, "_multi_index", None) or ctx.index_path
        raw_dir = getattr(ctx, "_multi_raw", None) or ctx.raw_dir
        agent = create_search_agent(
            config=config,
            index_path=index_path,
            raw_dir=str(raw_dir) if raw_dir else None,
            use_rerank=False,
            mode=mode,
            diagnostics=DiagnosticsCollector(
                db=_get_diagnostics_db(
                    str(index_path),
                    str(raw_dir) if raw_dir else None,
                ),
            ),  # Enable step-level timing
        )

        # Override the on_tool_call callback to push SSE events
        original_run = agent._run_tool_loop

        def hooked_run(query, context=None, skill=None, loaded_skill_content=None, history=None):
            """Monkey-patched tool_loop that pushes SSE events."""
            original_chat = agent._llm_client.chat_with_tools

            def hooked_chat_with_tools(
                messages, tools, max_iterations=8,
                temperature=None, max_tokens=None,
                max_total_tokens=None, context_management="level3",
                on_tool_call=None,
            ):
                # Wrap the on_tool_call to push SSE events
                # IMPORTANT: use captured `loop`, NOT asyncio.get_event_loop()
                def tool_hook(tc, result):
                    # Push tool_call event
                    loop.call_soon_threadsafe(
                        lambda tc=tc, result=result: asyncio.ensure_future(
                            q.put(make_event(AgentEventType.TOOL_CALL,
                                tool=tc.name,
                                arguments=tc.arguments,
                            ))
                        )
                    )
                    # Push tool_result event
                    success = getattr(result, "success", True)
                    content_raw = getattr(result, "content", "")  # keep original for search_result
                    content = content_raw
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + "..."  # truncated for display
                    loop.call_soon_threadsafe(
                        lambda s=success, c=content: asyncio.ensure_future(
                            q.put(make_event(AgentEventType.TOOL_RESULT,
                                tool=tc.name,
                                success=s,
                                content_preview=c,
                            ))
                        )
                    )
                    # Push search_result event when search tool returns hits.
                    # Use content_raw (before truncation) to keep valid JSON.
                    if tc.name == "search" and success and content_raw:
                        try:
                            import json as _json
                            parsed = _json.loads(content_raw) if isinstance(content_raw, str) else content_raw
                            results = parsed.get("results", []) if isinstance(parsed, dict) else []
                            if results:
                                # Normalize search hits for frontend display
                                hits = []
                                for r in results[:10]:
                                    hits.append({
                                        "doc_id": r.get("doc_id", ""),
                                        "title": r.get("title", r.get("doc_id", "?")),
                                        "score": r.get("score", 0),
                                        "snippet": (r.get("snippet", "") or "")[:200],
                                    })
                                loop.call_soon_threadsafe(
                                    lambda hits=hits: asyncio.ensure_future(
                                        q.put(make_event(AgentEventType.SEARCH_RESULT,
                                            query=tc.arguments.get("query", ""),
                                            results=hits,
                                            total=len(results),
                                        ))
                                    )
                                )
                        except Exception:
                            pass  # search_result is non-critical
                    # Check abort
                    if abort and abort.is_set():
                        raise RuntimeError("Session aborted by user")
                    # Forward to original callback if any
                    if on_tool_call:
                        on_tool_call(tc, result)

                agent._llm_client.chat_with_tools = hooked_chat_with_tools
                try:
                    return original_chat(
                        messages=messages, tools=tools,
                        max_iterations=max_iterations,
                        temperature=temperature, max_tokens=max_tokens,
                        max_total_tokens=max_total_tokens,
                        context_management=context_management,
                        on_tool_call=tool_hook,
                    )
                finally:
                    agent._llm_client.chat_with_tools = original_chat

            return original_run(query, context, skill, loaded_skill_content, history=history)

        agent._run_tool_loop = hooked_run

        # Run agent (blocking, in thread)
        prior_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in ctx.messages[:-1]  # exclude the just-added current prompt
            if m.get("role") in ("user", "assistant")
        ] if ctx.messages else None
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: agent.run(query=prompt, skill=skill, loaded_skill_content=loaded_skill_content, history=prior_messages)
        )

        # ── Emit Agentic RAG pipeline events from tool_calls ──
        raw_tool_calls = getattr(response, "tool_calls", []) or []

        # Extract strategy info from tool_calls pattern analysis
        strategy_info = None
        sufficiency_checks = []
        draft_verified = False
        feedback_rounds = 0

        for tc in raw_tool_calls:
            tool_name = tc.get("tool", "")
            if tool_name == "_sufficiency_check":
                sufficiency_checks.append(tc)
                feedback_rounds += 1
            elif tool_name == "_draft_verification":
                draft_verified = tc.get("sufficient", False)

        # Infer complexity from feedback rounds and tool call count
        real_tool_count = sum(1 for tc in raw_tool_calls if not tc.get("tool", "").startswith("_"))
        if feedback_rounds > 0 or real_tool_count > 6:
            complexity_level = "complex"
        elif real_tool_count > 3:
            complexity_level = "medium"
        else:
            complexity_level = "simple"

        # Emit STRATEGY_INFO event
        strategy_info = {
            "complexity": complexity_level,
            "tool_calls_count": real_tool_count,
            "decomposed": any("分解" in str(tc.get("arguments", {})) for tc in raw_tool_calls),
        }
        await q.put(make_event(AgentEventType.STRATEGY_INFO,
            complexity=complexity_level,
            tool_calls_count=real_tool_count,
            decomposed=strategy_info["decomposed"],
        ))

        # Emit SUFFICIENCY_CHECK events
        for sc in sufficiency_checks:
            await q.put(make_event(AgentEventType.SUFFICIENCY_CHECK,
                round=sc.get("arguments", {}).get("round", 0),
                sufficient=sc.get("sufficient", False),
                coverage_score=sc.get("coverage_score", 0),
                missing_aspects=sc.get("missing_aspects", []),
            ))

        # Push answer (break into chunks for visual streaming effect)
        answer = response.answer or ""
        chunk_size = 50
        for i in range(0, len(answer), chunk_size):
            if abort and abort.is_set():
                await q.put(make_event(AgentEventType.ABORTED,
                    message="用户中止"))
                return
            chunk = answer[i:i + chunk_size]
            await q.put(make_event(AgentEventType.ANSWER_CHUNK, content=chunk))
            await asyncio.sleep(0.02)  # Small delay for visual effect

        # answer_complete (with enriched Agentic RAG fields)
        await q.put(make_event(AgentEventType.ANSWER_COMPLETE,
            answer=answer,
            success=response.success,
            sources=response.sources or [],
            search_hits=getattr(response, "search_hits", []) or [],
            tokens_used=response.tokens_used or 0,
            processing_time=response.processing_time or 0,
            complexity=complexity_level,
            feedback_rounds=feedback_rounds,
            draft_verified=draft_verified,
        ))

        # Update session
        ctx.add_message("assistant", answer)
        ctx.sources = response.sources or []
        _get_session_manager().save(ctx)

    except RuntimeError as e:
        if "aborted" in str(e).lower():
            await q.put(make_event(AgentEventType.ABORTED, message="用户中止"))
        else:
            await q.put(make_event(AgentEventType.ERROR, message=str(e)))
    except Exception as e:
        logger.exception("Agent execution failed for session %s", ctx.session_id)
        await q.put(make_event(AgentEventType.ERROR, message=str(e)[:500]))


async def _sse_event_generator(ctx):
    """Async generator yielding SSE frames from the event queue."""
    from src.web.sse_events import AgentEventType, heartbeat, make_event

    q = ctx.event_queue
    abort = ctx.abort_event
    last_heartbeat = time.time()

    while True:
        # Check for abort
        if abort and abort.is_set():
            yield make_event(AgentEventType.ABORTED, message="会话已终止")
            break

        try:
            # Wait for next event with timeout (for heartbeat)
            event_str = await asyncio.wait_for(q.get(), timeout=30.0)
            yield event_str
            last_heartbeat = time.time()

            # Stop after answer_complete or error
            if "answer_complete" in event_str or '"error"' in event_str.lower():
                break
            if "aborted" in event_str.lower():
                break

        except asyncio.TimeoutError:
            # Send heartbeat to keep connection alive
            yield heartbeat()

    # Cleanup
    ctx.event_queue = None
    ctx.abort_event = None


# ─── Session Endpoints ──────────────────────────────────────


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions."""
    mgr = _get_session_manager()
    return {"sessions": mgr.list_sessions()}


@app.post("/api/sessions")
async def create_session(
    index_path: str = Query(..., description="Path to Tantivy index (comma-separated for multi-index)"),
    raw_dir: str | None = Query(None, description="Path to raw markdown directory (comma-separated)"),
    model: str = Query("deepseek-v4-pro", description="Model name"),
):
    """Create a new search session. Supports multi-index via comma-separated paths."""
    # Support comma-separated paths
    ipaths = [Path(p.strip()) for p in index_path.split(",") if p.strip()]
    for ipath in ipaths:
        if not ipath.exists():
            raise HTTPException(status_code=404, detail=f"Index not found: {ipath}")
    # Use first path as primary index, store all as comma-separated string
    primary_index = ipaths[0]
    rdir = Path(raw_dir.split(",")[0].strip()) if raw_dir else None
    mgr = _get_session_manager()
    ctx = mgr.create(index_path=primary_index, raw_dir=rdir, model=model)
    # Store full multi-index path as session attribute
    ctx._multi_index = index_path
    ctx._multi_raw = raw_dir or ""
    return {"session_id": ctx.session_id, "created": ctx.created, "indexes": len(ipaths)}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and message history."""
    mgr = _get_session_manager()
    ctx = mgr.get(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {
        "session_id": ctx.session_id,
        "prompt": ctx.prompt,
        "messages": ctx.messages,
        "sources": ctx.sources,
        "model": ctx.model,
        "created": ctx.created,
        "last_active": ctx.last_active,
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (abort if running)."""
    mgr = _get_session_manager()
    if mgr.delete(session_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@app.post("/api/sessions/{session_id}/prompt")
async def submit_prompt(
    session_id: str,
    prompt: str = Query(..., description="User prompt/question"),
    mode: str = Query("auto", description="Agent mode (auto/tool_loop/review)"),
    skill: str = Query("", description="Skill name or __load__<path> for external skill"),
):
    """Submit a prompt to an existing session (starts async agent execution).
    
    Mode detection:
    - 'auto': Auto-detect if prompt is compliance review → direct LLM
    - 'review': Force direct LLM review (skip search)
    - 'tool_loop': Force Agent search mode
    
    Skill support:
    - Built-in: summarize, compare, extract-table, detailed, timeline, action-items
    - External: __load__skills/消保岗位专业技能矩阵.md (prefix triggers file loading)
    """
    mgr = _get_session_manager()
    ctx = mgr.get(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Handle __load__ prefix for external skill files
    _skill_name = skill
    _loaded_content = None
    if skill.startswith("__load__"):
        _skill_name = ""
        skill_path = skill[8:]  # strip "__load__"
        from src.agent.skill_loader import load_skill_content
        _loaded_content = load_skill_content(skill_path)
        if _loaded_content:
            logger.info("Loaded external skill: %s", skill_path)

    # Auto-detect review mode
    from src.web.review_prompts import detect_review_mode, enhance_review_prompt

    is_review = (mode == "review") or (mode == "auto" and detect_review_mode(prompt))
    enhanced = enhance_review_prompt(prompt) if is_review else prompt

    # Auto-detect direct mode (AI Assistant fast path)
    from src.web.intent_classifier import ExecutionMode, get_classifier
    classifier = get_classifier()
    intent_mode, intent_reason = classifier.classify_with_reason(prompt)
    is_direct = (mode == "direct") or (
        mode == "auto" and not is_review and intent_mode == ExecutionMode.DIRECT
    )

    # Create event queue and abort signal
    ctx.event_queue = asyncio.Queue()
    ctx.abort_event = asyncio.Event()

    # Start execution in background
    config = _get_config()
    if is_review:
        # Review mode: inject REVIEW_ENHANCEMENT as skill content,
        # but still use RAG pipeline to search local documents
        from src.web.review_prompts import REVIEW_ENHANCEMENT
        review_skill = (_loaded_content or "") + "\n\n" + REVIEW_ENHANCEMENT if _loaded_content else REVIEW_ENHANCEMENT
        logger.info("Review mode for session %s (RAG + review rules)", session_id)
        asyncio.create_task(_run_agent_with_events(ctx, config, prompt, mode, skill="review", loaded_skill_content=review_skill))
    elif is_direct:
        logger.info("Web search mode for session %s: %s", session_id, intent_reason)
        asyncio.create_task(_run_direct_answer_with_events(ctx, config, prompt))
    else:
        asyncio.create_task(_run_agent_with_events(ctx, config, prompt, mode, skill=_skill_name, loaded_skill_content=_loaded_content))

    return {
        "status": "processing",
        "session_id": session_id,
        "mode": "review" if is_review else ("direct" if is_direct else "tool_loop"),
        "execution_mode": "review" if is_review else ("direct" if is_direct else "search_agent"),
        "intent_reason": intent_reason if is_direct else None,
    }


@app.get("/api/sessions/{session_id}/events")
async def stream_events(session_id: str):
    """SSE event stream for a session."""
    mgr = _get_session_manager()
    ctx = mgr.get(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if ctx.event_queue is None:
        raise HTTPException(status_code=400, detail="No active query in this session")

    return StreamingResponse(
        _sse_event_generator(ctx),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/sessions/{session_id}/abort")
async def abort_session(session_id: str):
    """Abort a running session."""
    mgr = _get_session_manager()
    ctx = mgr.get(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if ctx.abort_event:
        ctx.abort_event.set()
        return {"status": "aborted"}
    raise HTTPException(status_code=400, detail="No active query in this session")


# ─── Database Query Endpoints ──────────────────────────────

# ConvertDB instance cache (per raw_dir)
_convert_dbs: dict[str, Any] = {}


def _get_convert_db(raw_dir: str):
    """Get or create a cached ConvertDB instance for the given raw_dir."""
    from urllib.parse import unquote

    key = str(Path(unquote(raw_dir)).resolve())
    if key in _convert_dbs:
        return _convert_dbs[key]

    db_path = Path(key) / "convert.db"
    if not db_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"convert.db not found at: {key}",
        )

    from src.storage.convert_db import ConvertDB
    db = ConvertDB(db_path).open()
    _convert_dbs[key] = db
    logger.info("Opened ConvertDB: %s", key)
    return db


@app.get("/api/db/{raw_dir:path}/stats")
async def db_stats(raw_dir: str):
    """转换统计摘要 — 文件状态计数 + 批次进度"""
    db = _get_convert_db(raw_dir)
    stats = db.get_stats()

    # Add file counts by status
    statuses = ["success", "failed", "pending", "skipped"]
    by_status = {}
    for s in statuses:
        by_status[s] = db.count_files(status=s)

    latest_batch = db.get_latest_batch()

    return {
        "raw_dir": raw_dir,
        **stats,
        "by_status": by_status,
        "latest_batch": dict(latest_batch) if latest_batch else None,
    }


@app.get("/api/db/{raw_dir:path}/files")
async def db_files(
    raw_dir: str,
    status: str | None = Query(None),
    extension: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    stream: bool = Query(False, description="Use SSE streaming for large result sets"),
):
    """文件列表 — 支持按状态/扩展名筛选，大结果集 SSE 流式"""
    db = _get_convert_db(raw_dir)

    # Query files
    if status:
        rows = db.get_files_by_status(status)
    elif extension:
        rows = db.get_files_by_extension(extension)
    else:
        rows = db.get_files_by_status("success") + db.get_files_by_status("failed")

    total = len(rows)
    rows = rows[offset:offset + limit]

    if stream and total > 100:
        # SSE streaming for large result sets
        async def file_stream():

            from src.web.sse_events import AgentEventType, sse_encode
            yield sse_encode(AgentEventType("meta"), {"total": total, "limit": limit, "offset": offset})
            for row in rows:
                yield sse_encode(AgentEventType("row"), {
                    "id": row.get("id"),
                    "filename": row.get("filename"),
                    "extension": row.get("extension"),
                    "status": row.get("status"),
                    "file_size": row.get("file_size"),
                    "convert_time": row.get("convert_time"),
                    "last_error": row.get("last_error"),
                    "tags": row.get("tags"),
                })
                await asyncio.sleep(0.01)
            yield sse_encode(AgentEventType("done"), {"count": len(rows)})

        return StreamingResponse(
            file_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "files": [
            {
                "id": r.get("id"),
                "filename": r.get("filename"),
                "extension": r.get("extension"),
                "status": r.get("status"),
                "file_size": r.get("file_size"),
                "convert_time": r.get("convert_time"),
                "last_error": r.get("last_error"),
                "tags": r.get("tags"),
                "pipeline_version": r.get("pipeline_version"),
            }
            for r in rows
        ],
    }


@app.get("/api/db/{raw_dir:path}/files/{file_id}")
async def db_file_detail(raw_dir: str, file_id: int):
    """单文件详情（含 OCR 用量、错误信息、跳过原因）"""
    db = _get_convert_db(raw_dir)
    f = db.get_file_by_id(file_id)
    if f is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return dict(f)


@app.get("/api/db/{raw_dir:path}/batches")
async def db_batches(raw_dir: str):
    """批次历史"""
    db = _get_convert_db(raw_dir)
    latest = db.get_latest_batch()
    active = db.get_active_batch()
    return {
        "latest_batch": dict(latest) if latest else None,
        "active_batch": dict(active) if active else None,
    }


@app.get("/api/db/{raw_dir:path}/token/summary")
async def db_token_summary(
    raw_dir: str,
    days: int = Query(7, ge=1, le=365),
):
    """Token 用量摘要（按类型 + 按模型）"""
    db = _get_convert_db(raw_dir)
    summary = db.get_token_usage_summary(days=days)
    return {"raw_dir": raw_dir, "days": days, **summary}


@app.get("/api/db/{raw_dir:path}/token/daily")
async def db_token_daily(
    raw_dir: str,
    days: int = Query(30, ge=1, le=365),
    format: str = Query("sse", pattern="^(sse|json)$"),
):
    """Token 逐日用量。默认 SSE 流式，支持 ?format=json 返回 JSON。"""
    db = _get_convert_db(raw_dir)
    rows = db.get_token_usage_daily(days=days)

    if format == "json":
        return JSONResponse({
            "raw_dir": raw_dir,
            "days": days,
            "days_data": [dict(r) for r in rows],
        })

    async def daily_stream():
        from src.web.sse_events import AgentEventType, sse_encode
        yield sse_encode(AgentEventType("meta"), {"days": days, "count": len(rows)})
        for row in rows:
            yield sse_encode(AgentEventType("row"), dict(row))
            await asyncio.sleep(0.01)
        yield sse_encode(AgentEventType("done"), {"count": len(rows)})

    return StreamingResponse(
        daily_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/db/{raw_dir:path}/token/models")
async def db_token_models(raw_dir: str, days: int = Query(7)):
    """按模型统计 Token 用量"""
    db = _get_convert_db(raw_dir)
    rows = db.get_token_usage_by_model(days=days)
    return {"raw_dir": raw_dir, "days": days, "models": [dict(r) for r in rows]}


@app.get("/api/db/{raw_dir:path}/budget")
async def db_budget(raw_dir: str):
    """预算状态"""
    db = _get_convert_db(raw_dir)
    try:
        from src.stats.budget_guard import BudgetGuard
        guard = BudgetGuard(db)
        budgets = guard.get_budgets()
        checks = [guard.check_budget() for _ in budgets] if budgets else []
        return {"budgets": [dict(b) for b in budgets], "checks": checks}
    except Exception as e:
        return {"error": str(e), "budgets": []}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    return HealthResponse()


@app.post("/query", response_model=QueryResponse)
async def query_search(request: QueryRequest):
    """BM25 keyword search (fast path, ~35ms).

    Returns paginated preview results with snippets and highlights.
    """
    searcher = _get_searcher(request.index_path)

    # Run synchronous BM25 search in executor with timeout guard
    loop = asyncio.get_event_loop()
    try:
        results = await asyncio.wait_for(
            loop.run_in_executor(None, searcher.search, request.query, request.limit),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Search timed out") from None

    preview_items = [
        SearchPreviewItem(
            doc_id=r.doc_id,
            title=r.title,
            score=r.score,
            snippet=r.snippet,
            source_path=str(r.source_path) if r.source_path else None,
            highlights=r.highlights,
        )
        for r in results.results
    ]

    resp = QueryResponse(
        results=preview_items,
        total=results.total,
        limit=results.limit,
        has_more=results.has_more,
        query=results.query,
        execution_time=results.execution_time,
    )
    _log_search_api(request.query, resp.model_dump(), "bm25", request.index_path, enabled=request.log)
    return resp


# ── Search mode endpoints ───────────────────────────────────


class SearchModeRequest(BaseModel):
    query: str
    index_path: str
    raw_dir: str = ""
    limit: int = 10
    log: bool = True


@app.post("/api/search/grep")
async def search_grep(request: SearchModeRequest):
    """Grep search over raw .md files (no index required)."""
    if not request.raw_dir:
        raise HTTPException(400, "raw_dir required for grep search")
    import math as _math
    from collections import defaultdict

    from src.agent.tools.grep import GrepTool

    tool = GrepTool(raw_dir=Path(request.raw_dir), max_results=request.limit)
    result = tool.execute(
        pattern=request.query,
        max_results=request.limit,
        file_filter="*.md",
    )
    if not result.success:
        raise HTTPException(500, result.error or "Grep search failed")

    # Convert GrepTool line output to structured results (port of hybrid._convert_grep)
    file_data: dict[str, dict] = defaultdict(
        lambda: {"matches": 0, "first_match": ""}
    )
    output_lines = (
        result.data.split("\n") if isinstance(result.data, str) else []
    )
    for line in output_lines:
        if line.startswith("  "):
            continue  # skip context lines
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        file_key = parts[0]
        content = parts[2].strip()
        file_data[file_key]["matches"] += 1
        if not file_data[file_key]["first_match"]:
            file_data[file_key]["first_match"] = content[:200]

    results = []
    for file_key, data in file_data.items():
        match_count = data["matches"]
        synthetic_score = _math.log1p(match_count) / _math.log1p(50)
        file_path = Path(file_key)
        results.append({
            "doc_id": file_key.replace("\\", "/").replace("/", "_").rstrip(".md"),
            "title": file_path.stem,
            "snippet": data["first_match"],
            "score": round(synthetic_score, 4),
            "source_path": file_key,
            "grep_matches": match_count,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    resp = {"results": results, "total": len(results), "mode": "grep"}
    _log_search_api(request.query, resp, "grep", request.index_path, enabled=request.log)
    return resp


@app.post("/api/search/hybrid")
async def search_hybrid(request: SearchModeRequest):
    """BM25 + Grep hybrid search with RRF fusion."""
    if not request.raw_dir:
        raise HTTPException(400, "raw_dir required for hybrid search")
    from src.search.bm25_search import create_searcher
    from src.search.hybrid import HybridSearcher

    index_path = Path(request.index_path)
    if not index_path.exists():
        raise HTTPException(400, f"Index not found: {request.index_path}")

    bm25 = create_searcher(index_path=index_path, use_jieba=True, readonly=True)
    searcher = HybridSearcher(bm25_searcher=bm25, grep_raw_dir=Path(request.raw_dir),
                              bm25_weight=1.0, grep_weight=0.5)
    results = searcher.search(request.query, limit=request.limit)
    items = [{"doc_id": r.doc_id, "title": r.title, "score": round(getattr(r, "score", 0), 4),
              "snippet": r.snippet, "source_path": str(r.source_path) if r.source_path else None}
             for r in results.results]
    resp = {"results": items, "total": len(items), "mode": "hybrid"}
    _log_search_api(request.query, resp, "hybrid", request.index_path, enabled=request.log)
    return resp


@app.post("/api/search/tag")
async def search_tag(request: SearchModeRequest):
    """Tag-based recall — extract keywords from query, match document tags."""
    from src.search.bm25_search import create_searcher as _create_bm25

    index_path = Path(request.index_path)
    if not index_path.exists():
        raise HTTPException(400, f"Index not found: {request.index_path}")

    searcher = _create_bm25(index_path=index_path, use_jieba=True, readonly=True)
    # Use BM25 with tag-weighted query as fallback — QueryRouter needs pre-built index metadata
    results = searcher.search(request.query, limit=request.limit)
    items = [{"doc_id": r.doc_id, "title": r.title,
              "source_path": str(r.source_path) if r.source_path else None,
              "snippet": r.snippet[:200] if r.snippet else None}
             for r in results.results]
    resp = {"results": items, "total": len(items), "mode": "tag"}
    _log_search_api(request.query, resp, "tag", request.index_path, enabled=request.log)
    return resp


@app.get("/api/suggest")
async def suggest_query(
    q: str = Query(..., min_length=2, description="Partial query for suggestions"),
    index_path: str = Query(..., description="Path to Tantivy index"),
    limit: int = Query(5, ge=1, le=20),
):
    """Search-as-you-type query suggestions from indexed document titles."""
    from src.search.bm25_search import create_searcher as _create_bm25

    idx = Path(index_path)
    if not idx.exists():
        raise HTTPException(404, f"Index not found: {index_path}")

    searcher = _create_bm25(index_path=idx, use_jieba=True, readonly=True)
    suggestions = searcher.suggest(partial_query=q, limit=limit)
    return {"suggestions": suggestions, "query": q}


@app.post("/query/agent")
async def query_agent(request: AgentQueryRequest):
    """AI agent search, web search, or compliance review.

    mode=direct: web search via LLM (no local index)
    mode=review or auto with review prompt: RAG + review rules
    mode=tool_loop (default): standard RAG pipeline
    """
    config = _get_config()

    # Auto-detect review mode
    from src.web.review_prompts import REVIEW_ENHANCEMENT, detect_review_mode, enhance_review_prompt
    is_review = (request.mode == "review") or (
        request.mode == "auto" and detect_review_mode(request.query)
    )

    # Auto-detect web search mode
    from src.web.intent_classifier import ExecutionMode, get_classifier
    classifier = get_classifier()
    intent_mode, intent_reason = classifier.classify_with_reason(request.query)
    is_direct = (request.mode == "direct") or (
        request.mode == "auto" and not is_review and intent_mode == ExecutionMode.DIRECT
    )

    # Web search mode: skip RAG, use internet search
    if is_direct:
        import litellm

        from src.web.intent_classifier import WEB_SEARCH_SYSTEM_PROMPT

        start = time.time()
        response = await litellm.acompletion(
            model=config.litellm_model,
            messages=[
                {"role": "system", "content": WEB_SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": request.query},
            ],
            api_key=config.active_api_key,
            api_base=config.active_base_url,
            max_tokens=4096, temperature=0.3, timeout=120, stream=False,
        )
        full_content = response.choices[0].message.content or ""
        total_tokens = getattr(response, "usage", None)
        total_tokens = total_tokens.total_tokens if total_tokens else 0
        resp = {
            "success": True, "answer": full_content,
            "execution_mode": "direct", "tokens_used": total_tokens,
            "processing_time": time.time() - start, "sources": [],
        }
        srch_id = _log_search_api(request.query, resp, "direct", request.index_path, enabled=request.log)
        _persist_api_session(request.session_id, request.query, full_content, request.index_path, srch_id=srch_id)
        return resp

    # Review mode: enhance prompt with rules + direct LLM
    if is_review:
        import litellm
        enhanced = enhance_review_prompt(request.query)

        start = time.time()
        response = await litellm.acompletion(
            model=config.litellm_model,
            messages=[{"role": "user", "content": enhanced}],
            api_key=config.active_api_key,
            api_base=config.active_base_url,
            max_tokens=4096, temperature=0.1, timeout=120, stream=False,
        )
        full_content = response.choices[0].message.content or ""
        total_tokens = getattr(response, "usage", None)
        total_tokens = total_tokens.total_tokens if total_tokens else 0
        resp = {
            "success": True, "answer": full_content,
            "execution_mode": "review", "tokens_used": total_tokens,
            "processing_time": time.time() - start, "sources": [],
        }
        srch_id = _log_search_api(request.query, resp, "review", request.index_path, enabled=request.log)
        _persist_api_session(request.session_id, request.query, full_content, request.index_path, srch_id=srch_id)
        return resp

    # Standard agent mode: RAG pipeline
    # Support comma-separated multi-index paths (Pi sends all configured indexes)
    index_paths = [Path(p.strip()) for p in request.index_path.split(",") if p.strip()]
    index_path = Path(request.index_path)  # for logging + create_search_agent parsing
    raw_dir = Path(request.raw_dir) if request.raw_dir else None

    # Check at least one index exists
    missing = [str(p) for p in index_paths if not p.exists()]
    if missing and len(missing) == len(index_paths):
        raise HTTPException(status_code=404, detail=f"No indexes found: {request.index_path}")

    from src.agent.search_agent import create_search_agent
    from src.stats.diagnostics import DiagnosticsCollector

    agent = create_search_agent(
        config=config, index_path=index_path, raw_dir=raw_dir,
        use_rerank=request.use_rerank, mode=request.mode,
        diagnostics=DiagnosticsCollector(
            db=_get_diagnostics_db(
                str(index_path),
                str(raw_dir) if raw_dir else None,
            ),
        ),
    )
    if not request.log:
        agent._no_log = True  # per-request log disabling

    # Generate srch_session_id for SearchLogger ↔ Session cross-reference
    from src.stats.search_logger import SearchLogger
    srch_id = SearchLogger.generate_session_id() if request.log or request.session_id else ""
    agent._srch_session_id = srch_id

    loaded_content = None
    if request.load_skill:
        from src.agent.skill_loader import load_skill_content
        loaded_content = load_skill_content(request.load_skill)

    # Inject review rules as skill content for review mode
    if is_review:
        loaded_content = (loaded_content or "") + "\n\n" + REVIEW_ENHANCEMENT

    # Run agent in thread pool with timeout protection (prevents indefinite blocking)
    history = None
    if request.session_id:
        try:
            sess_mgr = get_session_manager()
            sess_ctx = sess_mgr.get(request.session_id)
            if sess_ctx and sess_ctx.messages:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in sess_ctx.messages
                    if m.get("role") in ("user", "assistant")
                ]
        except Exception:
            pass
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: agent.run(
                    query=request.query,
                    skill=request.skill,
                    loaded_skill_content=loaded_content,
                    history=history,
                ),
            ),
            timeout=100.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Agent search timed out (100s). Try simplifying your query.",
        )
    result = response.to_dict()
    result["execution_mode"] = "review" if is_review else "search_agent"
    # Session persistence + SearchLogger cross-reference
    _persist_api_session(
        request.session_id,
        request.query,
        result.get("answer", ""),
        request.index_path,
        raw_dir=str(raw_dir) if raw_dir else "",
        sources=list(response.sources) if hasattr(response, "sources") else [],
        srch_id=srch_id,
    )
    return result


@app.post("/api/analyze")
async def analyze_documents(request: AnalyzeRequest):
    """Document deep analysis (compare/extract/summarize/table).

    When doc_id/doc_ids are omitted, auto-searches via BM25 first.
    Modes:
      - compare:    Compare multiple documents (doc_ids ≥2, or auto-search top-K)
      - extract:    Extract structured info from a document
      - summarize:  Generate document summary
      - table:      Extract table data from a document
    """
    config = _get_config()
    mode = request.mode

    # Auto-search mode: no doc_id/doc_ids provided
    if not request.doc_id and not request.doc_ids:
        from src.agent.analysis_agent import search_and_analyze
        resp = search_and_analyze(
            query=request.query,
            index_path=request.index_path,
            config=config,
            mode=mode,
            raw_dir=request.raw_dir,
            top_k=3,
            aspect=request.aspect,
        )
        result = resp.to_dict()
        srch_id = _log_search_api(request.query, result, "analyze", request.index_path, enabled=request.log)
        _persist_api_session(request.session_id, request.query, result.get("answer", ""), request.index_path, srch_id=srch_id)
        return result

    # Manual mode: use provided doc_id/doc_ids
    from src.agent.analysis_agent import create_analysis_agent
    raw_dir = request.raw_dir or str(Path(request.index_path).parent)
    agent = create_analysis_agent(config=config, raw_dir=raw_dir)

    if mode == "compare":
        if not request.doc_ids or len(request.doc_ids) < 2:
            raise HTTPException(400, "compare mode requires doc_ids with at least 2 entries, or omit doc_ids for auto-search")
        resp = agent.compare(doc_ids=request.doc_ids, aspect=request.aspect or request.query)
    elif mode == "summarize":
        if not request.doc_id:
            raise HTTPException(400, "summarize mode requires doc_id, or omit for auto-search")
        resp = agent.summarize(doc_id=request.doc_id, focus=request.query)
    elif mode == "table":
        if not request.doc_id:
            raise HTTPException(400, "table mode requires doc_id, or omit for auto-search")
        resp = agent.analyze_table(doc_id=request.doc_id)
    else:  # extract
        doc_id = request.doc_id or (request.doc_ids[0] if request.doc_ids else None)
        if not doc_id:
            raise HTTPException(400, "extract mode requires doc_id, or omit for auto-search")
        resp = agent.extract(doc_id=doc_id, query=request.query)

    result = resp.to_dict()
    srch_id = _log_search_api(request.query, result, "analyze", request.index_path, enabled=request.log)
    _persist_api_session(request.session_id, request.query, result.get("answer", ""), request.index_path, srch_id=srch_id)
    return result


@app.get("/document/path", response_model=DocumentResponse)
async def get_document_by_path(
    path: str = Query(..., description="Full path to the .md file"),
):
    """Get full document content by file path.

    Reads the .md file directly from the filesystem (no index needed).
    Strips YAML frontmatter before returning content.
    Used by Pi extension's doc_read tool as a source_path fallback.
    """
    from src.converter.frontmatter import strip_frontmatter

    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path}",
        )
    if not file_path.suffix == ".md":
        raise HTTPException(
            status_code=400,
            detail=f"Not a Markdown file: {path}",
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read file: {e}",
        )

    # Strip frontmatter — returns (was_stripped, content)
    _, content = strip_frontmatter(content)

    # Derive title from filename
    title = file_path.stem

    return DocumentResponse(
        doc_id="",
        title=title,
        full_content=content,
        source_path=str(file_path),
    )


@app.get("/document/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    index_path: str = Query(..., description="Path to Tantivy index"),
):
    """Get full document content by doc_id.

    Loads the complete document from the index.
    """
    searcher = _get_searcher(index_path)
    result = searcher.get_full_content(doc_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {doc_id}",
        )

    return DocumentResponse(
        doc_id=result.doc_id,
        title=result.title,
        full_content=result.full_content,
        source_path=str(result.source) if result.source else None,
    )


class RerankRequest(BaseModel):
    """Rerank request from Pi extension's doc_rerank tool."""

    query: str
    doc_ids: list[str] = []
    index_path: str = ""
    top_n: int = 5


@app.post("/rerank")
async def rerank_documents(request: RerankRequest):
    """Rerank documents by relevance using ZhipuAI Rerank API.

    Accepts a query + doc_ids, retrieves document content from the index,
    and returns ZhipuAI Rerank results. Falls back to original order on
    reranker failure (graceful degradation).
    """
    if not request.index_path:
        raise HTTPException(status_code=400, detail="index_path is required")

    if not request.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids must be non-empty")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    searcher = _get_searcher(request.index_path)

    # Retrieve document content for each doc_id
    documents: list[str] = []
    doc_meta: list[dict[str, str]] = []
    for doc_id in request.doc_ids:
        result = searcher.get_full_content(doc_id)
        if result is None:
            logger.warning("Rerank: doc_id %s not found, skipping", doc_id)
            continue
        documents.append(result.full_content or "")
        doc_meta.append({
            "doc_id": result.doc_id,
            "title": result.title or "",
            "source_path": str(result.source) if result.source else "",
        })

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="None of the requested doc_ids were found in the index",
        )

    # Try reranking via ZhipuAI
    try:
        from src.search.reranker import ZhipuAIReranker

        reranker = ZhipuAIReranker()
        reranked = reranker.rerank(
            query=request.query,
            documents=documents,
            top_n=request.top_n,
        )
    except Exception as exc:
        logger.warning("Rerank failed, returning original order: %s", exc)
        # Graceful degradation: return original order
        results = []
        for i, meta in enumerate(doc_meta):
            results.append({
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "score": round(1.0 - i * 0.01, 4),
                "source_path": meta["source_path"],
            })
        return {"results": results}

    # Map reranked results back to document metadata
    results = []
    for r in reranked:
        idx = r.index
        if idx < len(doc_meta):
            meta = doc_meta[idx]
            results.append({
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "score": r.relevance_score,
                "source_path": meta["source_path"],
            })

    return {"results": results}


@app.get("/status", response_model=StatusResponse)
async def system_status(
    index_path: str | None = Query(None, description="Path to Tantivy index"),
    raw_dir: str | None = Query(None, description="Path to raw markdown directory"),
):
    """System status endpoint.

    Returns index statistics if index_path is provided.
    """
    index_stats = None

    if index_path:
        try:
            searcher = _get_searcher(index_path)
            index_stats = searcher.get_index_stats()
        except HTTPException:
            index_stats = {"error": f"Index not found: {index_path}"}

    return StatusResponse(index=index_stats)


# ─── Static Files ────────────────────────────────────────────

# Mount web frontend static files
_static_dir = Path(__file__).resolve().parent / "web" / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    """Serve the web frontend."""
    from fastapi.responses import FileResponse
    index_file = _static_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"message": "doc-search Web API", "docs": "/docs"}


# ─── Upload Endpoints ──────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    status: str
    file_count: int


@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(..., description="Files to upload (multi-file supported)"),
    raw_dir: str = Form(..., description="Raw directory path for storage"),
    index_dir: str = Form("", description="Index directory path (optional, defaults to raw_dir/index)"),
):
    """Upload files for conversion and indexing.

    Submit files via multipart/form-data. Returns job_id immediately;
    progress streamed via GET /api/upload/{job_id}/events.
    """
    import threading

    from src.web.upload_manager import _run_upload_job, get_upload_manager

    raw_path = Path(raw_dir).resolve()
    index_path = Path(index_dir).resolve() if index_dir else raw_path / "index"
    raw_path.mkdir(parents=True, exist_ok=True)

    upload_dir = raw_path / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    saved_files: list[Path] = []
    for f in files:
        file_path = upload_dir / f.filename
        content = await f.read()
        file_path.write_bytes(content)
        saved_files.append(file_path)

    # Create upload job
    manager = get_upload_manager()
    job = manager.create(raw_path, index_path, saved_files)

    # Run conversion in background thread
    t = threading.Thread(target=_run_upload_job, args=(job,), daemon=True)
    t.start()

    return JSONResponse({
        "job_id": job.job_id,
        "status": "queued",
        "file_count": len(saved_files),
    })


@app.get("/api/upload/{job_id}")
async def upload_progress(job_id: str):
    """SSE progress stream for upload job."""
    from src.web.sse_events import heartbeat
    from src.web.upload_manager import get_upload_manager

    manager = get_upload_manager()
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"任务不存在: {job_id}")

    job.event_queue = asyncio.Queue()

    async def event_generator():
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(job.event_queue.get(), timeout=30)
                    yield frame
                    if '"complete"' in frame or '"error"' in frame:
                        break
                except asyncio.TimeoutError:
                    yield heartbeat()
        finally:
            job.event_queue = None

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/upload/history")
async def upload_history():
    """List recent upload jobs."""
    from src.web.upload_manager import get_upload_manager
    return {"jobs": get_upload_manager().list_jobs()}


# ─── Admin endpoints ────────────────────────────────────────


@app.post("/api/admin/build-index")
async def admin_build_index(raw_dir: str = Query(..., description="Raw directory path")):
    """Build/rebuild Tantivy search index from .md files."""
    import hashlib
    import shutil
    import time as _time
    from pathlib import Path as _Path

    from src.storage.index import TantivyIndexManager

    raw_path = _Path(raw_dir).resolve()
    index_path = raw_path / "index"
    if not raw_path.is_dir():
        raise HTTPException(400, f"Directory not found: {raw_dir}")

    if index_path.exists():
        shutil.rmtree(index_path)

    index_mgr = TantivyIndexManager(index_path=index_path, use_jieba=True)
    md_files = [f for f in raw_path.rglob("*.md") if not f.name.startswith("_")]

    if not md_files:
        index_mgr.close()
        return {"status": "ok", "indexed": 0}

    start = _time.time()
    count = 0
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            if not content.strip():
                continue
            # Strip YAML frontmatter before indexing
            from src.converter.frontmatter import strip_frontmatter
            _, content = strip_frontmatter(content)
            rel_path = md_file.relative_to(raw_path)
            doc_id = hashlib.sha256(rel_path.as_posix().encode()).hexdigest()[:16]
            index_mgr.add_document(doc_id=doc_id, title=md_file.stem, content=content,
                metadata={"filename": md_file.name, "source_path": str(rel_path)})
            count += 1
        except Exception:
            logger.warning("Failed to index uploaded file %s", md_file.name)

    index_mgr.commit()
    stats = index_mgr.get_stats()
    index_mgr.close()
    return {"status": "ok", "indexed": count, "elapsed_s": round(_time.time() - start, 2), "stats": stats}


@app.post("/api/admin/retry")
async def admin_retry(raw_dir: str = Query(..., description="Raw directory path")):
    """Reset failed files to pending for retry on next conversion."""
    from pathlib import Path as _Path

    from src.storage.convert_db import ConvertDB

    db_path = _Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")

    db = ConvertDB(db_path)
    try:
        db.open()
        failed = db.get_files_by_status("failed")
        for f in failed:
            db.update_file_status(f["id"], "pending", last_error=None)
        return {"status": "ok", "retried": len(failed)}
    finally:
        db.close()


@app.get("/api/admin/stats-export")
async def admin_stats_export(raw_dir: str = Query(..., description="Raw directory path"),
                             format: str = Query("json", description="Export format: json, csv, md")):
    """Export API usage statistics."""
    from io import StringIO

    from starlette.responses import Response

    from src.storage.convert_db import ConvertDB

    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        rows = db.get_token_usage_daily(days=30)

        if format == "csv":
            import csv as _csv
            buf = StringIO()
            w = _csv.writer(buf)
            w.writerow(["date", "input_tokens", "output_tokens", "total_tokens"])
            for r in rows:
                d = dict(r)
                w.writerow([d.get("date", ""), d.get("input_tokens", 0),
                           d.get("output_tokens", 0), d.get("total_tokens", 0)])
            return Response(buf.getvalue(), media_type="text/csv",
                          headers={"Content-Disposition": "attachment; filename=stats.csv"})

        if format == "md":
            lines = ["# Token Usage Statistics", "", f"Raw: {raw_dir}", "",
                     "| Date | Input | Output | Total |",
                     "|------|-------|--------|-------|"]
            for r in rows:
                d = dict(r)
                lines.append(f"| {d.get('date','')} | {d.get('input_tokens',0):,} | {d.get('output_tokens',0):,} | {d.get('total_tokens',0):,} |")
            return Response("\n".join(lines), media_type="text/markdown",
                          headers={"Content-Disposition": "attachment; filename=stats.md"})

        return {"raw_dir": raw_dir, "days": 30, "data": [dict(r) for r in rows]}
    finally:
        db.close()


@app.post("/api/admin/budget-set")
async def admin_budget_set(raw_dir: str = Query(...), name: str = Query("default"),
                           limit_cents: int = Query(...), period: str = Query("monthly")):
    """Set or update a budget."""
    from src.stats.budget_guard import BudgetGuard
    from src.storage.convert_db import ConvertDB
    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        guard = BudgetGuard(db)
        guard.set_budget(name=name, limit_cents=limit_cents, period=period)
        return {"status": "ok", "name": name, "limit_cents": limit_cents, "period": period}
    finally:
        db.close()


@app.delete("/api/admin/budget-remove")
async def admin_budget_remove(raw_dir: str = Query(...), budget_id: int = Query(...)):
    """Remove a budget by ID."""
    from src.stats.budget_guard import BudgetGuard
    from src.storage.convert_db import ConvertDB
    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        guard = BudgetGuard(db)
        guard.remove_budget(budget_id)
        return {"status": "ok", "removed_id": budget_id}
    finally:
        db.close()


@app.post("/api/admin/benchmark")
async def admin_benchmark(index_path: str = Query(...), raw_dir: str = Query(""),
                          modes: str = Query("bm25,grep"), limit: int = Query(10),
                          runs: int = Query(1)):
    """Run a lightweight benchmark comparing search modes."""
    import time as _time

    from src.search.bm25_search import create_searcher as _bm25
    from src.search.hybrid import HybridSearcher
    from src.storage.index import TantivyIndexManager

    idx = Path(index_path)
    if not idx.exists():
        raise HTTPException(400, f"Index not found: {index_path}")

    queries = ["年假", "报销", "合同", "安全", "数据"]
    results = []

    for mode_name in modes.split(","):
        mode_name = mode_name.strip()
        mode_times = []
        for _ in range(runs):
            t0 = _time.time()
            try:
                if mode_name == "bm25":
                    s = _bm25(index_path=idx, use_jieba=True, readonly=True)
                    s.search("test", limit=limit)
                elif mode_name == "grep" and raw_dir:
                    from src.agent.tools.grep import GrepTool
                    GrepTool(raw_dir=Path(raw_dir)).execute(pattern="test", max_matches=limit)
                elif mode_name == "hybrid" and raw_dir:
                    mgr = TantivyIndexManager(index_path=idx, readonly=True)
                    HybridSearcher(index_manager=mgr, raw_dir=Path(raw_dir)).search("test", limit=limit)
            except Exception:
                logger.warning("Benchmark run failed for mode=%s, index=%s", mode_name, idx)
            mode_times.append(round((_time.time() - t0) * 1000, 1))
        if mode_times:
            results.append({"mode": mode_name, "avg_ms": round(sum(mode_times) / len(mode_times), 1),
                           "min_ms": min(mode_times), "max_ms": max(mode_times), "runs": runs})

    return {"status": "ok", "results": results}


# ─── Admin Diagnostics API ──────────────────────────────────


@app.get("/api/admin/diagnostics/summary")
async def admin_diagnostics_summary(raw_dir: str = Query(..., description="Raw directory path"),
                                     days: int = Query(7, description="Days to include")):
    """Get query performance diagnostics summary."""
    from src.storage.convert_db import ConvertDB

    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        return db.get_diagnostics_summary(days=days)
    finally:
        db.close()


@app.get("/api/admin/diagnostics/slow-queries")
async def admin_diagnostics_slow(raw_dir: str = Query(..., description="Raw directory path"),
                                  threshold_ms: int = Query(30000, description="Slow query threshold (ms)"),
                                  limit: int = Query(20, description="Max results")):
    """Get slow queries above threshold."""
    from src.storage.convert_db import ConvertDB

    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        return {"queries": db.get_slow_queries(threshold_ms=threshold_ms, limit=limit)}
    finally:
        db.close()


@app.get("/api/admin/diagnostics/step-breakdown")
async def admin_diagnostics_steps(raw_dir: str = Query(..., description="Raw directory path"),
                                   days: int = Query(7, description="Days to include")):
    """Get step timing breakdown."""
    from src.storage.convert_db import ConvertDB

    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        return db.get_step_breakdown(days=days)
    finally:
        db.close()


@app.get("/api/admin/diagnostics/llm-calls")
async def admin_diagnostics_llm(raw_dir: str = Query(..., description="Raw directory path"),
                                 days: int = Query(7, description="Days to include")):
    """Get LLM call statistics by type."""
    from src.storage.convert_db import ConvertDB

    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        return {"stats": db.get_llm_call_stats(days=days)}
    finally:
        db.close()


# ─── Token Management API ────────────────────────────────────


class CreateTokenRequest(BaseModel):
    name: str = Field(..., description="Token name (e.g. 'pi-extension')")
    scopes: list[str] = Field(default=["*"], description="Scopes: *, search, agent, admin, read")
    expires_days: int = Field(default=0, description="Days until expiry (0 = never)")


@app.post("/api/admin/tokens")
async def admin_create_token(request: CreateTokenRequest):
    """Create a new API token. Requires admin scope."""
    valid_scopes = {"*", "search", "agent", "admin", "read"}
    if not all(s in valid_scopes for s in request.scopes):
        raise HTTPException(400, f"Invalid scope. Valid: {valid_scopes}")

    token = _token_store.create(
        name=request.name,
        scopes=request.scopes,
        expires_days=request.expires_days,
    )
    logger.info("Token created: %s (%s) scopes=%s", token.id, token.name, token.scopes)
    return token.to_dict(include_key=True)


@app.get("/api/admin/tokens")
async def admin_list_tokens():
    """List all tokens (keys redacted). Requires admin scope."""
    return {"tokens": _token_store.list(), "total": _token_store.count}


@app.delete("/api/admin/tokens/{token_id}")
async def admin_revoke_token(token_id: str):
    """Revoke a token. Requires admin scope."""
    if _token_store.revoke(token_id):
        logger.info("Token revoked: %s", token_id)
        return {"status": "revoked", "token_id": token_id}
    raise HTTPException(404, f"Token not found: {token_id}")


# ─── Search Feedback API ─────────────────────────────────────


class FeedbackRequest(BaseModel):
    """Search result feedback (thumbs up/down)."""

    query: str
    rating: int
    doc_id: str | None = None
    doc_title: str | None = None
    index_path: str | None = None
    session_id: str | None = None


def _resolve_feedback_db(
    index_path: str | None, raw_dir: str | None = None
):
    """Find an existing convert.db from index_path or raw_dir."""
    from src.storage.convert_db import ConvertDB

    candidates: list[Path] = []
    if raw_dir:
        candidates.append(Path(str(raw_dir)) / "convert.db")
    if index_path:
        idx = Path(str(index_path).split(",")[0].strip())
        candidates.append(idx.parent / "convert.db")
        candidates.append(idx.parent.parent / "convert.db")

    for candidate in candidates:
        if candidate.exists():
            db = ConvertDB(candidate)
            db.open()
            return db
    return None


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Record search result feedback (thumbs up/down)."""
    if request.rating not in (1, -1):
        raise HTTPException(422, "rating must be 1 (up) or -1 (down)")

    db = _resolve_feedback_db(request.index_path)
    if db is None:
        logger.info(
            "Feedback received (no DB): query=%s rating=%d", request.query[:60], request.rating
        )
        return {"status": "ok", "persisted": False}

    try:
        db.record_feedback(
            query=request.query,
            rating=request.rating,
            doc_id=request.doc_id,
            doc_title=request.doc_title,
            index_path=request.index_path,
            session_id=request.session_id,
        )
    except Exception:
        logger.exception("Failed to persist feedback")
        return {"status": "ok", "persisted": False}
    finally:
        try:
            db.close()
        except Exception:
            pass

    logger.info("Feedback recorded: query=%s rating=%d", request.query[:60], request.rating)
    return {"status": "ok", "persisted": True}


@app.get("/api/admin/feedback")
async def admin_feedback(
    raw_dir: str = Query(..., description="Raw directory path"),
    days: int = Query(7, description="Days to include"),
):
    """Get search feedback summary. Requires admin scope."""
    from src.storage.convert_db import ConvertDB

    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        return db.get_feedback_summary(days=days)
    finally:
        db.close()


@app.get("/api/admin/auth-log")
async def admin_auth_log(
    raw_dir: str = Query(..., description="Raw directory path"),
    days: int = Query(7, description="Days to include"),
    token_id: str | None = Query(None, description="Filter by token ID"),
    limit: int = Query(100, description="Max results"),
):
    """Get authentication audit log records. Requires admin scope."""
    from src.storage.convert_db import ConvertDB

    db_path = Path(raw_dir).resolve() / "convert.db"
    if not db_path.exists():
        raise HTTPException(404, f"No convert.db found at {raw_dir}")
    db = ConvertDB(db_path)
    try:
        db.open()
        return {"records": db.get_auth_log(days=days, token_id=token_id, limit=limit)}
    finally:
        db.close()


# ─── Dify External Knowledge Base API ───────────────────────


@app.post("/retrieval")
async def dify_retrieval(request: Request):
    """Dify External Knowledge Base API — retrieval endpoint.

    Implements the Dify External Knowledge API specification.
    Dify sends POST requests to {base_url}/retrieval to search
    connected knowledge bases.

    Auth: requires DIFY_API_KEY via Bearer token.
    If DIFY_API_KEY is not set, the endpoint is open (dev mode).
    """
    from fastapi.responses import JSONResponse as _JSONResponse

    from src.web.dify_retrieval import (
        DifyErrorCode,
        DifyErrorResponse,
        DifyRetrievalRequest,
        get_dify_api_key,
        get_retrieval_service,
    )

    # Auth check — separate from WEB_API_KEY middleware
    _dify_key = get_dify_api_key()
    if _dify_key:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != _dify_key:
            logger.warning("Dify retrieval auth failed from %s", request.client)
            return _JSONResponse(
                status_code=401,
                content=DifyErrorResponse(
                    error_code=DifyErrorCode.AUTH_FAILED,
                    error_msg="Authorization failed. Please check your API key.",
                ).model_dump(),
            )

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        return _JSONResponse(
            status_code=400,
            content=DifyErrorResponse(
                error_code=DifyErrorCode.INVALID_AUTH_HEADER,
                error_msg="Invalid request body. Expected JSON.",
            ).model_dump(),
        )

    try:
        retrieval_request = DifyRetrievalRequest(**body)
    except Exception as e:
        return _JSONResponse(
            status_code=400,
            content=DifyErrorResponse(
                error_code=DifyErrorCode.INVALID_AUTH_HEADER,
                error_msg=f"Invalid request: {e}",
            ).model_dump(),
        )

    # Execute retrieval
    try:
        service = get_retrieval_service()
        response = service.retrieve(retrieval_request)
        return response.model_dump()
    except KeyError as e:
        return _JSONResponse(
            status_code=404,
            content=DifyErrorResponse(
                error_code=DifyErrorCode.KNOWLEDGE_BASE_NOT_FOUND,
                error_msg=str(e),
            ).model_dump(),
        )
    except FileNotFoundError as e:
        logger.error("Index not found for knowledge_id '%s': %s", retrieval_request.knowledge_id, e)
        return _JSONResponse(
            status_code=500,
            content=DifyErrorResponse(
                error_code=DifyErrorCode.INTERNAL_ERROR,
                error_msg=f"Index not available: {e}",
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Dify retrieval failed")
        return _JSONResponse(
            status_code=500,
            content=DifyErrorResponse(
                error_code=DifyErrorCode.INTERNAL_ERROR,
                error_msg=f"Internal error: {e}",
            ).model_dump(),
        )


# ─── CLI Entry Point ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="doc-search API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--web", action="store_true", help="Launch browser on start")
    parser.add_argument("--open", action="store_true", help="Open browser after start (alias for --web)")
    args = parser.parse_args()

    # Print startup info
    url = f"http://{args.host}:{args.port}"
    auth_note = " 🔒 Token required" if _auth_mode != "open" else ""
    print("doc-search Web API v0.9.1")
    print(f"  API:  {url}/docs")
    print(f"  Web:  {url}/{auth_note}")
    print(f"  Host: {args.host}:{args.port}")
    if _auth_mode == "token":
        print(f"  Auth: token-based ({_token_store.count} tokens)")
        print("  Headers: Authorization: Bearer <key> | X-API-Key: <key>")
    elif _auth_mode == "legacy":
        print("  Auth: legacy key (WEB_API_KEY=***)")
        print("  Headers: Authorization: Bearer <key> | X-API-Key: <key>")
    else:
        print("  Auth: disabled (set WEB_API_KEY or create tokens.json)")

    if args.web or args.open:
        import threading
        import webbrowser
        def _open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "src.api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
