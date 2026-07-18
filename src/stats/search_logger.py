"""
SearchLogger — 异步搜索记录系统。

所有搜索入口 (CLI/Web/API/Pi) 统一记录问答和执行信息。
异步 fire-and-forget，不影响搜索性能。
.md 文件格式适配 LoRA/指令微调训练数据。

存储:
  - DB:   <SEARCH_LOG_DIR>/search_logs.db (集中 SQLite)
  - .md:  <SEARCH_LOG_DIR>/<session_id>.md (训练友好格式)

关闭:
  - 环境变量: NO_SEARCH_LOG=1
  - API 参数: log=false
  - CLI 参数: --no-log
"""

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 模块级开关
_SEARCH_LOG_DISABLED = os.environ.get("NO_SEARCH_LOG", "") == "1"

# Pending daemon threads (for flush support — CLI needs to wait before process exit)
_pending_threads: List[threading.Thread] = []
_threads_lock = threading.Lock()

# 默认存储目录
_DEFAULT_LOG_DIR = Path(os.environ.get(
    "SEARCH_LOG_DIR",
    "",
)) or Path.home() / ".doc-search" / "search_logs"

# DB schema 版本
_SCHEMA_VERSION = "1"


def _get_log_dir(custom_dir: Optional[Path] = None) -> Path:
    """获取日志目录，优先用传入参数，其次环境变量，最后默认值。"""
    if custom_dir:
        return Path(custom_dir)
    return _DEFAULT_LOG_DIR


class SearchLogDB:
    """搜索记录 SQLite 数据库 (独立于 convert.db)。"""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """初始化表结构 (幂等)。"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_logs (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id        TEXT NOT NULL UNIQUE,
                    query             TEXT NOT NULL,
                    answer            TEXT,
                    source            TEXT NOT NULL,
                    search_mode       TEXT NOT NULL,
                    index_path        TEXT,
                    raw_dir           TEXT,
                    model             TEXT,
                    success           INTEGER DEFAULT 1,
                    processing_time   REAL DEFAULT 0,
                    tokens_used       INTEGER DEFAULT 0,
                    tool_calls_count  INTEGER DEFAULT 0,
                    sources_count     INTEGER DEFAULT 0,
                    search_hits_json  TEXT,
                    tool_calls_json   TEXT,
                    skill             TEXT,
                    difficulty        TEXT,
                    tags              TEXT,
                    md_file_path      TEXT,
                    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sl_session ON search_logs(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sl_created ON search_logs(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sl_mode ON search_logs(search_mode)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sl_source ON search_logs(source)"
            )
            # schema version
            cur = conn.execute(
                "SELECT value FROM search_logs_schema WHERE key='version'"
            )
            if not cur.fetchone():
                conn.execute(
                    "INSERT INTO search_logs_schema (key, value) VALUES ('version', ?)",
                    (_SCHEMA_VERSION,),
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # 确保 schema 表存在
        conn.execute(
            "CREATE TABLE IF NOT EXISTS search_logs_schema (key TEXT PRIMARY KEY, value TEXT)"
        )
        return conn

    def add_search_log(self, **kwargs) -> int:
        """插入一条搜索日志。"""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR REPLACE INTO search_logs
                    (session_id, query, answer, source, search_mode,
                     index_path, raw_dir, model, success, processing_time,
                     tokens_used, tool_calls_count, sources_count,
                     search_hits_json, tool_calls_json, skill,
                     difficulty, tags, md_file_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    kwargs.get("session_id", ""),
                    kwargs.get("query", ""),
                    kwargs.get("answer", ""),
                    kwargs.get("source", ""),
                    kwargs.get("search_mode", ""),
                    kwargs.get("index_path", ""),
                    kwargs.get("raw_dir", ""),
                    kwargs.get("model", ""),
                    int(kwargs.get("success", True)),
                    kwargs.get("processing_time", 0.0),
                    kwargs.get("tokens_used", 0),
                    kwargs.get("tool_calls_count", 0),
                    kwargs.get("sources_count", 0),
                    kwargs.get("search_hits_json", ""),
                    kwargs.get("tool_calls_json", ""),
                    kwargs.get("skill", ""),
                    kwargs.get("difficulty", ""),
                    kwargs.get("tags", ""),
                    kwargs.get("md_file_path", ""),
                ),
            )
            return cur.lastrowid or 0

    def get_search_logs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        search_mode: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[str] = None,
        success_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """查询搜索日志。"""
        sql = "SELECT * FROM search_logs WHERE 1=1"
        params: list = []
        if search_mode:
            sql += " AND search_mode = ?"
            params.append(search_mode)
        if source:
            sql += " AND source = ?"
            params.append(source)
        if tags:
            sql += " AND tags LIKE ?"
            params.append(f"%{tags}%")
        if success_only:
            sql += " AND success = 1"
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        """总记录数。"""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as n FROM search_logs").fetchone()
            return row["n"] if row else 0


# ── 模块级 DB 实例缓存 ──────────────────────────────────────────────

_db_instances: Dict[str, SearchLogDB] = {}
_db_lock = threading.Lock()


def _get_db(log_dir: Path) -> SearchLogDB:
    """获取 (或创建) 指定目录的 SearchLogDB 单例。"""
    key = str(log_dir.resolve())
    with _db_lock:
        if key not in _db_instances:
            _db_instances[key] = SearchLogDB(log_dir / "search_logs.db")
        return _db_instances[key]


def _normalize_hits(raw_hits: List) -> List[Dict[str, Any]]:
    """规范化 search_hits: 提取 snippet (来自 highlights 或 直接 snippet), 统一格式。

    BM25 结果用 ``highlights`` (字符串列表); Agent 结果用 ``snippet`` (字符串)。
    训练数据 .md 格式期望 ``snippet`` key。
    """
    hits = []
    for h in raw_hits:
        if not isinstance(h, dict):
            continue
        title = h.get("title", h.get("doc_id", "unknown"))
        score = h.get("score", 0)
        snippet = h.get("snippet", "")
        if not snippet:
            highlights = h.get("highlights", [])
            if isinstance(highlights, list) and highlights:
                snippet = str(highlights[0])
        if not snippet:
            text = h.get("text", "")
            if text:
                snippet = str(text)[:500]
        hits.append({"title": title, "score": score, "snippet": snippet})
    return hits


# ── SearchLogger ─────────────────────────────────────────────────────


class SearchLogger:
    """异步搜索记录器 — fire-and-forget daemon thread。

    使用方式::

        SearchLogger.log_async(
            session_id="srch_20260617_143052_a1b2c3",
            query="年假如何申请",
            response=agent_response,
            source="cli",
            search_mode="agent",
        )
    """

    @staticmethod
    def is_enabled() -> bool:
        """是否启用搜索记录。"""
        return not _SEARCH_LOG_DISABLED

    @staticmethod
    def generate_session_id() -> str:
        """生成 session ID: srch_YYYYMMDD_HHMMSS_6hex。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hex_id = uuid.uuid4().hex[:6]
        return f"srch_{ts}_{hex_id}"

    @staticmethod
    def log_async(
        session_id: str,
        query: str,
        response: Any,
        source: str,
        search_mode: str,
        *,
        index_path: str = "",
        raw_dir: str = "",
        model: str = "",
        difficulty: str = "",
        tags: str = "",
        skill: str = "",
        log_dir: Optional[Path] = None,
    ):
        """异步记录 — 立即返回，后台线程完成写入。

        永不抛异常，不影响调用方。记录失败仅 debug 日志。
        """
        if not SearchLogger.is_enabled():
            return
        if not session_id:
            session_id = SearchLogger.generate_session_id()

        thread = threading.Thread(
            target=SearchLogger._do_log,
            kwargs=dict(
                session_id=session_id,
                query=query,
                response=response,
                source=source,
                search_mode=search_mode,
                index_path=index_path,
                raw_dir=raw_dir,
                model=model,
                difficulty=difficulty,
                tags=tags,
                skill=skill,
                log_dir=log_dir,
            ),
            daemon=True,
        )
        with _threads_lock:
            _pending_threads.append(thread)
        thread.start()

    @staticmethod
    def flush(timeout: float = 5.0):
        """等待所有 pending 日志线程完成 (CLI 退出前调用)。

        daemon thread 在主进程退出时会被杀死，可能导致 DB 写入未完成。
        此方法在 CLI 命令末尾调用，确保日志完整写入。
        """
        with _threads_lock:
            threads = list(_pending_threads)
            _pending_threads.clear()
        for t in threads:
            t.join(timeout=timeout)

    @staticmethod
    def _do_log(
        session_id: str,
        query: str,
        response: Any,
        source: str,
        search_mode: str,
        index_path: str,
        raw_dir: str,
        model: str,
        difficulty: str,
        tags: str,
        skill: str,
        log_dir: Optional[Path],
    ):
        """实际写入 — 在 daemon thread 中执行。"""
        try:
            record = SearchLogger._extract(response)
            target_dir = _get_log_dir(log_dir)

            # 1. 写 .md 文件
            md_file_path = ""
            try:
                md_file_path = SearchLogger._write_md(
                    target_dir, session_id, query, record,
                    source, search_mode, model, index_path,
                    raw_dir, difficulty, tags, skill,
                )
            except Exception as e:
                logger.debug(f"SearchLogger MD error: {e}")

            # 2. 写 DB
            try:
                db = _get_db(target_dir)
                db.add_search_log(
                    session_id=session_id,
                    query=query,
                    answer=record["answer"],
                    source=source,
                    search_mode=search_mode,
                    index_path=index_path,
                    raw_dir=raw_dir,
                    model=model,
                    success=record["success"],
                    processing_time=record["processing_time"],
                    tokens_used=record["tokens_used"],
                    tool_calls_count=len(record["tool_calls"]),
                    sources_count=len(record["sources"]),
                    search_hits_json=json.dumps(
                        record["search_hits"], ensure_ascii=False
                    ),
                    tool_calls_json=json.dumps(
                        record["tool_calls"], ensure_ascii=False
                    ),
                    skill=skill,
                    difficulty=difficulty,
                    tags=tags,
                    md_file_path=md_file_path,
                )
            except Exception as e:
                logger.debug(f"SearchLogger DB error: {e}")

        except Exception as e:
            logger.debug(f"SearchLogger error: {e}")

    @staticmethod
    def _extract(response: Any) -> Dict[str, Any]:
        """从各种 response 类型统一提取数据。

        支持:
          - AgentResponse (有 .answer, .success, .tool_calls 等)
          - dict (API JSON 响应)
          - str (纯文本回答)
          - list (搜索结果列表)
        """
        # AgentResponse — 检查是否有 answer 属性
        if hasattr(response, "answer"):
            return {
                "answer": response.answer or "",
                "success": getattr(response, "success", True),
                "processing_time": getattr(response, "processing_time", 0.0),
                "tokens_used": getattr(response, "tokens_used", 0),
                "sources": getattr(response, "sources", []),
                "search_hits": getattr(response, "search_hits", []),
                "tool_calls": getattr(response, "tool_calls", []),
                "reasoning": getattr(response, "reasoning", ""),
                "error": getattr(response, "error", ""),
            }

        # dict — API JSON
        if isinstance(response, dict):
            answer = response.get("answer", "")
            if not answer:
                # 关键词搜索结果 → 生成摘要
                results = response.get("results", [])
                if results:
                    titles = [
                        r.get("title", r.get("doc_id", ""))
                        for r in results[:10]
                    ]
                    answer = f"找到 {len(results)} 个结果: " + ", ".join(titles)
                elif response.get("results_summary"):
                    answer = response["results_summary"]

            # 规范化 search_hits: BM25 用 highlights, _write_md 期望 snippet
            raw_hits = response.get("search_hits", response.get("results", []))
            normalized_hits = _normalize_hits(raw_hits)

            # 规范化 sources: 从 search_hits 推导 (如果 response 没有显式提供)
            sources = response.get("sources", [])
            if not sources and raw_hits:
                sources = [
                    h.get("source_path") or h.get("title") or h.get("doc_id", "")
                    for h in raw_hits[:20] if isinstance(h, dict)
                ]

            # processing_time: 也检查 execution_time (BM25 用这个 key)
            pt = response.get("processing_time")
            if pt is None or pt == 0.0:
                pt = response.get("execution_time", 0.0)

            return {
                "answer": answer,
                "success": response.get("success", True),
                "processing_time": pt,
                "tokens_used": response.get("tokens_used", 0),
                "sources": sources,
                "search_hits": normalized_hits,
                "tool_calls": response.get("tool_calls", []),
                "reasoning": "",
                "error": response.get("error", ""),
            }

        # str — 纯文本
        if isinstance(response, str):
            return {
                "answer": response,
                "success": True,
                "processing_time": 0.0,
                "tokens_used": 0,
                "sources": [],
                "search_hits": [],
                "tool_calls": [],
                "reasoning": "",
                "error": "",
            }

        # list — 搜索结果列表
        if isinstance(response, list):
            return {
                "answer": f"找到 {len(response)} 个结果",
                "success": True,
                "processing_time": 0.0,
                "tokens_used": 0,
                "sources": [],
                "search_hits": response,
                "tool_calls": [],
                "reasoning": "",
                "error": "",
            }

        # fallback
        return {
            "answer": str(response),
            "success": True,
            "processing_time": 0.0,
            "tokens_used": 0,
            "sources": [],
            "search_hits": [],
            "tool_calls": [],
            "reasoning": "",
            "error": "",
        }

    @staticmethod
    def _write_md(
        log_dir: Path,
        session_id: str,
        query: str,
        record: Dict[str, Any],
        source: str,
        search_mode: str,
        model: str,
        index_path: str,
        raw_dir: str,
        difficulty: str,
        tags: str,
        skill: str,
    ) -> str:
        """生成训练友好的 .md 文件，返回文件路径。"""
        log_dir.mkdir(parents=True, exist_ok=True)
        md_path = log_dir / f"{session_id}.md"

        ts = datetime.now().isoformat(timespec="seconds")
        tag_list = f"[{tags}]" if tags else "[]"

        lines: List[str] = []

        # ── YAML Frontmatter ──
        lines.append("---")
        lines.append(f'session_id: "{session_id}"')
        lines.append(f'timestamp: "{ts}"')
        lines.append(f"source: {source}")
        lines.append(f"search_mode: {search_mode}")
        if model:
            lines.append(f'model: "{model}"')
        if index_path:
            # 转义反斜杠
            escaped = index_path.replace("\\", "\\\\")
            lines.append(f'index_path: "{escaped}"')
        if raw_dir:
            escaped = raw_dir.replace("\\", "\\\\")
            lines.append(f'raw_dir: "{escaped}"')
        lines.append(f"success: {str(record['success']).lower()}")
        lines.append(f"processing_time: {record['processing_time']:.1f}")
        lines.append(f"tokens_used: {record['tokens_used']}")
        lines.append(f"tool_calls_count: {len(record['tool_calls'])}")
        lines.append(f"sources_count: {len(record['sources'])}")
        if difficulty:
            lines.append(f"difficulty: {difficulty}")
        lines.append(f"tags: {tag_list}")
        if skill:
            lines.append(f"skill: {skill}")
        lines.append("---")
        lines.append("")

        # ── Instruction (用户查询) ──
        lines.append("# Instruction")
        lines.append("")
        lines.append(query)
        lines.append("")

        # ── Retrieved Context (搜索命中) ──
        hits = record["search_hits"]
        if hits:
            lines.append("# Retrieved Context")
            lines.append("")
            for hit in hits[:10]:
                if isinstance(hit, dict):
                    title = hit.get("title", hit.get("doc_id", "unknown"))
                    score = hit.get("score", 0)
                    snippet = hit.get("snippet", "")
                    if not snippet:
                        highlights = hit.get("highlights", [])
                        if isinstance(highlights, list) and highlights:
                            snippet = str(highlights[0])
                    lines.append(f"## {title} (score: {score:.2f})")
                    if snippet:
                        lines.append(f"> {snippet}")
                    lines.append("")
                else:
                    lines.append(f"## {hit}")
                    lines.append("")

        # ── Reasoning Trace (工具调用链) ──
        tool_calls = record["tool_calls"]
        if tool_calls:
            lines.append("# Reasoning Trace")
            lines.append("")
            for i, tc in enumerate(tool_calls, 1):
                if not isinstance(tc, dict):
                    lines.append(f"## Tool Call {i}: {tc}")
                    lines.append("")
                    continue
                name = tc.get("name", tc.get("tool", "unknown"))
                args = tc.get("arguments", tc.get("args", {}))
                exec_time = tc.get("execution_time", tc.get("time", 0))
                result_meta = tc.get("result_metadata", tc.get("metadata", {}))

                lines.append(f"## Tool Call {i}: {name}")
                if isinstance(args, dict):
                    for k, v in args.items():
                        val_str = str(v)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        lines.append(f"- **{k}**: {val_str}")
                elif args:
                    lines.append(f"- **args**: {args}")
                if exec_time:
                    lines.append(f"- **Time**: {exec_time:.3f}s")
                if isinstance(result_meta, dict):
                    for k, v in result_meta.items():
                        if k not in ("result", "output"):
                            val_str = str(v)
                            if len(val_str) > 200:
                                val_str = val_str[:200] + "..."
                            lines.append(f"- **{k}**: {val_str}")
                lines.append("")

        # ── Response (最终回答) ──
        lines.append("# Response")
        lines.append("")
        answer = record["answer"] or "(no answer)"
        lines.append(answer)
        lines.append("")

        # ── Error (如果有) ──
        if record["error"]:
            lines.append("# Error")
            lines.append("")
            lines.append(record["error"])
            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return str(md_path)
