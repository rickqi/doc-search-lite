"""AgentMemory — 轻量级跨会话记忆系统.

从已有的 search_logs.db 中召回历史问答, 零新增存储.
精确命中的重复查询直接返回历史答案, 避免重复 LLM 调用.

Usage:
    from src.stats.memory import AgentMemory
    memory = AgentMemory()
    
    # 查询前: 检查是否已有精确命中
    cached = memory.recall(query)
    if cached:
        return cached.answer  # 0 延迟
    
    # 查询后: 记录学习
    memory.learn(session_id, result)
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认 search_logs 目录 (与 SearchLogger 保持一致)
_DEFAULT_LOG_DIR = Path(os.environ.get("SEARCH_LOG_DIR", "")) or Path.home() / ".doc-search" / "search_logs"


class AgentMemory:
    """轻量级 Agent 记忆系统 — 基于 search_logs.db, 零新增存储.

    recall(): 三层召回 (精确 → 模糊 → 诊断)
    learn():  从执行结果中学习, 更新 tags
    feedback(): 记录用户反馈
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or (_DEFAULT_LOG_DIR / "search_logs.db")
        self._lock = threading.Lock()
        self._ensure_feedback_table()

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------

    def recall(self, query: str) -> Optional[Dict[str, Any]]:
        """从 search_logs 中召回历史答案.

        三层策略:
          Layer 1: 精确匹配 (query 完全一致) → 直接返回历史答案
          Layer 2: 模糊匹配 (query 包含关键词) → 返回相似问答
          Layer 3: 诊断数据 (从 convert.db) → 返回性能基准

        Args:
            query: 用户查询

        Returns:
            Dict with keys: answer, search_mode, processing_time, source
            或 None (无匹配)
        """
        if not query:
            return None

        # Layer 1: 精确匹配
        result = self._exact_match(query)
        if result:
            return result

        # Layer 2: 模糊匹配
        return self._fuzzy_match(query)

    def _exact_match(self, query: str) -> Optional[Dict[str, Any]]:
        """精确匹配: query 完全一致."""
        try:
            with self._lock:
                cur = self._conn().execute(
                    """SELECT answer, search_mode, processing_time, 
                              tokens_used, tool_calls_json, index_path, session_id
                       FROM search_logs 
                       WHERE query = ? AND source = 'agent' AND success = 1
                       ORDER BY created_at DESC LIMIT 1""",
                    (query,),
                )
                row = cur.fetchone()
        except Exception as e:
            logger.warning("AgentMemory recall (exact) failed: %s", e)
            return None

        if not row:
            return None

        logger.info("AgentMemory: exact match for [%s]", query[:40])
        return {
            "answer": row[0],
            "search_mode": row[1],
            "processing_time": row[2],
            "tokens_used": row[3],
            "tool_calls": json.loads(row[4]) if row[4] else [],
            "index_path": row[5],
            "session_id": row[6],
            "source": "exact_hit",
        }

    def _fuzzy_match(self, query: str) -> Optional[Dict[str, Any]]:
        """模糊匹配: 提取关键词 LIKE 搜索."""
        clean = query.strip()
        if len(clean) < 2:
            return None

        # 多层级关键词匹配
        p_full = clean[:20]
        p_short = clean[:4] if len(clean) >= 4 else clean[:2]
        p_min = clean[:2]

        try:
            with self._lock:
                cur = self._conn().execute(
                    """SELECT query, answer, search_mode, processing_time,
                              tokens_used, index_path, session_id
                       FROM search_logs
                       WHERE (query LIKE ? OR query LIKE ? OR query LIKE ?)
                         AND source = 'agent' AND success = 1
                       ORDER BY 
                         CASE 
                           WHEN query = ? THEN 0
                           WHEN query LIKE ? THEN 1
                           ELSE 2
                         END,
                         created_at DESC
                       LIMIT 3""",
                    (f"%{p_full}%", f"%{p_short}%", f"%{p_min}%",
                     clean, f"{clean[:20]}%"),
                )
                rows = cur.fetchall()
        except Exception as e:
            logger.warning("AgentMemory recall (fuzzy) failed: %s", e)
            return None

        if not rows:
            return None

        results = []
        for row in rows:
            results.append({
                "query": row[0],
                "answer": row[1],
                "search_mode": row[2],
                "processing_time": row[3],
                "tokens_used": row[4],
                "index_path": row[5],
                "session_id": row[6],
                "source": "fuzzy_hit",
            })

        logger.info("AgentMemory: fuzzy match for [%s] → %d results",
                     query[:40], len(results))
        return results[0]  # 返回最相似的

    # ------------------------------------------------------------------
    # Context 注入
    # ------------------------------------------------------------------

    def format_context(self, recall_result: Optional[Dict[str, Any]], max_entries: int = 3) -> Optional[str]:
        """将模糊匹配结果格式化为 Agent context 注入文本.

        适合注入到 system prompt 中, 帮助 LLM 参考历史问答.

        Args:
            recall_result: recall() 返回的模糊匹配结果 dict.
            max_entries: 最多引用几条历史.

        Returns:
            格式化 markdown 文本, 或 None (输入无效时).
        """
        if not recall_result or recall_result.get("source") != "fuzzy_hit":
            return None
        lines = ["[历史相关问答 — 以下信息来自之前的搜索记录，仅作参考]"]
        # Single result from fuzzy_match returns one dict
        lines.append(f"用户曾问: {recall_result.get('query', '')}")
        answer = (recall_result.get("answer", "") or "")[:300]
        lines.append(f"回答: {answer}")
        index_path = recall_result.get("index_path", "")
        if index_path:
            lines.append(f"来源知识库: {index_path}")
        lines.append("---")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 学习
    # ------------------------------------------------------------------

    def learn(self, session_id: str, metadata: Dict[str, Any]):
        """从一次成功的 Agent 执行中学习.

        更新 search_logs 的 tags 字段, 记录有效的搜索策略.

        Args:
            session_id: SearchLogger 生成的 session ID
            metadata: 执行元数据 (mode, index_path, confidence, tool_count, latency)
        """
        tags = {
            "learned_at": datetime.now().isoformat(),
            "effective_mode": metadata.get("mode", ""),
            "effective_index": str(metadata.get("index_path", "")),
            "confidence": metadata.get("confidence", 0),
            "tool_count": metadata.get("tool_count", 0),
            "latency_s": metadata.get("latency", 0),
            "search_count": metadata.get("search_count", 0),
            "read_count": metadata.get("read_count", 0),
        }
        try:
            with self._lock:
                self._conn().execute(
                    "UPDATE search_logs SET tags = ? WHERE session_id = ?",
                    (json.dumps(tags, ensure_ascii=False), session_id),
                )
                self._conn().commit()
        except Exception as e:
            logger.warning("AgentMemory learn failed: %s", e)

    # ------------------------------------------------------------------
    # 反馈
    # ------------------------------------------------------------------

    def feedback(self, session_id: str, rating: int, comment: str = ""):
        """记录用户对答案的反馈.

        Args:
            session_id: 对应的搜索 session ID
            rating: 1-5 星评分
            comment: 可选用户评论
        """
        if rating < 1 or rating > 5:
            logger.warning("Invalid rating: %d (must be 1-5)", rating)
            return
        try:
            with self._lock:
                self._conn().execute(
                    """INSERT INTO answer_feedback 
                       (session_id, rating, comment, created_at)
                       VALUES (?, ?, ?, datetime('now'))""",
                    (session_id, rating, comment),
                )
                self._conn().commit()
            logger.info("Feedback recorded: session=%s rating=%d", session_id, rating)
        except Exception as e:
            logger.warning("AgentMemory feedback failed: %s", e)

    def get_feedback_stats(self) -> Dict[str, Any]:
        """获取反馈统计."""
        try:
            with self._lock:
                cur = self._conn().execute(
                    """SELECT COUNT(*) as total,
                              AVG(rating) as avg_rating,
                              SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
                              SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
                       FROM answer_feedback"""
                )
                row = cur.fetchone()
                return {
                    "total": row[0] or 0,
                    "avg_rating": round(row[1], 2) if row[1] else 0,
                    "positive": row[2] or 0,
                    "negative": row[3] or 0,
                }
        except Exception as e:
            logger.warning("AgentMemory get_feedback_stats failed: %s", e)
            return {"total": 0, "avg_rating": 0, "positive": 0, "negative": 0}

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """获取数据库连接 (如果尚未存在则创建)."""
        if not hasattr(self, "_connection") or self._connection is None:
            self._connection = sqlite3.connect(str(self._db_path))
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _ensure_feedback_table(self):
        """确保 answer_feedback 表存在."""
        try:
            with self._lock:
                self._conn().execute("""CREATE TABLE IF NOT EXISTS answer_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                    comment TEXT DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )""")
                self._conn().commit()
        except Exception as e:
            logger.warning("AgentMemory init (feedback table) failed: %s", e)

    def close(self):
        """关闭数据库连接."""
        if hasattr(self, "_connection") and self._connection:
            self._connection.close()
            self._connection = None
