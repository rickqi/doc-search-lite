"""SQLite 持久化存储模块，用于批量文档转换的状态管理。"""

import json
import sqlite3
from pathlib import Path
from typing import Any

# Pipeline version tracks text-only post-processing applied to converted files.
# "1" = no fixes, "2" = table_fix + ocr_postprocess + tag_extractor integrated
PIPELINE_VERSION = "3"


class ConvertDB:
    """SQLite 状态管理器，用于批量文档转换的断点续传和增量更新。

    管理五个表：directories, files, batches, skipped, config。
    支持 WAL 模式、上下文管理器、启动恢复。
    """

    SCHEMA_VERSION = "2.1"

    def __init__(self, db_path: Path) -> None:
        """初始化转换数据库。

        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """获取数据库连接，未打开时抛出异常。"""
        if self._conn is None:
            raise RuntimeError("数据库未打开，请先调用 open() 或使用 with 语句")
        return self._conn

    @conn.setter
    def conn(self, value: sqlite3.Connection | None) -> None:
        """设置数据库连接。"""
        self._conn = value

    def open(self) -> "ConvertDB":
        """打开数据库连接，设置 PRAGMA，初始化 schema。

        Returns:
            self，支持链式调用
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # 性能和可靠性配置
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-8000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=67108864")
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()
        self._set_config("schema_version", self.SCHEMA_VERSION)
        return self

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ConvertDB":
        """上下文管理器入口。"""
        return self.open()

    def __exit__(self, *args: Any) -> None:
        """上下文管理器退出，关闭连接。"""
        self.close()

    # ── Schema 管理 ──────────────────────────────

    def _init_schema(self) -> None:
        """创建所有表和索引（如果不存在）。"""
        cursor = self.conn.cursor()

        # 1. 目录树
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS directories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id       INTEGER REFERENCES directories(id),
                name            TEXT NOT NULL,
                relative_path   TEXT NOT NULL UNIQUE,
                depth           INTEGER NOT NULL DEFAULT 0,
                file_count      INTEGER NOT NULL DEFAULT 0,
                total_size      INTEGER NOT NULL DEFAULT 0,
                index_generated INTEGER NOT NULL DEFAULT 0,
                index_mtime     TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dirs_relpath ON directories(relative_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dirs_parent  ON directories(parent_id)")

        # 2. 文件记录
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                directory_id    INTEGER NOT NULL REFERENCES directories(id),
                filename        TEXT NOT NULL,
                relative_path   TEXT NOT NULL UNIQUE,
                extension       TEXT NOT NULL,
                file_size       INTEGER NOT NULL DEFAULT 0,
                source_mtime    TEXT,
                source_hash     TEXT,
                status          TEXT NOT NULL DEFAULT 'pending',
                converter       TEXT,
                convert_time    REAL,
                convert_at      TEXT,
                convert_version TEXT,
                attempt_count   INTEGER NOT NULL DEFAULT 0,
                last_error      TEXT,
                ocr_used        INTEGER NOT NULL DEFAULT 0,
                ocr_model       TEXT,
                output_path     TEXT,
                output_size     INTEGER,
                output_hash     TEXT,
                ocr_input_tokens  INTEGER NOT NULL DEFAULT 0,
                ocr_output_tokens INTEGER NOT NULL DEFAULT 0,
                ocr_total_tokens  INTEGER NOT NULL DEFAULT 0,
                metadata_json   TEXT,
                pipeline_version TEXT DEFAULT '1',
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_relpath ON files(relative_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_status  ON files(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_dir     ON files(directory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_ext     ON files(extension)")

        # 3. 批次历史
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_type      TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'running',
                total_files     INTEGER NOT NULL DEFAULT 0,
                processed       INTEGER NOT NULL DEFAULT 0,
                success_count   INTEGER NOT NULL DEFAULT 0,
                failed_count    INTEGER NOT NULL DEFAULT 0,
                skipped_count   INTEGER NOT NULL DEFAULT 0,
                started_at      TEXT NOT NULL DEFAULT (datetime('now')),
                finished_at     TEXT,
                error_summary   TEXT,
                config_json     TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status)")

        # 4. 跳过记录
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skipped (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id         INTEGER NOT NULL REFERENCES files(id),
                reason          TEXT NOT NULL,
                detail          TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # 5. 配置
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key             TEXT PRIMARY KEY,
                value           TEXT NOT NULL,
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # 6. Token 使用量明细表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id         INTEGER REFERENCES files(id),
                model           TEXT NOT NULL,
                input_tokens    INTEGER NOT NULL DEFAULT 0,
                output_tokens   INTEGER NOT NULL DEFAULT 0,
                total_tokens    INTEGER NOT NULL DEFAULT 0,
                call_type       TEXT NOT NULL DEFAULT 'ocr',
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_file ON token_usage(file_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_model ON token_usage(model)")

        # 7. 搜索结果反馈
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT NOT NULL,
                doc_id          TEXT,
                doc_title       TEXT,
                rating          INTEGER NOT NULL,
                index_path      TEXT,
                session_id      TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created ON search_feedback(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_doc ON search_feedback(doc_title)")

        # 8. 认证审计日志
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id        TEXT,
                endpoint        TEXT NOT NULL,
                method          TEXT NOT NULL,
                client_ip       TEXT,
                status_code     INTEGER,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_log_created ON auth_log(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_log_token ON auth_log(token_id)")

        self.conn.commit()
        self._migrate_schema()

    # ── Schema 迁移 ──────────────────────────────

    def _migrate_schema(self) -> None:
        """对已有数据库执行 schema 升级（ALTER TABLE）。"""
        version = self._get_config("schema_version", "1.0") or "1.0"

        if version < "1.1":
            self._migrate_10_to_11()
        if version < "2.0":
            self._migrate_11_to_20()
        if version < "2.1":
            self._migrate_20_to_21()

        # Always ensure pipeline_version column exists (idempotent)
        self._ensure_pipeline_version_column()

    def _migrate_10_to_11(self) -> None:
        """Schema 1.0 → 1.1: 添加 token 列、metadata_json 列、token_usage 表。"""
        # 检查列是否已存在（幂等）
        columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(files)").fetchall()
        }
        if "ocr_input_tokens" not in columns:
            self.conn.execute("ALTER TABLE files ADD COLUMN ocr_input_tokens INTEGER NOT NULL DEFAULT 0")
        if "ocr_output_tokens" not in columns:
            self.conn.execute("ALTER TABLE files ADD COLUMN ocr_output_tokens INTEGER NOT NULL DEFAULT 0")
        if "ocr_total_tokens" not in columns:
            self.conn.execute("ALTER TABLE files ADD COLUMN ocr_total_tokens INTEGER NOT NULL DEFAULT 0")
        if "metadata_json" not in columns:
            self.conn.execute("ALTER TABLE files ADD COLUMN metadata_json TEXT")

        # 确保 token_usage 表存在
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id         INTEGER REFERENCES files(id),
                model           TEXT NOT NULL,
                input_tokens    INTEGER NOT NULL DEFAULT 0,
                output_tokens   INTEGER NOT NULL DEFAULT 0,
                total_tokens    INTEGER NOT NULL DEFAULT 0,
                call_type       TEXT NOT NULL DEFAULT 'ocr',
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_token_file ON token_usage(file_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_token_model ON token_usage(model)")

        self._set_config("schema_version", "1.1")
        self.conn.commit()

    def _migrate_11_to_20(self) -> None:
        """Schema 1.1 → 2.0: Extend token_usage, add pricing and budget tables."""
        # ── Extend token_usage with new columns (idempotent) ──
        columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(token_usage)").fetchall()
        }
        if "cost_millicents" not in columns:
            self.conn.execute(
                "ALTER TABLE token_usage ADD COLUMN cost_millicents INTEGER NOT NULL DEFAULT 0"
            )
        if "source_dir" not in columns:
            self.conn.execute(
                "ALTER TABLE token_usage ADD COLUMN source_dir TEXT"
            )
        if "batch_id" not in columns:
            self.conn.execute(
                "ALTER TABLE token_usage ADD COLUMN batch_id INTEGER REFERENCES batches(id)"
            )
        if "session_id" not in columns:
            self.conn.execute(
                "ALTER TABLE token_usage ADD COLUMN session_id TEXT"
            )
        if "request_meta" not in columns:
            self.conn.execute(
                "ALTER TABLE token_usage ADD COLUMN request_meta TEXT"
            )

        # ── New indexes on token_usage ──
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_type ON token_usage(call_type)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_source ON token_usage(source_dir)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_created ON token_usage(created_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_session ON token_usage(session_id)"
        )

        # ── pricing table ──
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pricing (
                model           TEXT PRIMARY KEY,
                input_price     REAL NOT NULL,
                output_price    REAL NOT NULL,
                unit_price      REAL,
                currency        TEXT DEFAULT 'CNY',
                effective_from  TEXT NOT NULL,
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.conn.executemany(
            """INSERT OR IGNORE INTO pricing (model, input_price, output_price, effective_from)
               VALUES (?, ?, ?, ?)""",
            [
                ("glm-ocr", 0.005, 0.005, "2025-01-01"),
                ("zai/glm-4", 0.05, 0.05, "2025-01-01"),
                ("zai/glm-4-flash", 0.001, 0.001, "2025-01-01"),
                ("zai/glm-4-plus", 0.05, 0.05, "2025-01-01"),
                ("deepseek/deepseek-chat", 0.001, 0.002, "2025-01-01"),
                ("rerank", 0, 0, "2025-01-01"),
            ],
        )

        # ── budget table ──
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS budget (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                limit_cents     INTEGER NOT NULL,
                period          TEXT NOT NULL DEFAULT 'monthly',
                alert_threshold REAL NOT NULL DEFAULT 0.8,
                block_exceed    INTEGER NOT NULL DEFAULT 0,
                reset_day       INTEGER DEFAULT 1,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        self._set_config("schema_version", "2.0")
        self.conn.commit()

    def _migrate_20_to_21(self) -> None:
        """Schema 2.0 → 2.1: Add query_diagnostics and llm_call_log tables."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_diagnostics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                query_hash      TEXT NOT NULL,
                query_preview   TEXT,
                complexity      TEXT,
                total_ms        INTEGER NOT NULL DEFAULT 0,
                success         INTEGER NOT NULL DEFAULT 0,
                error_type      TEXT,
                llm_call_count  INTEGER NOT NULL DEFAULT 0,
                llm_total_ms    INTEGER NOT NULL DEFAULT 0,
                llm_input_tokens  INTEGER NOT NULL DEFAULT 0,
                llm_output_tokens INTEGER NOT NULL DEFAULT 0,
                llm_retry_count INTEGER NOT NULL DEFAULT 0,
                tool_call_count INTEGER NOT NULL DEFAULT 0,
                tool_total_ms   INTEGER NOT NULL DEFAULT 0,
                tool_cache_hits INTEGER NOT NULL DEFAULT 0,
                step_timings    TEXT,
                model           TEXT,
                provider        TEXT,
                search_count    INTEGER NOT NULL DEFAULT 0,
                read_count      INTEGER NOT NULL DEFAULT 0,
                result_count    INTEGER NOT NULL DEFAULT 0,
                coverage_score  REAL,
                feedback_rounds INTEGER NOT NULL DEFAULT 0,
                final_sufficient INTEGER NOT NULL DEFAULT 0,
                source_dir      TEXT,
                search_mode     TEXT,
                metadata_json   TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_session ON query_diagnostics(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_created ON query_diagnostics(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_complexity ON query_diagnostics(complexity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_model ON query_diagnostics(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_success ON query_diagnostics(success)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_call_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnostic_id   INTEGER REFERENCES query_diagnostics(id),
                call_type       TEXT NOT NULL,
                call_sequence   INTEGER NOT NULL DEFAULT 0,
                latency_ms      INTEGER NOT NULL,
                ttft_ms         INTEGER,
                input_tokens    INTEGER NOT NULL DEFAULT 0,
                output_tokens   INTEGER NOT NULL DEFAULT 0,
                total_tokens    INTEGER NOT NULL DEFAULT 0,
                retry_count     INTEGER NOT NULL DEFAULT 0,
                model           TEXT NOT NULL,
                cache_hit       INTEGER NOT NULL DEFAULT 0,
                cached_tokens   INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_diag ON llm_call_log(diagnostic_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_type ON llm_call_log(call_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_created ON llm_call_log(created_at)")

        self._set_config("schema_version", "2.1")
        self.conn.commit()

    def _ensure_pipeline_version_column(self) -> None:
        """幂等确保 pipeline_version 列存在（无论 schema 版本）。"""
        columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(files)").fetchall()
        }
        if "pipeline_version" not in columns:
            self.conn.execute(
                "ALTER TABLE files ADD COLUMN pipeline_version TEXT DEFAULT '1'"
            )
            self.conn.commit()

    def add_token_usage(
        self,
        file_id: int,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        call_type: str = "ocr",
    ) -> None:
        """记录一次 API 调用的 token 使用量。

        Args:
            file_id: 关联的文件 ID
            model: 使用的模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            total_tokens: 总 token 数
            call_type: 调用类型 (ocr/llm)
        """
        self.conn.execute(
            """INSERT INTO token_usage (file_id, model, input_tokens, output_tokens, total_tokens, call_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, model, input_tokens, output_tokens, total_tokens, call_type),
        )
        self.conn.commit()

    def get_token_summary(self) -> dict:
        """获取 token 使用量汇总。

        Returns:
            包含总量和按模型分组的汇总
        """
        total = self.conn.execute(
            "SELECT COALESCE(SUM(input_tokens),0) as input_total, "
            "COALESCE(SUM(output_tokens),0) as output_total, "
            "COALESCE(SUM(total_tokens),0) as grand_total FROM token_usage"
        ).fetchone()

        by_model = self.conn.execute(
            "SELECT model, SUM(input_tokens) as input_tokens, "
            "SUM(output_tokens) as output_tokens, SUM(total_tokens) as total_tokens, "
            "COUNT(*) as call_count FROM token_usage GROUP BY model"
        ).fetchall()

        return {
            "input_tokens": total["input_total"],
            "output_tokens": total["output_total"],
            "total_tokens": total["grand_total"],
            "by_model": [dict(r) for r in by_model],
        }

    def add_token_usage_extended(
        self,
        call_type: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cost_millicents: int = 0,
        file_id: int = None,
        source_dir: str = None,
        batch_id: int = None,
        session_id: str = None,
        request_meta: str = None,
    ) -> int:
        """Record an API call with extended fields including cost and session tracking.

        Args:
            call_type: Call type (ocr, llm_chat, rerank, etc.)
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            total_tokens: Total token count
            cost_millicents: Cost in millicents (1/100000 yuan)
            file_id: Associated file ID
            source_dir: Source directory name for grouping
            batch_id: Associated batch ID
            session_id: Agent session ID
            request_meta: JSON-encoded request metadata

        Returns:
            Row ID of the inserted record
        """
        cursor = self.conn.execute(
            """INSERT INTO token_usage
               (file_id, model, input_tokens, output_tokens, total_tokens,
                call_type, cost_millicents, source_dir, batch_id, session_id,
                request_meta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, model, input_tokens, output_tokens, total_tokens,
             call_type, cost_millicents, source_dir, batch_id, session_id,
             request_meta),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_pricing(self, model: str) -> dict | None:
        """Get pricing row for a model.

        Args:
            model: Model name

        Returns:
            Pricing dict or None if not found
        """
        row = self.conn.execute(
            "SELECT * FROM pricing WHERE model = ?", (model,)
        ).fetchone()
        return dict(row) if row else None

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> int:
        """Calculate cost in millicents for a given model and token usage.

        Cost = (input_tokens * input_price + output_tokens * output_price) / 1000 * 100000
        where prices are in CNY per 1000 tokens.

        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in millicents (1 millicent = 1/100000 yuan)
        """
        pricing = self.get_pricing(model)
        if pricing is None:
            return 0
        input_price = pricing["input_price"]  # CNY per 1K tokens
        output_price = pricing["output_price"]
        # (tokens / 1000 * price) * 100000 = tokens * price * 100
        cost_input = input_tokens * input_price * 100
        cost_output = output_tokens * output_price * 100
        return int(round(cost_input + cost_output))

    def get_token_usage_summary(
        self, source_dir: str = None, days: int = None
    ) -> dict:
        """Get aggregated token usage summary grouped by call_type.

        Args:
            source_dir: Filter by source directory
            days: Limit to last N days

        Returns:
            Dict with call_type keys containing aggregated counts
        """
        conditions: list[str] = []
        params: list = []

        if source_dir is not None:
            conditions.append("source_dir = ?")
            params.append(source_dir)
        if days is not None:
            conditions.append("created_at >= datetime('now', ?)")
            params.append(f"-{days} days")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        rows = self.conn.execute(
            f"""SELECT call_type,
                       COUNT(*) as call_count,
                       COALESCE(SUM(input_tokens), 0) as input_tokens,
                       COALESCE(SUM(output_tokens), 0) as output_tokens,
                       COALESCE(SUM(total_tokens), 0) as total_tokens,
                       COALESCE(SUM(cost_millicents), 0) as cost_millicents
                FROM token_usage {where}
                GROUP BY call_type""",
            params,
        ).fetchall()

        # Grand totals
        total_row = self.conn.execute(
            f"""SELECT COUNT(*) as call_count,
                       COALESCE(SUM(input_tokens), 0) as input_tokens,
                       COALESCE(SUM(output_tokens), 0) as output_tokens,
                       COALESCE(SUM(total_tokens), 0) as total_tokens,
                       COALESCE(SUM(cost_millicents), 0) as cost_millicents
                FROM token_usage {where}""",
            params,
        ).fetchone()

        return {
            "by_type": {row["call_type"]: dict(row) for row in rows},
            "total": dict(total_row),
        }

    def get_token_usage_daily(
        self, days: int = 30, source_dir: str = None
    ) -> list[dict]:
        """Get daily token usage breakdown.

        Args:
            days: Number of days to look back
            source_dir: Filter by source directory

        Returns:
            List of dicts with date, call counts, and token totals
        """
        conditions = ["DATE(created_at) >= DATE('now', ?)"]
        params: list = [f"-{days} days"]

        if source_dir is not None:
            conditions.append("source_dir = ?")
            params.append(source_dir)

        where = "WHERE " + " AND ".join(conditions)

        rows = self.conn.execute(
            f"""SELECT DATE(created_at) as date,
                       COUNT(*) as call_count,
                       COALESCE(SUM(input_tokens), 0) as input_tokens,
                       COALESCE(SUM(output_tokens), 0) as output_tokens,
                       COALESCE(SUM(total_tokens), 0) as total_tokens,
                       COALESCE(SUM(cost_millicents), 0) as cost_millicents
                FROM token_usage {where}
                GROUP BY DATE(created_at)
                ORDER BY date DESC""",
            params,
        ).fetchall()

        return [dict(r) for r in rows]

    def get_token_usage_by_model(
        self, source_dir: str = None, days: int = None
    ) -> list[dict]:
        """Get token usage aggregated by model.

        Args:
            source_dir: Filter by source directory
            days: Limit to last N days

        Returns:
            List of dicts with per-model aggregated usage
        """
        conditions: list[str] = []
        params: list = []

        if source_dir is not None:
            conditions.append("source_dir = ?")
            params.append(source_dir)
        if days is not None:
            conditions.append("created_at >= datetime('now', ?)")
            params.append(f"-{days} days")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        rows = self.conn.execute(
            f"""SELECT model,
                       COUNT(*) as call_count,
                       COALESCE(SUM(input_tokens), 0) as input_tokens,
                       COALESCE(SUM(output_tokens), 0) as output_tokens,
                       COALESCE(SUM(total_tokens), 0) as total_tokens,
                       COALESCE(SUM(cost_millicents), 0) as cost_millicents
                FROM token_usage {where}
                GROUP BY model
                ORDER BY total_tokens DESC""",
            params,
        ).fetchall()

        return [dict(r) for r in rows]

    def _get_config(self, key: str, default: str | None = None) -> str | None:
        """获取配置项。

        Args:
            key: 配置键名
            default: 键不存在时的默认值

        Returns:
            配置值，不存在则返回 default
        """
        cursor = self.conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row["value"] if row else default

    def _set_config(self, key: str, value: str) -> None:
        """设置配置项（upsert）。

        Args:
            key: 配置键名
            value: 配置值
        """
        self.conn.execute(
            """INSERT INTO config (key, value, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')""",
            (key, value),
        )
        self.conn.commit()

    # ── 目录操作 ─────────────────────────────────

    def upsert_directory(
        self,
        relative_path: str,
        parent_id: int | None = None,
        depth: int = 0,
        name: str = "",
    ) -> int:
        """插入或更新目录记录。

        如果 relative_path 已存在，则更新 name 和 depth。
        如果不存在则插入新记录。

        Args:
            relative_path: 相对路径（如 "风险行为数据分析/UEBA"）
            parent_id: 父目录 ID
            depth: 目录深度
            name: 目录名称

        Returns:
            目录记录 ID
        """
        self.conn.execute(
            """INSERT INTO directories (parent_id, name, relative_path, depth)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(relative_path) DO UPDATE SET
                   name = excluded.name,
                   depth = excluded.depth,
                   updated_at = datetime('now')""",
            (parent_id, name, relative_path, depth),
        )
        self.conn.commit()

        # 获取插入或已存在的行 ID
        row = self.conn.execute(
            "SELECT id FROM directories WHERE relative_path = ?",
            (relative_path,),
        ).fetchone()
        return row["id"]

    def get_directory(self, relative_path: str) -> dict | None:
        """根据相对路径获取目录记录。

        Args:
            relative_path: 目录相对路径

        Returns:
            目录记录字典，不存在返回 None
        """
        cursor = self.conn.execute(
            "SELECT * FROM directories WHERE relative_path = ?",
            (relative_path,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_directory_by_id(self, dir_id: int) -> dict | None:
        """根据 ID 获取目录记录。

        Args:
            dir_id: 目录 ID

        Returns:
            目录记录字典，不存在返回 None
        """
        cursor = self.conn.execute(
            "SELECT * FROM directories WHERE id = ?", (dir_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_subdirectories(self, parent_id: int) -> list[dict]:
        """列出指定目录的所有子目录。

        Args:
            parent_id: 父目录 ID

        Returns:
            子目录记录列表
        """
        cursor = self.conn.execute(
            "SELECT * FROM directories WHERE parent_id = ? ORDER BY name",
            (parent_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_directory_stats(self, dir_id: int, file_count: int, total_size: int) -> None:
        """更新目录的文件统计信息。

        Args:
            dir_id: 目录 ID
            file_count: 文件数量
            total_size: 文件总大小（字节）
        """
        self.conn.execute(
            """UPDATE directories
               SET file_count = ?, total_size = ?, updated_at = datetime('now')
               WHERE id = ?""",
            (file_count, total_size, dir_id),
        )
        self.conn.commit()

    def set_index_generated(self, dir_id: int, generated: bool = True) -> None:
        """设置目录的索引生成标记。

        Args:
            dir_id: 目录 ID
            generated: 是否已生成索引
        """
        self.conn.execute(
            """UPDATE directories
               SET index_generated = ?, index_mtime = datetime('now'), updated_at = datetime('now')
               WHERE id = ?""",
            (int(generated), dir_id),
        )
        self.conn.commit()

    # ── 文件操作 ─────────────────────────────────

    def upsert_file(
        self,
        relative_path: str,
        directory_id: int,
        filename: str,
        extension: str,
        file_size: int,
        source_mtime: str,
        source_hash: str,
    ) -> int:
        """插入或更新文件记录。

        - 如果 relative_path 不存在，插入新记录（status='pending'）
        - 如果已存在且之前状态为 'success'，更新 mtime/hash/size 并重置为 'pending'
        - 如果已存在且状态为 'pending' 或 'converting'，不覆盖状态

        Args:
            relative_path: 文件相对路径
            directory_id: 所属目录 ID
            filename: 文件名
            extension: 文件扩展名（如 ".xlsx"）
            file_size: 文件大小（字节）
            source_mtime: 源文件修改时间（ISO 格式）
            source_hash: 源文件 SHA256 哈希

        Returns:
            文件记录 ID
        """
        # 检查是否已存在
        existing = self.conn.execute(
            "SELECT id, status FROM files WHERE relative_path = ?",
            (relative_path,),
        ).fetchone()

        if existing is None:
            # 插入新记录
            self.conn.execute(
                """INSERT INTO files
                   (directory_id, filename, relative_path, extension,
                    file_size, source_mtime, source_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (directory_id, filename, relative_path, extension,
                 file_size, source_mtime, source_hash),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT id FROM files WHERE relative_path = ?",
                (relative_path,),
            ).fetchone()
            return row["id"]

        file_id = existing["id"]
        current_status = existing["status"]

        # 仅当之前状态为 'success' 且文件有变化时重置为 pending
        if current_status == "success":
            # Check if file actually changed (mtime or hash differs)
            prev_mtime = self.conn.execute(
                "SELECT source_mtime, source_hash FROM files WHERE id = ?",
                (file_id,),
            ).fetchone()
            mtime_changed = prev_mtime["source_mtime"] != source_mtime
            hash_changed = (
                source_hash
                and prev_mtime["source_hash"]
                and prev_mtime["source_hash"] != source_hash
            )

            if mtime_changed or hash_changed:
                self.conn.execute(
                    """UPDATE files
                       SET file_size = ?, source_mtime = ?, source_hash = ?,
                           status = 'pending', converter = NULL, convert_time = NULL,
                           convert_at = NULL, last_error = NULL, ocr_used = 0,
                           output_path = NULL, output_size = NULL, output_hash = NULL,
                           attempt_count = 0, updated_at = datetime('now')
                       WHERE id = ?""",
                    (file_size, source_mtime, source_hash, file_id),
                )
                self.conn.commit()
            else:
                # File unchanged, just update basic info
                self.conn.execute(
                    """UPDATE files
                       SET file_size = ?, source_mtime = ?, source_hash = ?,
                           updated_at = datetime('now')
                       WHERE id = ?""",
                    (file_size, source_mtime, source_hash, file_id),
                )
                self.conn.commit()
        else:
            # 对于 pending / converting / failed / skipped 状态，仅更新基本信息
            self.conn.execute(
                """UPDATE files
                   SET file_size = ?, source_mtime = ?, source_hash = ?,
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (file_size, source_mtime, source_hash, file_id),
            )
            self.conn.commit()

        return file_id

    def get_file(self, relative_path: str) -> dict | None:
        """根据相对路径获取文件记录。

        Args:
            relative_path: 文件相对路径

        Returns:
            文件记录字典，不存在返回 None
        """
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE relative_path = ?",
            (relative_path,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_file_by_id(self, file_id: int) -> dict | None:
        """根据 ID 获取文件记录。

        Args:
            file_id: 文件记录 ID

        Returns:
            文件记录字典，不存在返回 None
        """
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_pending_files(self, limit: int = 1000) -> list[dict]:
        """获取待处理的文件列表。

        Args:
            limit: 最大返回数量

        Returns:
            待处理文件记录列表
        """
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE status = 'pending' ORDER BY id LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_files_by_directory(self, directory_id: int) -> list[dict]:
        """获取指定目录下的所有文件。

        Args:
            directory_id: 目录 ID

        Returns:
            文件记录列表
        """
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE directory_id = ? ORDER BY filename",
            (directory_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_files_by_status(self, status: str) -> list[dict]:
        """获取指定状态的所有文件。

        Args:
            status: 文件状态（pending/converting/success/failed/skipped）

        Returns:
            文件记录列表
        """
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE status = ? ORDER BY id",
            (status,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_files_by_extension(self, extension: str) -> list[dict]:
        """获取指定扩展名的所有文件。

        Args:
            extension: 文件扩展名（如 ".pdf"）

        Returns:
            文件记录列表
        """
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE extension = ? ORDER BY filename",
            (extension,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_file_status(self, file_id: int, status: str, **kwargs: Any) -> None:
        """更新文件状态及附加字段。

        支持通过 kwargs 更新：converter, convert_time, convert_at,
        convert_version, attempt_count, last_error, ocr_used, ocr_model,
        output_path, output_size, output_hash。

        Args:
            file_id: 文件记录 ID
            status: 新状态
            **kwargs: 需要同时更新的其他字段
        """
        allowed_fields = {
            "converter", "convert_time", "convert_at", "convert_version",
            "attempt_count", "last_error", "ocr_used", "ocr_model",
            "output_path", "output_size", "output_hash",
            "ocr_input_tokens", "ocr_output_tokens", "ocr_total_tokens",
            "metadata_json", "pipeline_version",
        }

        set_parts = ["status = ?", "updated_at = datetime('now')"]
        values: list = [status]

        for key, value in kwargs.items():
            if key in allowed_fields:
                set_parts.append(f"{key} = ?")
                values.append(value)

        values.append(file_id)
        sql = f"UPDATE files SET {', '.join(set_parts)} WHERE id = ?"
        self.conn.execute(sql, values)
        self.conn.commit()

    def count_files(self, status: str | None = None) -> int:
        """统计文件数量。

        Args:
            status: 按状态过滤，None 表示全部

        Returns:
            文件数量
        """
        if status is None:
            cursor = self.conn.execute("SELECT COUNT(*) AS cnt FROM files")
        else:
            cursor = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM files WHERE status = ?",
                (status,),
            )
        row = cursor.fetchone()
        return row["cnt"]

    def mark_file_skipped(self, file_id: int, reason: str, detail: str = "") -> None:
        """将文件标记为已跳过并记录原因。

        Args:
            file_id: 文件记录 ID
            reason: 跳过原因（如 unsupported_format）
            detail: 详细说明
        """
        self.update_file_status(file_id, "skipped")
        self.conn.execute(
            """INSERT INTO skipped (file_id, reason, detail)
               VALUES (?, ?, ?)""",
            (file_id, reason, detail),
        )
        self.conn.commit()

    # ── 批次操作 ─────────────────────────────────

    def create_batch(
        self,
        batch_type: str,
        total_files: int,
        config: dict | None = None,
    ) -> int:
        """创建新的批次记录。

        Args:
            batch_type: 批次类型（full/incremental/resume）
            total_files: 总文件数
            config: 运行配置快照

        Returns:
            批次 ID
        """
        config_json = json.dumps(config, ensure_ascii=False) if config else None
        cursor = self.conn.execute(
            """INSERT INTO batches (batch_type, total_files, config_json)
               VALUES (?, ?, ?)""",
            (batch_type, total_files, config_json),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def update_batch_progress(
        self,
        batch_id: int,
        processed: int,
        success: int,
        failed: int,
        skipped: int,
    ) -> None:
        """更新批次处理进度。

        Args:
            batch_id: 批次 ID
            processed: 已处理数量
            success: 成功数量
            failed: 失败数量
            skipped: 跳过数量
        """
        self.conn.execute(
            """UPDATE batches
               SET processed = ?, success_count = ?, failed_count = ?, skipped_count = ?
               WHERE id = ?""",
            (processed, success, failed, skipped, batch_id),
        )
        self.conn.commit()

    def complete_batch(self, batch_id: int, status: str = "completed") -> None:
        """完成批次并设置结束时间。

        Args:
            batch_id: 批次 ID
            status: 最终状态（completed/failed/interrupted）
        """
        self.conn.execute(
            """UPDATE batches
               SET status = ?, finished_at = datetime('now')
               WHERE id = ?""",
            (status, batch_id),
        )
        self.conn.commit()

    def get_active_batch(self) -> dict | None:
        """获取当前活跃的批次（running 或 interrupted 状态）。

        Returns:
            活跃批次记录，不存在返回 None
        """
        cursor = self.conn.execute(
            """SELECT * FROM batches
               WHERE status IN ('running', 'interrupted')
               ORDER BY id DESC LIMIT 1"""
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_latest_batch(self) -> dict | None:
        """获取最新的一条批次记录。

        Returns:
            最新批次记录，不存在返回 None
        """
        cursor = self.conn.execute(
            "SELECT * FROM batches ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def mark_interrupted_batches(self) -> None:
        """启动恢复：将所有 running 状态的批次标记为 interrupted，
        将所有 converting 状态的文件重置为 pending。
        """
        self.conn.execute(
            "UPDATE batches SET status = 'interrupted' WHERE status = 'running'"
        )
        self.conn.execute(
            "UPDATE files SET status = 'pending' WHERE status = 'converting'"
        )
        self.conn.commit()

    # ── 统计信息 ─────────────────────────────────

    def get_stats(self) -> dict:
        """获取数据库统计摘要。

        Returns:
            包含目录数、各状态文件数、最近批次信息的字典
        """
        dir_count = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM directories"
        ).fetchone()["cnt"]

        file_total = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM files"
        ).fetchone()["cnt"]

        status_counts = {}
        for row in self.conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM files GROUP BY status"
        ).fetchall():
            status_counts[row["status"]] = row["cnt"]

        batch_count = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM batches"
        ).fetchone()["cnt"]

        skipped_count = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM skipped"
        ).fetchone()["cnt"]

        latest_batch = self.get_latest_batch()

        return {
            "directory_count": dir_count,
            "file_total": file_total,
            "status_counts": status_counts,
            "batch_count": batch_count,
            "skipped_count": skipped_count,
            "latest_batch": latest_batch,
        }

    # ── Query Diagnostics ──────────────────────────────────

    _DIAGNOSTIC_COLUMNS = [
        "session_id", "query_hash", "query_preview", "complexity",
        "total_ms", "success", "error_type",
        "llm_call_count", "llm_total_ms", "llm_input_tokens", "llm_output_tokens", "llm_retry_count",
        "tool_call_count", "tool_total_ms", "tool_cache_hits",
        "step_timings", "model", "provider",
        "search_count", "read_count", "result_count", "coverage_score",
        "feedback_rounds", "final_sufficient",
        "source_dir", "search_mode", "metadata_json",
    ]

    def add_query_diagnostic(self, **kwargs) -> int:
        """Write a query diagnostic record. Returns row ID."""
        cols = [c for c in self._DIAGNOSTIC_COLUMNS if c in kwargs]
        vals = [kwargs[c] for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        cursor = self.conn.execute(
            f"INSERT INTO query_diagnostics ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def add_llm_call_log(self, diagnostic_id: int, call_type: str,
                         call_sequence: int, latency_ms: int,
                         input_tokens: int = 0, output_tokens: int = 0,
                         total_tokens: int = 0, retry_count: int = 0,
                         model: str = "", cache_hit: int = 0,
                         cached_tokens: int = 0) -> int:
        """Write an LLM call log entry. Returns row ID."""
        cursor = self.conn.execute(
            """INSERT INTO llm_call_log
               (diagnostic_id, call_type, call_sequence, latency_ms,
                input_tokens, output_tokens, total_tokens, retry_count,
                model, cache_hit, cached_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (diagnostic_id, call_type, call_sequence, latency_ms,
             input_tokens, output_tokens, total_tokens, retry_count,
             model, cache_hit, cached_tokens),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_diagnostics_summary(self, days: int = 7, source_dir: str = None) -> dict:
        """Get query diagnostics summary with percentile latency."""
        where = "1=1"
        params: list = []
        if days:
            where += " AND created_at >= datetime('now', ?)"
            params.append(f"-{days} days")
        if source_dir:
            where += " AND source_dir = ?"
            params.append(source_dir)

        row = self.conn.execute(
            f"""SELECT COUNT(*) as total_queries,
                       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as success_count,
                       COALESCE(AVG(total_ms), 0) as avg_ms,
                       COALESCE(SUM(llm_call_count), 0) as total_llm_calls,
                       COALESCE(SUM(tool_cache_hits), 0) as total_cache_hits,
                       COALESCE(SUM(tool_call_count), 0) as total_tool_calls
                FROM query_diagnostics WHERE {where}""",
            params,
        ).fetchone()

        by_complexity = self.conn.execute(
            f"""SELECT complexity, COUNT(*) as count, COALESCE(AVG(total_ms), 0) as avg_ms,
                       COALESCE(AVG(llm_call_count), 0) as avg_llm_calls
                FROM query_diagnostics WHERE {where} AND complexity IS NOT NULL
                GROUP BY complexity""",
            params,
        ).fetchall()

        total = row["total_queries"]
        tool_calls = row["total_tool_calls"]
        cache_hits = row["total_cache_hits"]

        return {
            "total_queries": total,
            "success_rate": (row["success_count"] / total * 100) if total else 0,
            "avg_ms": row["avg_ms"],
            "avg_llm_calls": (row["total_llm_calls"] / total) if total else 0,
            "cache_hit_rate": (cache_hits / tool_calls * 100) if tool_calls else 0,
            "by_complexity": [dict(r) for r in by_complexity],
        }

    def get_slow_queries(self, threshold_ms: int = 30000, limit: int = 20,
                         source_dir: str = None) -> list:
        """Get slow queries above threshold."""
        where = "total_ms >= ?"
        params: list = [threshold_ms]
        if source_dir:
            where += " AND source_dir = ?"
            params.append(source_dir)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT id, session_id, query_preview, complexity, total_ms,
                       llm_call_count, llm_total_ms, tool_call_count,
                       success, error_type, model, created_at
                FROM query_diagnostics WHERE {where}
                ORDER BY total_ms DESC LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_step_breakdown(self, days: int = 7, source_dir: str = None) -> dict:
        """Get step timing breakdown aggregated across queries."""
        where = "1=1"
        params: list = []
        if days:
            where += " AND created_at >= datetime('now', ?)"
            params.append(f"-{days} days")
        if source_dir:
            where += " AND source_dir = ?"
            params.append(source_dir)

        rows = self.conn.execute(
            f"SELECT step_timings FROM query_diagnostics WHERE {where} AND step_timings IS NOT NULL",
            params,
        ).fetchall()

        from collections import defaultdict
        step_data: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            try:
                timings = json.loads(row["step_timings"])
                for step, ms in timings.items():
                    step_data[step].append(float(ms))
            except (json.JSONDecodeError, ValueError):
                continue

        result = {}
        for step, values in sorted(step_data.items()):
            values.sort()
            n = len(values)
            result[step] = {
                "count": n,
                "avg_ms": round(sum(values) / n, 1),
                "p50_ms": round(values[n // 2], 1) if n else 0,
                "p90_ms": round(values[int(n * 0.9)], 1) if n else 0,
                "max_ms": round(max(values), 1) if values else 0,
            }
        return result

    def get_llm_call_stats(self, days: int = 7, source_dir: str = None) -> list:
        """Get LLM call statistics grouped by call_type."""
        where = "1=1"
        params: list = []
        if days:
            where += " AND lc.created_at >= datetime('now', ?)"
            params.append(f"-{days} days")
        if source_dir:
            where += " AND qd.source_dir = ?"
            params.append(source_dir)

        rows = self.conn.execute(
            f"""SELECT lc.call_type,
                       COUNT(*) as call_count,
                       COALESCE(AVG(lc.latency_ms), 0) as avg_latency_ms,
                       COALESCE(SUM(lc.input_tokens), 0) as total_input_tokens,
                       COALESCE(SUM(lc.output_tokens), 0) as total_output_tokens,
                       COALESCE(SUM(lc.retry_count), 0) as total_retries,
                       COALESCE(SUM(lc.cache_hit), 0) as total_cache_hits,
                       COALESCE(SUM(lc.cached_tokens), 0) as total_cached_tokens
                FROM llm_call_log lc
                JOIN query_diagnostics qd ON lc.diagnostic_id = qd.id
                WHERE {where}
                GROUP BY lc.call_type
                ORDER BY call_count DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def record_feedback(
        self,
        query: str,
        rating: int,
        doc_id: str = None,
        doc_title: str = None,
        index_path: str = None,
        session_id: str = None,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO search_feedback
               (query, doc_id, doc_title, rating, index_path, session_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (query, doc_id, doc_title, rating, index_path, session_id),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_feedback_summary(self, days: int = 7) -> dict:
        where = "1=1"
        params: list = []
        if days:
            where += " AND created_at >= datetime('now', ?)"
            params.append(f"-{days} days")

        row = self.conn.execute(
            f"""SELECT
                   COUNT(*) as total,
                   COALESCE(SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END), 0) as total_up,
                   COALESCE(SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END), 0) as total_down
                FROM search_feedback WHERE {where}""",
            params,
        ).fetchone()

        worst_rows = self.conn.execute(
            f"""SELECT doc_title,
                       COUNT(*) as down_count,
                       SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as up_count
                FROM search_feedback
                WHERE {where} AND doc_title IS NOT NULL
                GROUP BY doc_title
                ORDER BY down_count DESC
                LIMIT 10""",
            params,
        ).fetchall()

        recent_rows = self.conn.execute(
            f"""SELECT query, doc_id, doc_title, rating, index_path,
                      session_id, created_at
                FROM search_feedback WHERE {where}
                ORDER BY created_at DESC LIMIT 100""",
            params,
        ).fetchall()

        r = row if row else {}
        return {
            "total": r["total"] if r else 0,
            "total_up": r["total_up"] if r else 0,
            "total_down": r["total_down"] if r else 0,
            "worst_rated_docs": [dict(w) for w in worst_rows],
            "recent": [dict(rr) for rr in recent_rows],
        }

    # ── Auth Audit Log ──────────────────────────────────────────

    def record_auth_log(
        self,
        endpoint: str,
        method: str,
        token_id: str = None,
        client_ip: str = None,
        status_code: int = 200,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO auth_log
               (token_id, endpoint, method, client_ip, status_code)
               VALUES (?, ?, ?, ?, ?)""",
            (token_id, endpoint, method, client_ip, status_code),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_auth_log(
        self,
        days: int = 7,
        token_id: str = None,
        limit: int = 100,
    ) -> list[dict]:
        where = "1=1"
        params: list = []
        if days:
            where += " AND created_at >= datetime('now', ?)"
            params.append(f"-{days} days")
        if token_id:
            where += " AND token_id = ?"
            params.append(token_id)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT id, token_id, endpoint, method, client_ip,
                       status_code, created_at
                FROM auth_log WHERE {where}
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
