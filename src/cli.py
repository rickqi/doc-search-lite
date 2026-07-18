"""doc-search CLI - 本地文档搜索系统命令行接口"""

import contextlib
import sys

# Fix Windows console encoding for Chinese and emoji output.
# Wrapped in try/except for headless environments (Task Scheduler, CI, etc.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # stdout may not be a real console (e.g., Task Scheduler)
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
from dotenv import load_dotenv

# Rich imports with fallback
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme

    _doc_search_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "result": "white",
        "score": "yellow",
        "source": "dim cyan",
        "tool": "magenta",
        "query": "bold yellow",
    })
    console: Console | None = Console(theme=_doc_search_theme)
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    console = None

import contextlib

from src.converter.base import ConvertResult
from src.converter.coordinator import ConverterCoordinator
from src.converter.csv import CSVConverter
from src.converter.image import ImageConverter
from src.converter.ocr import OCRServiceConfig
from src.converter.text import TextConverter
from src.search.bm25_search import create_searcher
from src.search.result_formatter import ResultFormatter, SearchResult
from src.stats.search_logger import SearchLogger
from src.storage.base import DocumentRecord
from src.storage.convert_db import ConvertDB
from src.storage.index import TantivyIndexManager
from src.storage.markdown_store import MarkdownStore
from src.storage.metadata import MetadataManager
from src.storage.raw_store import RawStore
from src.utils.dir_diff import (
    compare_directories,
)
from src.utils.file_watcher import ChangeSet, FileWatcher
from src.utils.hash import calculate_hash

logger = logging.getLogger(__name__)


def _get_version() -> str:
    """Read version from pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith("version ="):
                return line.split('"')[1]
    except Exception:
        logger.warning("Failed to read version from pyproject.toml, using default")
    return "0.2.0.dev0"


def _log_search_cli(
    query_text: str,
    response: Any,
    search_mode: str,
    index_path: str = "",
    raw_dir: str = "",
):
    """Fire-and-forget search logging for CLI — respects NO_SEARCH_LOG env."""
    if os.environ.get("NO_SEARCH_LOG"):
        return
    try:
        SearchLogger.log_async(
            session_id=SearchLogger.generate_session_id(),
            query=query_text,
            response=response,
            source="cli",
            search_mode=search_mode,
            index_path=index_path,
            raw_dir=raw_dir,
        )
        SearchLogger.flush(timeout=5.0)
    except Exception:
        logger.warning("Failed to log search, query=%s, search_mode=%s", query_text, search_mode)



# Load .env from project root (override=False: real env vars take precedence)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


# Simple in-memory task storage
class TaskManager:
    """Simple task manager for tracking conversion tasks."""

    _instance: Optional["TaskManager"] = None

    def __init__(self, task_dir: Path | None = None):
        self.task_dir = task_dir or Path.cwd() / ".doc-search" / "tasks"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, dict] = self._load_tasks()

    @classmethod
    def get_instance(cls, task_dir: Path | None = None) -> "TaskManager":
        if cls._instance is None:
            cls._instance = cls(task_dir)
        return cls._instance

    def _load_tasks(self) -> dict[str, dict]:
        """Load tasks from disk."""
        tasks_file = self.task_dir / "tasks.json"
        if tasks_file.exists():
            try:
                return json.loads(tasks_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_tasks(self) -> bool:
        """Save tasks to disk."""
        tasks_file = self.task_dir / "tasks.json"
        try:
            tasks_file.write_text(
                json.dumps(self._tasks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return True
        except OSError:
            return False

    def create_task(self, task_type: str, params: dict) -> str:
        """Create a new task and return its ID."""
        task_id = f"{task_type}_{int(time.time() * 1000)}"
        self._tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "params": params,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "progress": 0,
            "total": 0,
            "errors": [],
            "result": None,
        }
        self._save_tasks()
        return task_id

    def update_task(self, task_id: str, **kwargs) -> bool:
        """Update task fields."""
        if task_id not in self._tasks:
            return False
        self._tasks[task_id].update(kwargs)
        self._tasks[task_id]["updated_at"] = datetime.now().isoformat()
        return self._save_tasks()

    def get_task(self, task_id: str) -> dict | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self, status: str | None = None) -> list[dict]:
        """List all tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status and status != "all":
            tasks = [t for t in tasks if t.get("status") == status]
        return sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return self._save_tasks()
        return False


@click.group()
@click.version_option(version=_get_version())
def cli():
    """本地文档搜索系统 - 将文档转换为可搜索的索引并进行智能查询"""
    pass


def _collect_files(source: Path, formats: tuple, recursive: bool = True) -> list[Path]:
    """Collect files from source directory based on formats."""
    files = []
    extensions = set(
        f.lower() if f.startswith(".") else f".{f.lower()}" for f in formats
    )

    if source.is_file():
        if not formats or source.suffix.lower() in extensions:
            files.append(source)
    elif source.is_dir():
        if recursive:
            for f in source.rglob("*"):
                if f.is_file() and (not formats or f.suffix.lower() in extensions):
                    files.append(f)
        else:
            for f in source.iterdir():
                if f.is_file() and (not formats or f.suffix.lower() in extensions):
                    files.append(f)

    return files


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# ── 文件过滤常量 ──────────────────────────────

SKIP_EXTENSIONS = {
    '.crdownload', '.js', '.css', '.woff', '.ttf',
    '.m4a', '.mp3', '.wav', '.mp4',
    '.xmind', '.egg-info', '.pyc', '.pyo',
}

SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.xls',
    '.html', '.htm', '.csv', '.txt',
    '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif',
    '.msg',
    '.zip', '.7z', '.rar', '.tar', '.gz', '.tgz', '.bz2', '.xz',
}

# Large file threshold: skip hashing for files > 50MB
_LARGE_FILE_THRESHOLD = 50 * 1024 * 1024


def _scan_and_sync(db: ConvertDB, source_root: Path) -> int:
    """扫描源目录并同步文件记录到 SQLite。

    Returns:
        待处理文件数量
    """
    # 1. Walk source_root recursively
    dir_cache: dict[str, int] = {}  # relative_path -> dir_id

    # Register root directory
    root_id = db.upsert_directory(".", parent_id=None, depth=0, name=source_root.name)
    dir_cache["."] = root_id

    for dirpath, _dirnames, filenames in os.walk(source_root):
        dir_path = Path(dirpath)
        rel_dir = dir_path.relative_to(source_root)
        rel_dir_str = str(rel_dir).replace("\\", "/")
        if rel_dir_str == ".":
            rel_dir_str = "."

        # 2. For each directory: db.upsert_directory()
        if rel_dir_str not in dir_cache:
            parent_rel = str(rel_dir.parent).replace("\\", "/")
            if parent_rel == ".":
                parent_rel = "."
            parent_id = dir_cache.get(parent_rel, root_id)
            depth = len(rel_dir.parts)
            dir_id = db.upsert_directory(
                rel_dir_str, parent_id=parent_id, depth=depth, name=dir_path.name
            )
            dir_cache[rel_dir_str] = dir_id
        else:
            dir_id = dir_cache[rel_dir_str]

        dir_file_count = 0
        dir_total_size = 0

        for fname in filenames:
            file_path = dir_path / fname
            ext = file_path.suffix.lower()
            rel_path = str(file_path.relative_to(source_root)).replace("\\", "/")
            try:
                file_stat = file_path.stat()
                file_size = file_stat.st_size
            except OSError:
                continue

            dir_file_count += 1
            dir_total_size += file_size

            # Skip known useless extensions
            if ext in SKIP_EXTENSIONS:
                continue

            # 3. Supported extensions: upsert with hash
            if ext in SUPPORTED_EXTENSIONS:
                mtime_iso = datetime.fromtimestamp(
                    file_stat.st_mtime
                ).isoformat()

                # For large files, skip hashing and use mtime only
                if file_size > _LARGE_FILE_THRESHOLD:
                    file_hash = f"mtime:{mtime_iso}"
                else:
                    # Check if file already exists and mtime hasn't changed
                    existing = db.get_file(rel_path)
                    if existing and existing.get("source_mtime") == mtime_iso:
                        file_hash = existing.get("source_hash", "")
                    else:
                        try:
                            file_hash = calculate_hash(file_path)
                        except OSError:
                            file_hash = ""

                db.upsert_file(
                    relative_path=rel_path,
                    directory_id=dir_id,
                    filename=fname,
                    extension=ext,
                    file_size=file_size,
                    source_mtime=mtime_iso,
                    source_hash=file_hash,
                )
            else:
                # 4. Unsupported extension: mark as skipped
                file_id = db.upsert_file(
                    relative_path=rel_path,
                    directory_id=dir_id,
                    filename=fname,
                    extension=ext,
                    file_size=file_size,
                    source_mtime="",
                    source_hash="",
                )
                db.mark_file_skipped(file_id, reason="unsupported_format", detail=ext)

        # 5. Update directory stats
        if dir_file_count > 0:
            db.update_directory_stats(dir_id, dir_file_count, dir_total_size)

    # 6. Return count of pending files
    return db.count_files("pending")


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), default="./output", help="输出目录路径"
)
@click.option(
    "--formats",
    "-f",
    multiple=True,
    default=[],
    help="指定转换格式 (可多次指定, 如: pdf,docx)",
)
@click.option("--ocr/--no-ocr", default=True, help="启用或禁用OCR识别")
@click.option("--resume", is_flag=True, help="恢复上次中断的转换任务")
@click.option("--force", is_flag=True, help="强制重新转换,跳过缓存")
@click.option("--dry-run", is_flag=True, help="模拟运行,不实际执行")
@click.option("--parallel", "-p", type=int, default=1, help="并行处理文档数量")
@click.option("--index/--no-index", default=True, help="转换后是否创建搜索索引")
def convert(source, output, formats, ocr, resume, force, dry_run, parallel, index):
    """将源文档转换为可搜索格式

    支持的格式: PDF, DOCX, XLSX, PPTX, HTML等

    示例:
        doc-search convert ./docs -o ./output
        doc-search convert ./input -o ./output --formats pdf,docx --parallel 4
    """
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()

    click.echo(f"📂 源目录: {source_path}")
    click.echo(f"📁 输出目录: {output_path}")

    if dry_run:
        click.echo("🔍 模拟运行模式 - 不执行实际转换")

    # Collect files to convert
    files = _collect_files(source_path, formats)

    if not files:
        click.echo("❌ 未找到匹配的文件")
        return

    click.echo(f"📄 找到 {len(files)} 个文件待转换")

    if dry_run:
        for f in files[:10]:  # Show first 10
            click.echo(f"  - {f.relative_to(source_path)}")
        if len(files) > 10:
            click.echo(f"  ... 还有 {len(files) - 10} 个文件")
        return

    # Initialize components
    output_path.mkdir(parents=True, exist_ok=True)
    index_path = output_path / "index"
    coordinator = ConverterCoordinator(enable_ocr_fallback=ocr)
    store = MarkdownStore(input_base=source_path, output_base=output_path)
    index_manager = TantivyIndexManager(index_path=index_path) if index else None
    metadata_manager = MetadataManager(index_path=output_path / "metadata.json")

    # Create task
    task_manager = TaskManager.get_instance(output_path / ".tasks")
    task_id = task_manager.create_task(
        "convert",
        {
            "source": str(source_path),
            "output": str(output_path),
            "formats": list(formats),
            "total_files": len(files),
        },
    )
    task_manager.update_task(task_id, status="running", total=len(files))

    # Track results
    success_count = 0
    failed_count = 0
    skipped_count = 0

    # Convert files with progress bar
    with click.progressbar(files, label="转换中") as bar:
        for i, file_path in enumerate(bar):
            try:
                # Check if already converted (unless force)
                if not force and store.exists_by_source(file_path):
                    skipped_count += 1
                    task_manager.update_task(task_id, progress=i + 1)
                    continue

                # Convert document
                result: ConvertResult = coordinator.convert(
                    source=file_path,
                    output_dir=output_path,
                    options={"disable_ocr_fallback": not ocr},
                )

                if result.success:
                    # Create document record
                    record = DocumentRecord(
                        id=store._generate_doc_id(file_path),
                        source_path=file_path,
                        output_path=store.get_output_path(file_path),
                        title=file_path.stem,
                        content_hash=calculate_hash(file_path),
                        file_size=file_path.stat().st_size,
                        file_mtime=datetime.fromtimestamp(file_path.stat().st_mtime),
                        metadata={
                            "converter": result.converter_name,
                            "convert_time": result.convert_time,
                            "ocr_used": result.ocr_used,
                            **result.metadata,
                        },
                    )

                    # Save to store
                    if result.images:
                        store.save_with_images(record, result.markdown, result.images)
                    else:
                        store.save(record, result.markdown)

                    # Add to index
                    if index_manager:
                        index_manager.add_document(
                            doc_id=record.id,
                            title=record.title,
                            content=result.markdown,
                            metadata={
                                "filename": file_path.name,
                                "source_path": str(file_path),
                                "modified_time": record.file_mtime,
                            },
                        )

                    # Save metadata
                    metadata_manager.save(
                        file_path,
                        {
                            "source_path": file_path,
                            "output_path": record.output_path,
                            "content_hash": record.content_hash,
                            "modified_time": file_path.stat().st_mtime,
                            "convert_time": result.convert_time,
                        },
                    )

                    success_count += 1
                else:
                    failed_count += 1
                    current_task = task_manager.get_task(task_id)
                    current_errors = (
                        current_task.get("errors", []) if current_task else []
                    )
                    task_manager.update_task(
                        task_id,
                        errors=current_errors
                        + [f"{file_path.name}: {'; '.join(result.errors)}"],
                    )

                task_manager.update_task(task_id, progress=i + 1)

            except Exception as e:
                failed_count += 1
                current_task = task_manager.get_task(task_id)
                current_errors = current_task.get("errors", []) if current_task else []
                task_manager.update_task(
                    task_id,
                    errors=current_errors + [f"{file_path.name}: {str(e)}"],
                )

    # Commit index
    if index_manager:
        index_manager.commit()

    # Update task status
    task_manager.update_task(
        task_id,
        status="completed",
        result={
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
        },
    )

    # Print summary
    click.echo("\n📊 转换完成:")
    click.echo(f"  ✅ 成功: {success_count}")
    click.echo(f"  ⏭️ 跳过: {skipped_count}")
    click.echo(f"  ❌ 失败: {failed_count}")

    if index:
        stats = index_manager.get_stats() if index_manager else {}
        click.echo(f"  📇 索引文档数: {stats.get('num_docs', 0)}")

    if failed_count > 0:
        sys.exit(1)


def _is_tantivy_index(path: Path) -> bool:
    """Check if a directory contains a valid Tantivy index.

    Detects by looking for Tantivy segment files (.store).
    """
    return any(path.glob("*.store"))


def _parse_index_paths(index_arg):
    """Parse comma-separated index path string into list of resolved Paths.

    Args:
        index_arg: Comma-separated path string, or None for default.

    Returns:
        List of resolved Path objects.

    Raises:
        click.BadParameter: If any path does not exist.
    """
    if not index_arg:
        return [Path.cwd() / "output" / "index"]
    paths = [Path(p.strip()).resolve() for p in index_arg.split(",")]
    for p in paths:
        if not p.exists():
            raise click.BadParameter(f"路径不存在: {p}")
    return paths


def _export_search_results(
    results: list[dict[str, Any]],
    export_path: Path,
    query_text: str = "",
    sources_items: list[dict] | None = None,
) -> None:
    """Export search results to file (JSON/CSV/Markdown).

    Format auto-detected from file extension.
    """
    export_path = Path(export_path)
    suffix = export_path.suffix.lower()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Build structured result data
    export_data = {
        "query": query_text,
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results),
        "results": results,
    }
    if sources_items:
        export_data["sources"] = sources_items

    if suffix == ".json":
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        click.echo(f"📄 已导出 JSON: {export_path}")

    elif suffix == ".csv":
        import csv
        with open(export_path, "w", encoding="utf-8-sig", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        click.echo(f"📄 已导出 CSV: {export_path}")

    elif suffix == ".md":
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"# 搜索结果: {query_text}\n\n")
            f.write(f"**时间**: {export_data['timestamp']}  \n")
            f.write(f"**结果数**: {len(results)}\n\n---\n\n")
            for i, r in enumerate(results, 1):
                f.write(f"## {i}. {r.get('title', '无标题')}\n\n")
                f.write(f"- **评分**: {r.get('score', 'N/A')}\n")
                f.write(f"- **来源**: {r.get('source', 'N/A')}\n\n")
                snippet = r.get("snippet", "")
                if snippet:
                    f.write(f"{snippet}\n\n")
                f.write("---\n\n")
        click.echo(f"📄 已导出 Markdown: {export_path}")

    else:
        click.echo(f"⚠️  不支持的导出格式: {suffix} (支持的格式: .json, .csv, .md)")


@cli.command()
@click.argument("query_text")
@click.option("--index", "-i", type=str, help="指定索引目录 (多个用逗号分隔)")
@click.option("--limit", "-l", type=int, default=10, help="返回结果数量限制")
@click.option("--agent/--no-agent", default=False, help="是否使用AI代理增强查询")
@click.option(
    "--agent-mode",
    type=click.Choice(["auto", "semantic", "precision"]),
    default="semantic",
    help="AI代理模式",
)
@click.option("--model", type=str, help="指定AI模型")
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="输出格式",
)
@click.option("--interactive", "-I", is_flag=True, help="交互式查询模式")
@click.option("--sources", type=str, help="限制搜索来源")
@click.option(
    "--sources-detail",
    type=click.Choice(["none", "path", "file", "full"]),
    default="file",
    help="引用文件详情级别: none=不显示, path=仅路径, file=路径+文件名(默认), full=路径+文件名+搜索类型",
)
@click.option("--rerank", is_flag=True, help="启用 Rerank 重排序 (tool_loop 模式自动使用, pipeline 模式需显式启用)")
@click.option("--mode", type=click.Choice(["tool_loop", "pipeline"]), default="tool_loop",
              help="Agent模式 (tool_loop=LLM自主搜索, pipeline=固定流程)")
@click.option("--search-mode", type=click.Choice(["auto", "bm25", "grep", "hybrid", "tag"]),
              default="auto", help="搜索模式 (auto=自动检测, hybrid=BM25+Grep融合, tag=标签匹配召回)")
@click.option("--skill", type=click.Choice(["summarize", "compare", "extract-table", "detailed", "timeline", "action-items"]),
              default=None, help="分析技能 (搜索结果后处理模式)")
@click.option("--load-skill", type=str, default=None,
              help="加载外部 SKILL.md 技能文件 (技能名或文件路径)")
@click.option("--export", "-e", "export_path", type=click.Path(dir_okay=False, writable=True), default=None,
              help="导出搜索结果到文件 (根据扩展名自动识别格式: .json/.csv/.md)")
@click.option("--no-log", is_flag=True, help="不记录搜索日志")
def query(
    query_text,
    index,
    limit,
    agent,
    agent_mode,
    model,
    output_format,
    interactive,
    sources,
    sources_detail,
    rerank,
    mode,
    search_mode,
    skill,
    load_skill,
    export_path,
    no_log,
):
    """搜索文档内容

    不使用--agent: 关键词搜索(BM25 或 Grep 自动选择)
    使用--agent: AI语义问答

    路径自动检测:
        - 指向 index/ 子目录 → BM25 全文搜索 (快速, 有评分排序)
        - 指向 Markdown 父目录 → Grep 直接搜索原始文件 (无需索引)
        - 多路径逗号分隔 → 跨索引搜索

    示例:
        doc-search query "绩效考核流程" -i ./output/index
        doc-search query "年假如何申请？" -i ./output --agent
        doc-search query "年假" -i ./output  (自动检测: 无索引则用 Grep)
        doc-search query "年假" -i ./output --search-mode hybrid
        doc-search query "制度" -i "./raw/公司规章制度/index,./raw/DLP/index"
    """
    # Honor --no-log by setting env var (checked by _log_search_cli helper)
    if no_log:
        os.environ["NO_SEARCH_LOG"] = "1"

    # Parse index paths (supports comma-separated multi-index)
    try:
        index_paths = _parse_index_paths(index)
    except click.BadParameter as e:
        click.echo(f"❌ {e}")
        sys.exit(1)

    # Single-index path for backward compatibility
    index_path = index_paths[0]

    if not index_path.exists():
        click.echo(f"❌ 目录不存在: {index_path}")
        sys.exit(1)

    # Auto-detect: is this a Tantivy index or a raw markdown directory?
    is_tantivy_index = _is_tantivy_index(index_path)

    if interactive:
        _interactive_query_loop(
            index_path, limit, agent, agent_mode, output_format, model, sources,
            use_rerank=rerank,
            search_mode=mode,
            skill=skill,
            sources_detail=sources_detail,
        )
    elif agent:
        # Use SearchAgent for semantic Q&A
        _query_with_agent(
            query_text, index_path, limit, agent_mode, output_format, model,
            use_rerank=rerank,
            mode=mode,
            skill=skill,
            load_skill=load_skill,
            sources_detail=sources_detail,
            export_path=export_path,
        )
    elif len(index_paths) > 1:
        # Multi-index search
        _query_with_multi_index(query_text, index_paths, limit, output_format, sources_detail, export_path=export_path)
    elif search_mode == "hybrid" and is_tantivy_index:
        # Hybrid BM25 + Grep RRF fusion
        _query_with_hybrid(query_text, index_path, limit, output_format, sources_detail, export_path=export_path)
    elif search_mode == "tag":
        # Tag-based recall: extract tags from query, then search
        _query_with_tag(query_text, index_path, limit, output_format, sources_detail, export_path=export_path)
    else:
        if search_mode == "bm25" or (search_mode == "auto" and is_tantivy_index):
            # BM25 full-text search (fast, scored)
            _query_with_bm25(query_text, index_path, limit, output_format, sources, sources_detail, export_path=export_path)
        else:
            # Grep raw markdown files (no index needed)
            _query_with_grep(query_text, index_path, limit, output_format, sources_detail, export_path=export_path)

    # Ensure async search log writes complete before process exits
    SearchLogger.flush(timeout=5.0)


def _show_interactive_help():
    """Show available commands in interactive mode."""
    if _RICH_AVAILABLE:
        console.print()

        # Build a table for commands
        table = Table(
            title="📖 可用命令",
            show_header=False,
            border_style="dim",
            padding=(0, 2),
            expand=False,
        )
        table.add_column("Command", style="info", no_wrap=True, min_width=28)
        table.add_column("Description", style="dim")

        table.add_row("help", "显示此帮助信息")
        table.add_row("mode bm25", "切换到BM25关键词搜索模式")
        table.add_row("mode agent", "切换到AI语义搜索模式")
        table.add_row("limit N", "设置结果数量 (例如: limit 5)")
        table.add_row("format text", "设置输出格式: text/json/markdown")
        table.add_row("search-mode tool_loop", "切换到LLM自主搜索模式")
        table.add_row("search-mode pipeline", "切换到固定流程模式")
        table.add_row("skill <name>", "设置分析技能: summarize/compare/extract-table/detailed/timeline/action-items")
        table.add_row("skill off", "关闭分析技能")
        table.add_row("quit / exit / q", "退出交互模式")
        console.print(table)

        console.print()
        console.print("💡 [dim]直接输入搜索词或问题即可搜索[/dim]")
        console.print()
    else:
        click.echo("")
        click.echo("📖 可用命令:")
        click.echo("  help              显示此帮助信息")
        click.echo("  mode bm25         切换到BM25关键词搜索模式")
        click.echo("  mode agent        切换到AI语义搜索模式")
        click.echo("  limit N           设置结果数量 (例如: limit 5)")
        click.echo("  format text       设置输出格式: text/json/markdown")
        click.echo("  search-mode tool_loop  切换到LLM自主搜索模式")
        click.echo("  search-mode pipeline   切换到固定流程模式")
        click.echo("  skill <name>     设置分析技能: summarize/compare/extract-table/detailed/timeline/action-items")
        click.echo("  skill off        关闭分析技能")
        click.echo("  quit / exit / q   退出交互模式")
        click.echo("")
        click.echo("💡 直接输入搜索词或问题即可搜索")
        click.echo("")


def _interactive_query_loop(
    index_path: Path,
    limit: int,
    agent: bool,
    agent_mode: str,
    output_format: str,
    model: str | None,
    sources: str | None,
    use_rerank: bool = False,
    search_mode: str = "tool_loop",
    skill: str | None = None,
):
    """Interactive query REPL mode."""
    current_mode = "agent" if agent else "bm25"
    current_search_mode = search_mode
    current_limit = limit
    current_format = output_format
    current_skill = skill

    if _RICH_AVAILABLE:
        mode_label = "🤖 AI语义搜索" if current_mode == "agent" else "🔍 BM25关键词搜索"
        console.print(
            Panel(
                f"[bold]{mode_label}[/bold]\n\n"
                f"[dim]输入搜索词或问题开始搜索\n"
                f"输入 [info]help[/info] 查看命令列表[/dim]",
                title="📚 文档搜索交互模式",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        console.print()
    else:
        click.echo("🔍 文档搜索交互模式 (输入 'quit' 或 'exit' 退出)")
        click.echo(
            f"   模式: {'AI语义搜索' if current_mode == 'agent' else 'BM25关键词搜索'}"
        )
        click.echo("   输入 'help' 查看命令\n")

    while True:
        try:
            if _RICH_AVAILABLE:
                # Show mode badge in prompt
                mode_tag = "[bold cyan]🤖[/bold cyan]" if current_mode == "agent" else "[bold yellow]🔍[/bold yellow]"
                console.print(f"{mode_tag} ", end="")
                user_input = click.prompt("", default="", show_default=False)
            else:
                user_input = click.prompt("> ", default="", show_default=False)
        except (EOFError, KeyboardInterrupt):
            if _RICH_AVAILABLE:
                console.print("\n👋 [success]再见![/success]")
            else:
                click.echo("\n👋 再见!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        lower = user_input.lower()

        # Handle special commands
        if lower in ("quit", "exit", "q"):
            if _RICH_AVAILABLE:
                console.print("👋 [success]再见![/success]")
            else:
                click.echo("👋 再见!")
            break
        elif lower == "help":
            _show_interactive_help()
            continue
        elif lower.startswith("mode "):
            mode_arg = user_input[5:].strip().lower()
            if mode_arg in ("bm25", "agent"):
                current_mode = mode_arg
                mode_label = "AI语义搜索" if mode_arg == "agent" else "BM25关键词搜索"
                if _RICH_AVAILABLE:
                    console.print(f"✅ [success]切换到 {mode_label} 模式[/success]")
                else:
                    click.echo(
                        f"✅ 切换到 {mode_label} 模式"
                    )
            else:
                if _RICH_AVAILABLE:
                    console.print("❌ [error]可用模式: bm25, agent[/error]")
                else:
                    click.echo("❌ 可用模式: bm25, agent")
            continue
        elif lower.startswith("limit "):
            try:
                current_limit = int(user_input[6:].strip())
                if _RICH_AVAILABLE:
                    console.print(f"✅ [success]结果数量: {current_limit}[/success]")
                else:
                    click.echo(f"✅ 结果数量: {current_limit}")
            except ValueError:
                if _RICH_AVAILABLE:
                    console.print("❌ [error]请输入数字，例如: limit 5[/error]")
                else:
                    click.echo("❌ 请输入数字，例如: limit 5")
            continue
        elif lower.startswith("format "):
            fmt = user_input[7:].strip().lower()
            if fmt in ("text", "json", "markdown"):
                current_format = fmt
                if _RICH_AVAILABLE:
                    console.print(f"✅ [success]输出格式: {fmt}[/success]")
                else:
                    click.echo(f"✅ 输出格式: {fmt}")
            else:
                if _RICH_AVAILABLE:
                    console.print("❌ [error]可用格式: text, json, markdown[/error]")
                else:
                    click.echo("❌ 可用格式: text, json, markdown")
            continue
        elif lower.startswith("search-mode "):
            sm = user_input[12:].strip()
            if sm in ("tool_loop", "pipeline"):
                current_search_mode = sm
                if _RICH_AVAILABLE:
                    console.print(f"✅ [success]搜索模式: {sm}[/success]")
                else:
                    click.echo(f"✅ 搜索模式: {sm}")
            else:
                if _RICH_AVAILABLE:
                    console.print("❌ [error]可用模式: tool_loop, pipeline[/error]")
                else:
                    click.echo("❌ 可用模式: tool_loop, pipeline")
            continue
        elif lower.startswith("skill "):
            skill_arg = user_input[6:].strip().lower()
            valid_skills = ["summarize", "compare", "extract-table", "detailed", "timeline", "action-items"]
            if skill_arg == "off":
                current_skill = None
                if _RICH_AVAILABLE:
                    console.print("✅ [success]已关闭分析技能[/success]")
                else:
                    click.echo("✅ 已关闭分析技能")
            elif skill_arg in valid_skills:
                current_skill = skill_arg
                if _RICH_AVAILABLE:
                    console.print(f"✅ [success]分析技能: {skill_arg}[/success]")
                else:
                    click.echo(f"✅ 分析技能: {skill_arg}")
            else:
                if _RICH_AVAILABLE:
                    console.print(f"❌ [error]可用技能: {', '.join(valid_skills)}, off[/error]")
                else:
                    click.echo(f"❌ 可用技能: {', '.join(valid_skills)}, off")
            continue

        # Execute search
        if current_mode == "agent":
            _query_with_agent(
                user_input, index_path, current_limit, agent_mode, current_format, model,
                use_rerank=use_rerank,
                mode=current_search_mode,
                skill=current_skill,
            )
        else:
            _query_with_bm25(
                user_input, index_path, current_limit, current_format, sources
            )

        if _RICH_AVAILABLE:
            console.print()
        else:
            click.echo()  # Blank line between queries


def _truncate_source(source_path: Path, max_parents: int = 3) -> str:
    """Truncate source path to show only last N parent directories + filename."""
    parts = source_path.parts
    if len(parts) > max_parents + 1:
        return ".../" + "/".join(parts[-(max_parents + 1):])
    return str(source_path)


def _extract_filename(source_path) -> str:
    """Extract clean filename from a source path, stripping .md suffix if present."""
    if source_path is None:
        return "未知"
    path = Path(source_path)
    name = path.name
    # Strip trailing .md that was added during conversion (e.g. "报告.docx.md" → "报告.docx")
    if name.endswith(".md"):
        name = name[:-3]
    return name


def _format_sources_data(
    items: list,
    sources_detail: str = "file",
) -> list:
    """Format sources data for JSON/text output.

    Args:
        items: List of dicts, each with at minimum 'source_path'.
               Optional: 'title', 'search_type', 'score', 'line', 'index_name'.
        sources_detail: Detail level - 'none', 'path', 'file', 'full'.

    Returns:
        List of formatted source dicts, deduplicated by source_path.
    """
    if sources_detail == "none":
        return []

    seen = set()
    result = []
    for item in items:
        sp = item.get("source_path", "")
        if sp in seen:
            continue
        seen.add(sp)

        entry = {}
        if sources_detail in ("path", "file", "full"):
            entry["path"] = str(sp)

        if sources_detail in ("file", "full"):
            entry["filename"] = _extract_filename(sp)
            if item.get("title"):
                entry["title"] = item["title"]

        if sources_detail == "full":
            if item.get("search_type"):
                entry["search_type"] = item["search_type"]
            if item.get("score"):
                entry["score"] = round(item["score"], 4)
            if item.get("line"):
                entry["line"] = item["line"]
            if item.get("index_name"):
                entry["index_name"] = item["index_name"]

        if entry:
            result.append(entry)
    return result


def _display_sources_panel(
    items: list,
    sources_detail: str = "file",
    search_type_label: str = "",
):
    """Display a unified sources/references panel after search results.

    Args:
        items: List of dicts with 'source_path'. Optional: 'title', 'search_type', 'line'.
        sources_detail: 'none' to hide, 'path' for paths only, 'file' for path+filename, 'full' for all details.
        search_type_label: Override label for search type (e.g. "BM25", "Hybrid", "Grep").
    """
    if sources_detail == "none" or not items:
        return

    # Deduplicate by source_path
    seen = set()
    unique = []
    for item in items:
        sp = str(item.get("source_path", ""))
        if sp and sp not in seen:
            seen.add(sp)
            unique.append(item)
    if not unique:
        return

    if _RICH_AVAILABLE:
        lines = []
        for idx, item in enumerate(unique, 1):
            sp = item.get("source_path", "")
            filename = _extract_filename(sp)
            truncated = _truncate_source(Path(sp)) if sp else "未知"

            if sources_detail == "path":
                lines.append(f"[{idx}] {truncated}")
            elif sources_detail == "file":
                lines.append(f"[{idx}] [bold]{filename}[/bold]  [dim]{truncated}[/dim]")
            elif sources_detail == "full":
                st = item.get("search_type", search_type_label)
                type_tag = f"[bold cyan][{st}][/bold cyan] " if st else ""
                score_str = f" ({item['score']:.2f})" if item.get("score") else ""
                line_str = f":L{item['line']}" if item.get("line") else ""
                lines.append(
                    f"[{idx}] {type_tag}[bold]{filename}[/bold]{score_str}{line_str}  [dim]{truncated}[/dim]"
                )

        console.print()
        console.print(
            Panel(
                "\n".join(lines),
                title="📚 引用文件",
                border_style="cyan",
                padding=(0, 1),
            )
        )
    else:
        click.echo()
        click.echo("📚 引用文件:")
        for idx, item in enumerate(unique, 1):
            sp = item.get("source_path", "")
            filename = _extract_filename(sp)
            truncated = _truncate_source(Path(sp)) if sp else "未知"

            if sources_detail == "path":
                click.echo(f"  [{idx}] {truncated}")
            elif sources_detail == "file":
                click.echo(f"  [{idx}] {filename}  ({truncated})")
            elif sources_detail == "full":
                st = item.get("search_type", search_type_label)
                score_str = f" ({item['score']:.2f})" if item.get("score") else ""
                line_str = f":L{item['line']}" if item.get("line") else ""
                click.echo(f"  [{idx}] [{st}] {filename}{score_str}{line_str}  ({truncated})")


def _execute_list_search(
    strategy: str,
    query_text: str,
    path: Path,
    limit: int,
    output_format: str,
    sources_detail: str = "file",
    export_path: Path | None = None,
    index_paths: list | None = None,
):
    """Unified dispatcher for list-mode search strategies (bm25/grep/hybrid/tag/multi_index)."""
    _META = {
        "bm25":        ("BM25", "cyan", "🔍 BM25 搜索"),
        "grep":        ("Grep", "yellow", "🔍 Grep 搜索 (无索引, 直接搜索原始文件)"),
        "hybrid":      ("Hybrid", "magenta", "🔍 Hybrid 搜索 (BM25 + Grep RRF 融合)"),
        "tag":         ("Tag", "green", "🏷️ Tag 召回搜索"),
        "multi_index": ("Multi-Index", "green", None),
    }
    meta = _META.get(strategy, (strategy, "cyan", f"🔍 {strategy} 搜索"))
    label, border_style, panel_title = meta

    start_time = time.time()
    try:
        sources_items: list = []
        display_rows: list = []
        json_output: dict = {}
        summary_parts: list = []
        pre_lines: list = []
        post_lines: list = []
        execution_time = 0.0
        grep_style = False
        markdown_text = None
        log_extra: dict = {}
        log_path_key = "index_path"
        log_path_val = str(path)

        # ── Strategy-specific execution + normalization ──

        if strategy == "bm25":
            searcher = create_searcher(index_path=path, use_jieba=True, readonly=True)
            results = searcher.search(query_text, limit=limit)
            execution_time = results.execution_time

            if results.results:
                formatter = ResultFormatter(highlight_pattern=query_text)
                search_results_objs = [
                    SearchResult(
                        title=p.title, score=p.score, snippet=p.snippet,
                        source=p.source_path or Path(""),
                        metadata={"doc_id": p.doc_id},
                    )
                    for p in results.results
                ]
                for p in results.results:
                    sources_items.append({
                        "source_path": str(p.source_path) if p.source_path else "",
                        "title": p.title, "score": p.score, "search_type": "BM25",
                        "highlights": getattr(p, "highlights", []),
                        "snippet": getattr(p, "snippet", ""),
                    })
                    display_rows.append({
                        "idx": len(display_rows) + 1,
                        "title": p.title,
                        "score_text": f"{p.score:.2f}",
                        "source_display": _truncate_source(p.source_path) if p.source_path else "未知",
                        "snippet": formatter.highlight_text(p.snippet, max_length=120),
                    })
                summary_parts = [f"共 {len(results.results)} 条结果"]

                fmt_json = formatter.format_json(search_results_objs, include_summary=True)
                json_output = json.loads(fmt_json) if isinstance(fmt_json, str) else fmt_json

                markdown_text = formatter.format_markdown(search_results_objs, include_summary=True)

        elif strategy == "grep":
            grep_raw_dir = path
            if _is_tantivy_index(grep_raw_dir):
                grep_raw_dir = grep_raw_dir.parent

            _REGEX_META = set(".+*?[](){}|^$\\<>=!")
            effective_pattern = query_text
            auto_converted = False
            if " " in query_text.strip() and not any(c in _REGEX_META for c in query_text):
                words = [w.strip() for w in query_text.split() if w.strip()]
                if len(words) >= 2:
                    effective_pattern = "|".join(words)
                    auto_converted = True

            from src.agent.tools.grep import GrepTool

            grep_tool = GrepTool(raw_dir=grep_raw_dir, max_results=limit)
            result = grep_tool.execute(pattern=effective_pattern, case_sensitive=False, file_filter="*.md")

            if not result.success:
                if _RICH_AVAILABLE:
                    console.print(f"[error]Grep 搜索失败: {result.error}[/error]")
                else:
                    click.echo(f"❌ Grep 搜索失败: {result.error}")
                return

            total_matches = result.metadata.get("total_matches", 0)
            files_searched = result.metadata.get("files_searched", 0)
            execution_time = result.metadata.get("execution_time", 0)

            if total_matches > 0 and result.data != "No matches found.":
                grep_style = True
                log_path_key = "raw_dir"
                log_path_val = str(grep_raw_dir)

                if auto_converted:
                    pre_lines.append((
                        f'[dim]💡 多词查询已自动转换: "{query_text}" → OR 模式 "{effective_pattern}"[/dim]',
                        f'💡 多词查询已自动转换: "{query_text}" → OR 模式',
                    ))

                output_lines = result.data.split("\n") if isinstance(result.data, str) else []
                files_with_matches = len(set(
                    line.split(":")[0] for line in output_lines if ":" in line
                ))

                match_idx = 0
                for line in output_lines:
                    if line.startswith("  "):
                        display_rows.append({"is_context": True, "text": line.strip()})
                    elif ":" in line:
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            match_idx += 1
                            file_display = parts[0]
                            with contextlib.suppress(ValueError, TypeError):
                                file_display = str(Path(parts[0]).relative_to(grep_raw_dir))
                            display_rows.append({
                                "is_match": True,
                                "idx": match_idx,
                                "file_display": file_display,
                                "line": parts[1].strip(),
                                "content": parts[2].strip()[:120],
                                "source_path": parts[0],
                            })
                            sources_items.append({"source_path": parts[0], "search_type": "Grep"})

                summary_parts = [
                    f"共 {total_matches} 条匹配",
                    f"{files_with_matches} 个文件",
                    f"搜索了 {files_searched} 个文件",
                ]

                json_match_results = []
                for line in output_lines:
                    if line.startswith("  "):
                        continue
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        json_match_results.append({
                            "file": parts[0],
                            "line": int(parts[1].strip()) if parts[1].strip().isdigit() else 0,
                            "content": parts[2].strip()[:200],
                        })
                json_output = {
                    "results": json_match_results[:limit],
                    "files_searched": files_searched,
                    "files_with_matches": files_with_matches,
                    "total_matches": total_matches,
                    "search_mode": "grep",
                    "execution_time": round(execution_time, 3),
                }
                log_extra = {
                    "total_matches": total_matches,
                    "files_with_matches": files_with_matches,
                    "files_searched": files_searched,
                }

        elif strategy == "hybrid":
            from src.search.hybrid import HybridSearcher

            raw_dir = path.parent
            searcher_bm25 = create_searcher(index_path=path, use_jieba=True, readonly=True)
            hybrid = HybridSearcher(bm25_searcher=searcher_bm25, grep_raw_dir=raw_dir)
            results = hybrid.search(query_text, limit=limit)
            execution_time = results.execution_time

            if results.results:
                json_results = []
                for r in results.results:
                    src_type = (
                        "BM25+Grep" if (r.grep_matches > 0 and r.raw_score > 0)
                        else ("Grep" if r.grep_matches > 0 else "BM25")
                    )
                    sources_items.append({
                        "source_path": str(r.source_path) if r.source_path else "",
                        "title": r.title, "score": r.rrf_score, "search_type": src_type,
                    })
                    badge_rich = (
                        "[bold magenta]B+G[/bold magenta]" if (r.grep_matches > 0 and r.raw_score > 0)
                        else ("[bold yellow]G[/bold yellow]" if r.grep_matches > 0 else "[bold cyan]B[/bold cyan]")
                    )
                    badge_plain = (
                        "B+G" if (r.grep_matches > 0 and r.raw_score > 0)
                        else ("G" if r.grep_matches > 0 else "B")
                    )
                    extra_lines = []
                    if r.grep_matches > 0:
                        extra_lines.append(f"Grep 匹配: {r.grep_matches} 处")
                    display_rows.append({
                        "idx": r.rank,
                        "title": r.title,
                        "score_text": f"RRF: {r.rrf_score:.4f}",
                        "source_display": _truncate_source(r.source_path) if r.source_path else "未知",
                        "snippet": r.snippet[:120] if r.snippet else "",
                        "badge_rich": badge_rich,
                        "badge_plain": badge_plain,
                        "extra_lines": extra_lines,
                    })
                    json_results.append({
                        "rank": r.rank,
                        "title": r.title,
                        "source": str(r.source_path) if r.source_path else "",
                        "rrf_score": round(r.rrf_score, 6),
                        "snippet": r.snippet[:200],
                        "source_type": r.search_source.value,
                        "grep_matches": r.grep_matches,
                    })
                json_output = {
                    "results": json_results,
                    "total": results.total,
                    "query": results.query,
                    "sources_used": results.sources_used,
                    "bm25_count": results.bm25_count,
                    "grep_count": results.grep_count,
                    "execution_time": round(execution_time, 3),
                }
                summary_parts = [
                    f"共 {len(results.results)} 条结果",
                    f"BM25: {results.bm25_count}",
                    f"Grep: {results.grep_count}",
                ]
                log_extra = {"bm25_count": results.bm25_count, "grep_count": results.grep_count}

        elif strategy == "tag":
            from src.converter.tag_extractor import TagExtractor

            extractor = TagExtractor()
            tag_result = extractor.extract(markdown=query_text, filename="")
            extracted_tags = tag_result.tags
            doc_type = tag_result.doc_type

            tag_display = ", ".join(extracted_tags) if extracted_tags else "(无匹配标签)"
            post_lines.append((
                f"  [dim]提取标签: {tag_display}[/dim]",
                f"  提取标签: {tag_display}",
            ))
            post_lines.append((
                f"  [dim]文档类型: {doc_type}[/dim]",
                f"  文档类型: {doc_type}",
            ))

            search_terms = [query_text]
            search_terms.extend(extracted_tags)
            enhanced_query = " ".join(search_terms)

            is_tantivy = _is_tantivy_index(path)
            if is_tantivy:
                searcher = create_searcher(index_path=path, use_jieba=True, readonly=True)
                results = searcher.search(enhanced_query, limit=limit)
                execution_time = results.execution_time

                if results.results:
                    json_results = []
                    for p in results.results:
                        sources_items.append({
                            "source_path": str(p.source_path) if p.source_path else "",
                            "title": p.title, "score": p.score, "search_type": "Tag",
                        })
                        display_rows.append({
                            "idx": len(display_rows) + 1,
                            "title": p.title,
                            "score_text": f"{p.score:.2f}",
                            "source_display": _truncate_source(p.source_path) if p.source_path else "未知",
                            "snippet": p.snippet[:120] if p.snippet else "",
                        })
                        json_results.append({
                            "title": p.title,
                            "score": p.score,
                            "snippet": p.snippet,
                            "source": str(p.source_path) if p.source_path else "",
                        })
                    json_output = {
                        "query": query_text,
                        "extracted_tags": extracted_tags,
                        "doc_type": doc_type,
                        "enhanced_query": enhanced_query,
                        "results": json_results,
                        "total": len(results.results),
                        "execution_time": round(execution_time, 3),
                    }
                    summary_parts = [
                        f"共 {len(results.results)} 条结果",
                        f"🏷️ 标签: {', '.join(extracted_tags)}",
                    ]
                    log_extra = {"extracted_tags": extracted_tags}
            else:
                tag_raw_dir = path
                from src.agent.tools.grep import GrepTool

                grep_tool = GrepTool(raw_dir=tag_raw_dir, max_results=limit)
                grep_result = grep_tool.execute(
                    pattern=enhanced_query, case_sensitive=False, file_filter="*.md",
                )
                if grep_result.success and grep_result.data != "No matches found.":
                    grep_style = True
                    output_lines = (
                        grep_result.data.split("\n") if isinstance(grep_result.data, str) else []
                    )
                    shown = 0
                    for line in output_lines:
                        if line.startswith("  "):
                            continue
                        if ":" in line and shown < limit:
                            shown += 1
                            parts = line.split(":", 2)
                            if len(parts) >= 3:
                                display_rows.append({
                                    "is_match": True,
                                    "idx": shown,
                                    "file_display": parts[0],
                                    "line": parts[1].strip(),
                                    "content": parts[2].strip()[:120],
                                })

        elif strategy == "multi_index":
            from src.search.multi_index import MultiIndexSearcher

            searcher = MultiIndexSearcher(index_paths=index_paths)
            results = searcher.search(query_text, limit=limit)
            execution_time = results.execution_time

            if results.results:
                json_results = []
                for r in results.results:
                    sources_items.append({
                        "source_path": str(r.source_path) if r.source_path else "",
                        "title": r.title,
                        "score": r.rrf_score,
                        "search_type": "Multi-Index",
                        "index_name": r.index_name,
                    })
                    display_rows.append({
                        "idx": r.rank,
                        "title": r.title,
                        "score_text": f"RRF: {r.rrf_score:.4f}",
                        "source_display": _truncate_source(r.source_path) if r.source_path else "未知",
                        "snippet": r.snippet[:120] if r.snippet else "",
                        "extra_lines": [f"索引: {r.index_name}"],
                    })
                    json_results.append({
                        "rank": r.rank,
                        "title": r.title,
                        "source": str(r.source_path) if r.source_path else "",
                        "rrf_score": round(r.rrf_score, 6),
                        "index_name": r.index_name,
                        "snippet": r.snippet[:200],
                    })
                json_output = {
                    "results": json_results,
                    "total": results.total,
                    "query": results.query,
                    "indexes_searched": results.sources_used,
                    "execution_time": round(execution_time, 3),
                }
                summary_parts = [
                    f"共 {len(results.results)} 条结果",
                    f"搜索了 {len(results.sources_used)} 个索引",
                ]
                log_extra = {"indexes_searched": results.sources_used}
                log_path_val = ",".join(str(p) for p in index_paths)
                panel_title = f"🔍 多索引搜索 ({len(index_paths)} 个索引)"
                index_names = ", ".join(results.sources_used) if results.sources_used else "N/A"
                post_lines.append((
                    f"[dim]搜索索引: {index_names}[/dim]",
                    f"搜索索引: {index_names}",
                ))

        # ── Shared output pipeline ──
        time.time() - start_time

        if not sources_items and not display_rows:
            if _RICH_AVAILABLE:
                console.print("[dim]未找到匹配的结果。[/dim]")
            else:
                click.echo("未找到匹配的结果。")
            return

        log_data = {"results": sources_items, "execution_time": execution_time, **log_extra}

        if output_format == "json":
            json_output["sources"] = _format_sources_data(sources_items, sources_detail)
            click.echo(json.dumps(json_output, ensure_ascii=False, indent=2))
            if export_path:
                _export_search_results(sources_items, export_path, query_text)
            _log_search_cli(query_text, log_data, strategy, **{log_path_key: log_path_val})
            return

        if output_format == "markdown" and markdown_text is not None:
            click.echo(markdown_text)
            sources_data = _format_sources_data(sources_items, sources_detail)
            if sources_data:
                click.echo("\n## 📚 引用文件")
                for idx, s in enumerate(sources_data, 1):
                    click.echo(f"{idx}. {s.get('filename', '')} — {s.get('path', '')}")
            if export_path:
                _export_search_results(sources_items, export_path, query_text)
            _log_search_cli(query_text, log_data, strategy, **{log_path_key: log_path_val})
            return

        for rich_line, plain_line in pre_lines:
            if _RICH_AVAILABLE and rich_line:
                console.print(rich_line)
            elif plain_line:
                click.echo(plain_line)

        if _RICH_AVAILABLE:
            console.print(
                Panel(
                    Text(query_text, style="query"),
                    title=panel_title or f"🔍 {label} 搜索",
                    border_style=border_style,
                    padding=(0, 1),
                )
            )
            console.print()
        else:
            click.echo(f"🔍 {label} 搜索: {query_text}")
            click.echo()

        for rich_line, plain_line in post_lines:
            if _RICH_AVAILABLE and rich_line:
                console.print(rich_line)
            elif plain_line:
                click.echo(plain_line)
        if post_lines:
            if _RICH_AVAILABLE:
                console.print()
            else:
                click.echo()

        if grep_style:
            for row in display_rows:
                if row.get("is_context"):
                    if _RICH_AVAILABLE:
                        console.print(f"[dim]    {row['text'][:120]}[/dim]")
                elif row.get("is_match") and row["idx"] <= limit:
                    if _RICH_AVAILABLE:
                        console.print(f"[dim][{row['idx']}][/dim] {row['file_display']}:{row['line']}")
                        console.print(f"    {row['content']}")
                        console.print()
                    else:
                        click.echo(f"[{row['idx']}] {row['file_display']}:{row['line']}")
                        click.echo(f"    {row['content']}")
                        click.echo()
        else:
            for row in display_rows:
                if _RICH_AVAILABLE:
                    title_text = Text()
                    title_text.append(f"[{row['idx']}] ", style="bold white")
                    title_text.append(row["title"], style="result")
                    if row.get("score_text"):
                        title_text.append("  (", style="dim")
                        title_text.append(row["score_text"], style="score")
                        title_text.append(")", style="dim")
                    console.print(title_text)

                    badge = row.get("badge_rich")
                    if badge:
                        console.print(
                            f"    {badge}  [source]来源: {row['source_display']}[/source]"
                        )
                    else:
                        console.print(f"    [source]来源: {row['source_display']}[/source]")

                    snippet = row.get("snippet", "")
                    if snippet:
                        console.print(f"    [dim]{snippet}[/dim]")
                    for extra in row.get("extra_lines", []):
                        console.print(f"    [dim]{extra}[/dim]")
                    console.print()
                else:
                    badge = row.get("badge_plain", "")
                    badge_str = f" ({badge})" if badge else ""
                    score_str = f" ({row['score_text']})" if row.get("score_text") else ""
                    click.echo(f"[{row['idx']}] {row['title']}{badge_str}{score_str}")
                    click.echo(f"    来源: {row['source_display']}")
                    snippet = row.get("snippet", "")
                    if snippet:
                        click.echo(f"    {snippet}")
                    for extra in row.get("extra_lines", []):
                        click.echo(f"    {extra}")
                    click.echo()

        if not summary_parts:
            summary_parts = [f"共 {len(display_rows)} 条结果"]
        summary_parts.append(f"⏱️ {execution_time:.3f}秒")
        summary_line = " | ".join(summary_parts)
        if _RICH_AVAILABLE:
            console.print(summary_line, style="dim")
        else:
            click.echo(summary_line)

        _display_sources_panel(sources_items, sources_detail, label)

        if export_path:
            _export_search_results(sources_items, export_path, query_text)

        _log_search_cli(query_text, log_data, strategy, **{log_path_key: log_path_val})

    except Exception as e:
        msg = f"❌ {label} 搜索失败: {e}"
        if _RICH_AVAILABLE:
            console.print(msg, style="error")
        else:
            click.echo(msg)
        sys.exit(1)


def _query_with_bm25(
    query_text: str,
    index_path: Path,
    limit: int,
    output_format: str,
    sources: str | None,
    sources_detail: str = "file",
    export_path: Path | None = None,
):
    """Execute BM25 keyword search."""
    _execute_list_search(
        "bm25", query_text, index_path, limit, output_format, sources_detail, export_path
    )


def _query_with_grep(
    query_text: str,
    raw_dir: Path,
    limit: int,
    output_format: str,
    sources_detail: str = "file",
    export_path: Path | None = None,
):
    """Execute GrepTool search on raw markdown files (no BM25 index needed)."""
    _execute_list_search(
        "grep", query_text, raw_dir, limit, output_format, sources_detail, export_path
    )


def _query_with_hybrid(
    query_text: str,
    index_path: Path,
    limit: int,
    output_format: str,
    sources_detail: str = "file",
    export_path: Path | None = None,
):
    """Execute hybrid BM25 + Grep search with RRF fusion."""
    _execute_list_search(
        "hybrid", query_text, index_path, limit, output_format, sources_detail, export_path
    )


def _query_with_tag(
    query_text: str,
    index_path: Path,
    limit: int,
    output_format: str,
    sources_detail: str = "file",
    export_path: Path | None = None,
):
    """Execute tag-based recall search."""
    _execute_list_search(
        "tag", query_text, index_path, limit, output_format, sources_detail, export_path
    )


def _query_with_multi_index(
    query_text: str,
    index_paths: list,
    limit: int,
    output_format: str,
    sources_detail: str = "file",
    export_path: Path | None = None,
):
    """Execute multi-index search with cross-index RRF merge."""
    _execute_list_search(
        "multi_index", query_text, index_paths[0], limit, output_format,
        sources_detail, export_path, index_paths=index_paths,
    )

def _query_with_agent(
    query_text: str,
    index_path: Path,
    limit: int,
    agent_mode: str,
    output_format: str,
    model: str | None,
    use_rerank: bool = False,
    mode: str = "tool_loop",
    skill: str | None = None,
    load_skill: str | None = None,
    sources_detail: str = "file",
    export_path: Path | None = None,
):
    """Execute semantic Q&A with SearchAgent."""
    try:
        from src.agent.search_agent import create_search_agent
        from src.utils.config import Config

        # Load config
        try:
            config = Config.from_env()
        except ValueError:
            if _RICH_AVAILABLE:
                console.print(
                    "❌ 未配置API密钥，请设置 GLM_API_KEY 和 GLM_BASE_URL 环境变量",
                    style="error",
                )
            else:
                click.echo("❌ 未配置API密钥，请设置 GLM_API_KEY 和 GLM_BASE_URL 环境变量")
            sys.exit(1)

        if model:
            # Override model if specified
            config = Config(
                glm_api_key=config.glm_api_key,
                glm_base_url=config.glm_base_url,
                llm_model=model,
            )

        # Determine output base from index path
        output_base = index_path.parent

        # Determine raw_dir (markdown files live alongside index)
        raw_dir = output_base if output_base.is_dir() else None

        # Create agent with GrepTool + optional Reranker
        agent = create_search_agent(
            config=config,
            index_path=index_path,
            output_base=output_base,
            raw_dir=raw_dir,
            use_rerank=use_rerank,   # kept for pipeline mode compat
            mode=mode,            # NEW
        )

        # Load external skill content if requested
        loaded_skill_content = None
        if load_skill:
            from src.agent.skill_loader import load_skill_content
            loaded_skill_content = load_skill_content(load_skill)
            if loaded_skill_content is None:
                click.echo(f"⚠️ 未找到技能: {load_skill}")

        # Show spinner while thinking
        if _RICH_AVAILABLE:
            with console.status("[bold cyan]🤔 正在分析...[/bold cyan]", spinner="dots"):
                response = agent.run(
                    query=query_text,
                    context={"mode": agent_mode} if agent_mode != "auto" else None,
                    skill=skill,
                    loaded_skill_content=loaded_skill_content,
                )
        else:
            click.echo("🤔 正在分析...")
            response = agent.run(
                query=query_text,
                context={"mode": agent_mode} if agent_mode != "auto" else None,
                skill=skill,
                loaded_skill_content=loaded_skill_content,
            )

        if response.success:
            # Build enriched sources data from tool_calls
            agent_sources = []
            for tc in response.tool_calls:
                if tc.get("tool") == "read":
                    args = tc.get("arguments", {})
                    sp = args.get("source_path") or args.get("doc_id") or ""
                    if sp:
                        agent_sources.append({
                            "source_path": sp,
                            "search_type": "Agent-Read",
                        })
                elif tc.get("tool") == "search" and tc.get("success"):
                    args = tc.get("arguments", {})
                    agent_sources.append({
                        "source_path": f"search:{args.get('query', '')}",
                        "search_type": "Agent-Search",
                    })

            if output_format == "json":
                # Plain text for programmatic consumption
                data = {
                    "answer": response.answer,
                    "sources": response.sources,
                    "sources_detail": _format_sources_data(agent_sources, sources_detail),
                    "tokens_used": response.tokens_used,
                    "processing_time": response.processing_time,
                }
                click.echo(json.dumps(data, ensure_ascii=False, indent=2))
            elif output_format == "markdown":
                # Plain text for programmatic consumption
                click.echo(response.answer)
                if response.sources:
                    click.echo("\n## 来源")
                    for idx, source in enumerate(response.sources[:5], 1):
                        click.echo(f"{idx}. {source}")
            else:
                # Rich-enhanced text output
                if _RICH_AVAILABLE:
                    # Render answer as Markdown
                    console.print()
                    console.print(
                        Panel(
                            Markdown(response.answer),
                            title="📄 回答",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )

                    if response.sources:
                        source_lines = []
                        for idx, source in enumerate(response.sources[:5], 1):
                            source_lines.append(f"[{idx}] {source}")
                        console.print(
                            Panel(
                                "\n".join(source_lines),
                                title="📚 来源",
                                border_style="cyan",
                                padding=(0, 1),
                            )
                        )

                    # Display enriched sources panel
                    _display_sources_panel(agent_sources, sources_detail, "Agent")

                    # Styled footer
                    time_info = f"⏱️ {response.processing_time:.1f}秒"
                    token_info = (
                        f"📊 {response.tokens_used:,} tokens"
                        if response.tokens_used > 0
                        else ""
                    )
                    meta_parts = [p for p in [time_info, token_info] if p]
                    if meta_parts:
                        console.print(" | ".join(meta_parts), style="dim")
                else:
                    # Fallback plain text
                    click.echo("\n📄 回答:")
                    click.echo(response.answer)

                    if response.sources:
                        click.echo("\n📚 来源:")
                        for idx, source in enumerate(response.sources[:5], 1):
                            click.echo(f"  [{idx}] {source}")

                    time_info = f"⏱️ {response.processing_time:.1f}秒"
                    token_info = (
                        f"📊 {response.tokens_used} tokens"
                        if response.tokens_used > 0
                        else ""
                    )
                    meta_parts = [p for p in [time_info, token_info] if p]
                    if meta_parts:
                        click.echo(f"\n{' | '.join(meta_parts)}")

                    # Export agent results if requested
                    if export_path and response.sources:
                        agent_sources = [
                            {"source_path": s, "title": Path(s).stem, "score": 0, "search_type": "Agent"}
                            for s in response.sources
                        ]
                        _export_search_results(agent_sources, export_path, query_text)
        else:
            if _RICH_AVAILABLE:
                console.print(f"❌ 查询失败: {response.error}", style="error")
            else:
                click.echo(f"❌ 查询失败: {response.error}")
            sys.exit(1)

    except ImportError as e:
        if _RICH_AVAILABLE:
            console.print(f"❌ 无法加载Agent模块: {e}", style="error")
        else:
            click.echo(f"❌ 无法加载Agent模块: {e}")
        sys.exit(1)
    except Exception as e:
        if _RICH_AVAILABLE:
            console.print(f"❌ Agent查询失败: {e}", style="error")
        else:
            click.echo(f"❌ Agent查询失败: {e}")
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True), required=False)
@click.option("--detailed", "-d", is_flag=True, help="显示详细信息")
@click.option("--show-tasks", is_flag=True, help="显示任务列表")
@click.option("--show-errors", is_flag=True, help="显示错误信息")
@click.option("--show-metrics", is_flag=True, help="显示统计指标")
def status(path, detailed, show_tasks, show_errors, show_metrics):
    """显示系统状态或文档状态

    示例:
        doc-search status ./output
        doc-search status ./output --detailed --show-errors
    """
    path = Path(path).resolve() if path else Path.cwd() / "output"

    if not path.exists():
        click.echo(f"❌ 目录不存在: {path}")
        return

    click.echo(f"📁 工作目录: {path}")
    click.echo()

    # Document count
    md_files = list(path.rglob("*.md"))
    click.echo(f"📄 文档数量: {len(md_files)}")

    # Index status
    index_path = path / "index"
    if index_path.exists():
        try:
            index_manager = TantivyIndexManager(index_path=index_path)
            stats = index_manager.get_stats()
            click.echo("📇 索引状态:")
            click.echo(f"   文档数: {stats.get('num_docs', 0)}")
            click.echo(f"   路径: {stats.get('index_path', 'N/A')}")
            click.echo(
                f"   Jieba分词: {'启用' if stats.get('jieba_enabled') else '禁用'}"
            )
        except Exception as e:
            click.echo(f"📇 索引状态: 无法读取 ({e})")
    else:
        click.echo("📇 索引状态: 未创建")

    # Metadata
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata_manager = MetadataManager(index_path=metadata_path)
        click.echo(f"📋 元数据记录: {metadata_manager.get_count()}")

    # Task status
    if show_tasks:
        click.echo("\n📝 任务列表:")
        task_manager = TaskManager.get_instance(path / ".tasks")
        tasks = task_manager.list_tasks()
        if tasks:
            for task in tasks[:10]:
                status_icon = {
                    "pending": "⏳",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }.get(task.get("status", "pending"), "❓")
                click.echo(
                    f"  {status_icon} {task['id']}: {task['type']} ({task.get('progress', 0)}/{task.get('total', 0)})"
                )
        else:
            click.echo("  暂无任务记录")

    # Detailed file status
    if detailed and md_files:
        click.echo("\n📄 文件详情:")
        for md_file in md_files[:20]:
            meta_file = md_file.with_suffix(md_file.suffix + ".json")
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    status_str = meta.get("status", "active")
                    convert_time = meta.get("last_convert_time", 0)
                    click.echo(
                        f"  - {md_file.name}: {status_str} (转换耗时: {convert_time:.2f}s)"
                    )
                except (json.JSONDecodeError, KeyError):
                    click.echo(f"  - {md_file.name}: 无法读取元数据")
            else:
                click.echo(f"  - {md_file.name}: 无元数据")

    # Error information
    if show_errors:
        task_manager = TaskManager.get_instance(path / ".tasks")
        tasks_with_errors = [t for t in task_manager.list_tasks() if t.get("errors")]
        if tasks_with_errors:
            click.echo("\n❌ 错误信息:")
            for task in tasks_with_errors[:5]:
                click.echo(f"  任务 {task['id']}:")
                for error in task.get("errors", [])[:3]:
                    click.echo(f"    - {error}")

    # Metrics
    if show_metrics and md_files:
        click.echo("\n📊 统计指标:")
        total_size = sum(f.stat().st_size for f in md_files)
        click.echo(f"  总大小: {_format_file_size(total_size)}")
        click.echo(f"  平均文件大小: {_format_file_size(total_size // len(md_files))}")


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), default="./output", help="输出目录路径"
)
@click.option(
    "--strategy",
    type=click.Choice(["full", "incremental", "smart"]),
    default="smart",
    help="更新策略",
)
@click.option("--dry-run", is_flag=True, help="模拟运行,不实际执行")
@click.option("--parallel", "-p", type=int, default=1, help="并行处理文档数量")
def update(source, output, strategy, dry_run, parallel):
    """更新现有索引

    检测文件变更并增量更新索引

    策略:
    - full: 完全重建
    - incremental: 仅更新变更的文件
    - smart: 使用哈希检测精确变更

    示例:
        doc-search update ./docs -o ./output
        doc-search update ./docs -o ./output --strategy incremental
    """
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()

    click.echo(f"📂 源目录: {source_path}")
    click.echo(f"📁 输出目录: {output_path}")
    click.echo(f"🔄 策略: {strategy}")

    if not output_path.exists():
        click.echo("❌ 输出目录不存在，请先运行 convert 命令")
        sys.exit(1)

    metadata_path = output_path / "metadata.json"
    if not metadata_path.exists():
        click.echo("❌ 元数据文件不存在，请先运行 convert 命令")
        sys.exit(1)

    if dry_run:
        click.echo("🔍 模拟运行模式 - 不执行实际更新")

    if strategy == "full":
        # Full rebuild
        click.echo("🔄 执行完全重建...")
        if not dry_run:
            # Re-run convert with force flag
            click.echo("请使用 convert --force 命令进行完全重建")
        return

    # Detect changes
    metadata_manager = MetadataManager(index_path=metadata_path)
    watcher = FileWatcher(
        use_mtime_check=(strategy == "incremental"),
        use_hash_check=(strategy == "smart"),
    )

    # Get supported extensions
    extensions = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".html",
        ".htm",
    }

    click.echo("🔍 检测文件变更...")
    changes: ChangeSet = watcher.detect_changes(
        source_dir=source_path,
        metadata_manager=metadata_manager,
        extensions=extensions,
    )

    click.echo("\n📊 变更统计:")
    click.echo(f"  ➕ 新增: {len(changes.added)}")
    click.echo(f"  ✏️ 修改: {len(changes.modified)}")
    click.echo(f"  ➖ 删除: {len(changes.deleted)}")
    click.echo(f"  ✅ 未变更: {len(changes.unchanged)}")

    if not changes.has_changes:
        click.echo("\n✅ 没有检测到变更")
        return

    if dry_run:
        click.echo("\n变更的文件:")
        for f in changes.added[:5]:
            click.echo(f"  ➕ {f.relative_to(source_path)}")
        for f in changes.modified[:5]:
            click.echo(f"  ✏️ {f.relative_to(source_path)}")
        for f in changes.deleted[:5]:
            click.echo(f"  ➖ {f.name}")
        if len(changes.added) + len(changes.modified) + len(changes.deleted) > 15:
            click.echo("  ... 还有更多")
        return

    # Initialize components for update
    coordinator = ConverterCoordinator()
    store = MarkdownStore(input_base=source_path, output_base=output_path)
    index_path = output_path / "index"
    index_manager = (
        TantivyIndexManager(index_path=index_path) if index_path.exists() else None
    )

    # Process added and modified files
    files_to_process = changes.added + changes.modified
    success_count = 0
    failed_count = 0

    if files_to_process:
        with click.progressbar(files_to_process, label="更新中") as bar:
            for file_path in bar:
                try:
                    # Convert
                    result = coordinator.convert(
                        source=file_path,
                        output_dir=output_path,
                    )

                    if result.success:
                        # Create record
                        record = DocumentRecord(
                            id=store._generate_doc_id(file_path),
                            source_path=file_path,
                            output_path=store.get_output_path(file_path),
                            title=file_path.stem,
                            content_hash=calculate_hash(file_path),
                            file_size=file_path.stat().st_size,
                            file_mtime=datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ),
                            metadata=result.metadata,
                        )

                        # Save
                        if result.images:
                            store.save_with_images(
                                record, result.markdown, result.images
                            )
                        else:
                            store.save(record, result.markdown)

                        # Update index
                        if index_manager:
                            index_manager.update_document(
                                doc_id=record.id,
                                title=record.title,
                                content=result.markdown,
                                metadata={
                                    "filename": file_path.name,
                                    "source_path": str(file_path),
                                    "modified_time": record.file_mtime,
                                },
                            )

                        # Update metadata
                        metadata_manager.save(
                            file_path,
                            {
                                "source_path": file_path,
                                "output_path": record.output_path,
                                "content_hash": record.content_hash,
                                "modified_time": file_path.stat().st_mtime,
                            },
                        )

                        success_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    failed_count += 1
                    click.echo(f"\n❌ 处理失败 {file_path.name}: {e}")

    # Handle deleted files
    for file_path in changes.deleted:
        try:
            store.delete_by_source(file_path)
            if index_manager:
                doc_id = store._generate_doc_id(file_path)
                index_manager.delete_document(doc_id)
            metadata_manager.delete(file_path)
        except Exception:
            logger.warning("Failed to delete file from index/metadata: %s", file_path)

    # Commit index
    if index_manager:
        index_manager.commit()

    click.echo("\n📊 更新完成:")
    click.echo(f"  ✅ 成功: {success_count}")
    click.echo(f"  ❌ 失败: {failed_count}")
    click.echo(f"  🗑️ 已删除: {len(changes.deleted)}")


@cli.group()
def task():
    """任务管理命令"""
    pass


@task.command("list")
@click.option(
    "--status",
    type=click.Choice(["all", "pending", "running", "completed", "failed"]),
    default="all",
    help="按状态过滤",
)
@click.option("--limit", "-l", type=int, default=20, help="显示数量限制")
@click.option("--path", type=click.Path(), default="./output", help="工作目录")
def task_list(status, limit, path):
    """列出所有任务

    示例:
        doc-search task list
        doc-search task list --status failed
    """
    work_path = Path(path).resolve()
    task_manager = TaskManager.get_instance(work_path / ".tasks")
    tasks = task_manager.list_tasks(status=status)

    if not tasks:
        click.echo("暂无任务记录")
        return

    click.echo(f"📋 任务列表 (状态: {status}):\n")

    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
    }

    for task in tasks[:limit]:
        icon = status_icons.get(task.get("status", "pending"), "❓")
        progress = f"{task.get('progress', 0)}/{task.get('total', 0)}"
        created = task.get("created_at", "")[:19]

        click.echo(f"{icon} {task['id']}")
        click.echo(f"   类型: {task['type']} | 进度: {progress} | 创建时间: {created}")

        if task.get("errors"):
            click.echo(f"   错误: {len(task['errors'])} 个")


@task.command("show")
@click.argument("task_id")
@click.option("--verbose", "-v", is_flag=True, help="显示详细信息")
@click.option("--path", type=click.Path(), default="./output", help="工作目录")
def task_show(task_id, verbose, path):
    """显示任务详情

    示例:
        doc-search task show convert_1234567890
    """
    work_path = Path(path).resolve()
    task_manager = TaskManager.get_instance(work_path / ".tasks")
    task = task_manager.get_task(task_id)

    if not task:
        click.echo(f"❌ 任务不存在: {task_id}")
        return

    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
    }

    icon = status_icons.get(task.get("status", "pending"), "❓")

    click.echo(f"{icon} 任务: {task['id']}")
    click.echo(f"   类型: {task['type']}")
    click.echo(f"   状态: {task['status']}")
    click.echo(f"   进度: {task.get('progress', 0)}/{task.get('total', 0)}")
    click.echo(f"   创建时间: {task.get('created_at', 'N/A')}")
    click.echo(f"   更新时间: {task.get('updated_at', 'N/A')}")

    if task.get("params"):
        click.echo("\n📝 参数:")
        for key, value in task["params"].items():
            click.echo(f"   {key}: {value}")

    if task.get("result"):
        click.echo("\n📊 结果:")
        for key, value in task["result"].items():
            click.echo(f"   {key}: {value}")

    if verbose and task.get("errors"):
        click.echo("\n❌ 错误详情:")
        for error in task["errors"]:
            click.echo(f"   - {error}")


@task.command("resume")
@click.argument("task_id")
@click.option("--path", type=click.Path(), default="./output", help="工作目录")
def task_resume(task_id, path):
    """恢复中断的任务

    注意: 当前版本的resume需要重新执行convert命令

    示例:
        doc-search task resume convert_1234567890
    """
    work_path = Path(path).resolve()
    task_manager = TaskManager.get_instance(work_path / ".tasks")
    task = task_manager.get_task(task_id)

    if not task:
        click.echo(f"❌ 任务不存在: {task_id}")
        return

    if task["status"] not in ["pending", "failed"]:
        click.echo(f"❌ 任务状态为 {task['status']}，无法恢复")
        return

    params = task.get("params", {})
    click.echo("📋 任务信息:")
    click.echo(f"   源目录: {params.get('source', 'N/A')}")
    click.echo(f"   输出目录: {params.get('output', 'N/A')}")
    click.echo(f"   总文件数: {params.get('total_files', 0)}")
    click.echo(f"   已完成: {task.get('progress', 0)}")

    click.echo("\n💡 请使用以下命令恢复转换:")
    click.echo(
        f"   doc-search convert {params.get('source', '.')} -o {params.get('output', './output')}"
    )


@task.command("cancel")
@click.argument("task_id")
@click.option("--path", type=click.Path(), default="./output", help="工作目录")
def task_cancel(task_id, path):
    """取消任务

    示例:
        doc-search task cancel convert_1234567890
    """
    work_path = Path(path).resolve()
    task_manager = TaskManager.get_instance(work_path / ".tasks")
    task = task_manager.get_task(task_id)

    if not task:
        click.echo(f"❌ 任务不存在: {task_id}")
        return

    if task["status"] not in ["pending", "running"]:
        click.echo(f"❌ 任务状态为 {task['status']}，无法取消")
        return

    task_manager.update_task(task_id, status="cancelled")
    click.echo(f"✅ 任务已取消: {task_id}")


@task.command("retry")
@click.argument("task_id")
@click.option("--force", is_flag=True, help="强制重试,忽略错误检查")
@click.option("--path", type=click.Path(), default="./output", help="工作目录")
def task_retry(task_id, force, path):
    """重试失败的任务

    示例:
        doc-search task retry convert_1234567890
    """
    work_path = Path(path).resolve()
    task_manager = TaskManager.get_instance(work_path / ".tasks")
    task = task_manager.get_task(task_id)

    if not task:
        click.echo(f"❌ 任务不存在: {task_id}")
        return

    if task["status"] != "failed" and not force:
        click.echo("❌ 只有失败的任务才能重试。使用 --force 强制重试")
        return

    params = task.get("params", {})
    click.echo(f"📋 重试任务: {task_id}")
    click.echo(f"   源目录: {params.get('source', 'N/A')}")
    click.echo(f"   输出目录: {params.get('output', 'N/A')}")

    click.echo("\n💡 请使用以下命令重试:")
    click.echo(
        f"   doc-search convert {params.get('source', '.')} -o {params.get('output', './output')} --force"
    )


# ── Parallel conversion helper ────────────────────

_cli_pypdf_module = None


def _get_cli_pypdf():
    """Lazy load pypdf for PDF encryption check."""
    global _cli_pypdf_module
    if _cli_pypdf_module is None:
        try:
            import pypdf
            _cli_pypdf_module = pypdf
        except ImportError:
            pass
    return _cli_pypdf_module


def _convert_one_file(
    file_record: dict,
    source_root: Path,
    raw_root_path: Path,
    ocr: bool,
    ocr_config: OCRServiceConfig | None,
    ocr_engine: str = "zhipu",
) -> dict:
    """Convert a single file in a worker thread.

    Each thread gets its own ConverterCoordinator and RawStore instance
    to avoid shared-state issues. DB writes are handled by the caller.

    Returns a result dict with status and conversion output.
    """
    file_id = file_record["id"]
    rel_path = file_record["relative_path"]
    source_file = source_root / rel_path

    if not source_file.exists():
        return {"file_id": file_id, "status": "skipped", "error": "源文件已删除"}

    try:
        coordinator = ConverterCoordinator(
            ocr_config=ocr_config,
            enable_ocr_fallback=ocr,
        )
        coordinator.register_custom_converter(ImageConverter(), override=True)
        coordinator.register_custom_converter(CSVConverter(), override=True)
        coordinator.register_custom_converter(TextConverter(), override=True)

        store = RawStore(source_root, raw_root_path)

        options: dict[str, Any] = {}
        if ocr:
            options["ocr_api_key"] = os.environ.get("GLM_API_KEY", "")
            options["ocr_base_url"] = os.environ.get("GLM_BASE_URL", "")
            options["ocr_engine"] = ocr_engine

        # Try password dictionary for encrypted PDFs
        # The converter layer (pdf.py) now handles dictionary-based password attempts
        # internally, so this CLI-level pre-check is no longer needed.
        # Password dictionary is loaded via:
        #   - PASSWORD_DICT_PATH env var
        #   - --password-dict CLI option (passed through convert_db)

        output_dir = store.map_output_path(source_file).parent
        result = coordinator.convert(
            source=source_file,
            output_dir=output_dir,
            options=options,
        )

        return {
            "file_id": file_id,
            "rel_path": rel_path,
            "source_file": source_file,
            "result": result,
            "store": store,
            "output_dir": output_dir,
            "status": "success" if result.success else "failed",
            "error": "; ".join(result.errors) if result.errors else None,
        }
    except Exception as e:
        return {"file_id": file_id, "status": "failed", "error": f"{type(e).__name__}: {str(e)}"}


# ── batch-convert 命令 ────────────────────────────

@cli.command("batch-convert")
@click.argument("source", type=click.Path(exists=True))
@click.option("--raw-root", type=click.Path(), default=None, help="输出 raw 根目录 (默认: ./raw)")
@click.option("--mode", type=click.Choice(["full", "incremental", "resume"]), default="incremental", help="转换模式")
@click.option("--parallel", "-p", type=int, default=1, help="并行转换线程数（默认 1 为串行）")
@click.option("--ocr/--no-ocr", default=True, help="启用/禁用 OCR")
@click.option("--ocr-engine", type=click.Choice(["zhipu", "paddleocr", "paddleocr-http", "ppstructurev3"]), default="zhipu", help="OCR 引擎: zhipu (云端) / paddleocr (本地) / paddleocr-http (远程GPU) / ppstructurev3 (本地结构化解析)")
@click.option("--generate-index/--no-index", default=True, help="生成目录索引")
@click.option("--dry-run", is_flag=True, help="模拟运行")
@click.option("--force", is_flag=True, help="强制重新转换")
@click.option("--log-file", type=click.Path(), default=None, help="日志文件路径（Task Scheduler 环境下推荐使用）")
@click.option("--password-dict", type=click.Path(), default=None, help="密码字典文件路径（UTF-8，每行一个密码，附加到内置字典）")
def batch_convert(source, raw_root, mode, parallel, ocr, ocr_engine, generate_index, dry_run, force, log_file, password_dict):
    """批量转换源目录文档到独立 raw 目录

    支持断点续传: 自动检测上次中断位置继续执行

    示例:
        doc-search batch-convert "./my-docs" --raw-root "D:\\docs\\raw"
        doc-search batch-convert "./my-docs" --mode resume
        doc-search batch-convert "./my-docs" --raw-root "D:\\docs\\raw" --parallel 4
        doc-search batch-convert "./my-docs" --force
        doc-search batch-convert "./my-docs" --log-file "D:\\logs\\convert.log"
    """
    # ── Setup logging (file + console) ──
    import logging as _logging
    _root_logger = _logging.getLogger()
    _root_logger.setLevel(_logging.DEBUG)

    if log_file:
        _log_path = Path(log_file).resolve()
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _fh = _logging.FileHandler(str(_log_path), encoding="utf-8", mode="a")
        _fh.setLevel(_logging.DEBUG)
        _fh.setFormatter(_logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _root_logger.addHandler(_fh)
        _root_logger.info("=== batch-convert started ===")
    # Step 1: Setup paths
    source_root = Path(source).resolve()
    if raw_root is None:
        raw_root_path = Path.cwd() / "raw"
    else:
        raw_root_path = Path(raw_root).resolve()
    output_root = raw_root_path / source_root.name

    click.echo(f"📁 源目录: {source_root}")
    click.echo(f"📁 输出目录: {output_root}")
    click.echo(f"🔧 模式: {mode}")

    # Set password dictionary path for converter layer
    if password_dict and os.path.isfile(password_dict):
        os.environ["PASSWORD_DICT_PATH"] = str(Path(password_dict).resolve())
        click.echo(f"🔑 密码字典: {password_dict}")

    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    # Step 2: Open DB and setup storage
    db_path = output_root / "convert.db"
    db = ConvertDB(db_path)
    store = RawStore(source_root, raw_root_path)
    batch_id = None

    try:
        db.open()

        # Step 3: Recovery check
        db.mark_interrupted_batches()
        active_batch = db.get_active_batch()

        if active_batch and mode in ("resume", "incremental"):
            batch_id = active_batch["id"]
            click.echo(f"🔄 恢复中断批次: #{batch_id} (已处理 {active_batch['processed']}/{active_batch['total_files']})")
        else:
            # Step 4: Scan
            click.echo("📂 扫描源目录...")
            pending_count = _scan_and_sync(db, source_root)
            click.echo(f"   发现 {pending_count} 个待处理文件")

            if force:
                # Reset all files to pending
                all_files = db.get_files_by_status("success") + db.get_files_by_status("skipped")
                for f in all_files:
                    db.update_file_status(f["id"], "pending")
                pending_count = db.count_files("pending")
                click.echo(f"   强制模式: 已重置所有文件，共 {pending_count} 个待处理")

            # Create new batch
            batch_id = db.create_batch(
                batch_type=mode,
                total_files=pending_count,
                config={
                    "source": str(source_root),
                    "raw_root": str(raw_root_path),
                    "mode": mode,
                    "ocr": ocr,
                    "force": force,
                },
            )
            click.echo(f"📦 创建批次: #{batch_id}")

        # Get pending files
        pending_files = db.get_pending_files(limit=100000)
        total = len(pending_files)

        if total == 0:
            click.echo("✅ 没有待处理的文件")
            if batch_id:
                db.complete_batch(batch_id)
            return

        if dry_run:
            click.echo(f"\n📋 模拟运行: 共 {total} 个待处理文件")
            for f in pending_files[:20]:
                click.echo(f"   • {f['relative_path']} ({_format_file_size(f['file_size'])})")
            if total > 20:
                click.echo(f"   ... 还有 {total - 20} 个文件")
            return

        # Step 5: Convert files
        ocr_config = None
        if ocr:
            if ocr_engine == "paddleocr":
                click.echo("🔧 OCR 引擎: PaddleOCR (本地)")
            elif ocr_engine == "paddleocr-http":
                from src.converter.ocr import get_paddleocr_http_service
                svc = get_paddleocr_http_service()
                if svc.health():
                    click.echo(f"🔧 OCR 引擎: PaddleOCR HTTP ({svc.base_url})")
                else:
                    click.echo(f"⚠️  PaddleOCR HTTP 服务不可达: {svc.base_url}")
                    click.echo("   请先在 WSL 中启动: python /home/paddleocr_test/ocr_server.py")
            elif ocr_engine == "ppstructurev3":
                click.echo("🔧 OCR 引擎: PP-StructureV3 (本地结构化解析)")
            else:
                api_key = os.environ.get("GLM_API_KEY", "")
                if api_key:
                    ocr_config = OCRServiceConfig(
                        api_key=api_key,
                        base_url=os.environ.get("GLM_BASE_URL"),
                    )
                    click.echo("🔧 OCR 引擎: ZhipuAI (云端)")
                else:
                    click.echo("⚠️  GLM_API_KEY 未设置，OCR 可能不可用")

        success_count = 0
        failed_count = 0
        skipped_count = 0

        def _process_result(conv_result: dict) -> None:
            """Process a single conversion result and update DB/counters.

            Called from the main thread only (DB writes are serialized).
            """
            nonlocal success_count, failed_count, skipped_count

            file_id = conv_result["file_id"]
            status = conv_result.get("status", "failed")

            if status == "skipped":
                db.update_file_status(file_id, "skipped", last_error=conv_result.get("error", ""))
                skipped_count += 1
                return

            if status == "failed":
                db.update_file_status(file_id, "failed", last_error=conv_result.get("error", ""))
                failed_count += 1
                return

            # Success path
            result = conv_result["result"]
            source_file = conv_result["source_file"]
            store = conv_result["store"]
            output_dir = conv_result["output_dir"]

            # Save via RawStore
            metadata = result.metadata or {}
            metadata["source_path"] = str(source_file)
            metadata["converter"] = result.converter_name or "unknown"
            metadata["convert_time"] = result.convert_time
            if result.token_usage:
                metadata["token_usage"] = result.token_usage

            output_path = store.map_output_path(source_file)
            store.save(source_file, result.markdown, metadata)

            # Clean up duplicate file written by Converter internals
            converter_output = output_dir / f"{source_file.stem}.md"
            if converter_output.exists() and converter_output != output_path:
                with contextlib.suppress(OSError):
                    converter_output.unlink()
            converter_meta = output_dir / f"{source_file.stem}.md.json"
            if converter_meta.exists():
                with contextlib.suppress(OSError):
                    converter_meta.unlink()

            update_kwargs: dict[str, Any] = dict(
                converter=result.converter_name,
                convert_time=result.convert_time,
                output_path=str(output_path),
                output_size=output_path.stat().st_size if output_path.exists() else 0,
                ocr_used=int(result.ocr_used) if result.ocr_used else 0,
                ocr_model=result.ocr_model or None,
            )

            token_usage = result.token_usage or {}
            if token_usage:
                update_kwargs["ocr_input_tokens"] = token_usage.get("input_tokens", 0)
                update_kwargs["ocr_output_tokens"] = token_usage.get("output_tokens", 0)
                update_kwargs["ocr_total_tokens"] = token_usage.get("total_tokens", 0)
                db.add_token_usage(
                    file_id=file_id,
                    model=result.ocr_model or "unknown",
                    input_tokens=token_usage.get("input_tokens", 0),
                    output_tokens=token_usage.get("output_tokens", 0),
                    total_tokens=token_usage.get("total_tokens", 0),
                    call_type="ocr",
                )

            update_kwargs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            db.update_file_status(file_id, "success", **update_kwargs)
            success_count += 1

            # Update batch progress
            processed = success_count + failed_count + skipped_count
            db.update_batch_progress(
                batch_id,
                processed=processed,
                success=success_count,
                failed=failed_count,
                skipped=skipped_count,
            )

        # ── Parallel mode ──────────────────────
        if parallel > 1:
            click.echo(f"\n🔄 开始并行转换 {total} 个文件 (workers={parallel})...\n")

            # Mark all pending files as "converting" first
            for file_record in pending_files:
                attempt = file_record.get("attempt_count", 0) + 1
                db.update_file_status(file_record["id"], "converting", attempt_count=attempt)

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(
                        _convert_one_file,
                        file_record=fr,
                        source_root=source_root,
                        raw_root_path=raw_root_path,
                        ocr=ocr,
                        ocr_config=ocr_config,
                        ocr_engine=ocr_engine,
                    ): fr
                    for fr in pending_files
                }

                with click.progressbar(length=total, label="转换进度", show_pos=True) as bar:
                    for future in as_completed(futures):
                        try:
                            conv_result = future.result()
                            _process_result(conv_result)
                        except Exception as e:
                            fr = futures[future]
                            db.update_file_status(
                                fr["id"], "failed",
                                last_error=f"{type(e).__name__}: {str(e)}",
                            )
                            failed_count += 1
                        bar.update(1)

        # ── Sequential mode (original path) ──
        else:
            click.echo(f"\n🔄 开始转换 {total} 个文件...\n")

            with click.progressbar(pending_files, label="转换进度", show_pos=True) as bar:
                for file_record in bar:
                    file_id = file_record["id"]
                    rel_path = file_record["relative_path"]
                    source_file = source_root / rel_path

                    if not source_file.exists():
                        db.update_file_status(file_id, "skipped", last_error="源文件已删除")
                        skipped_count += 1
                        continue

                    attempt = file_record.get("attempt_count", 0) + 1
                    db.update_file_status(file_id, "converting", attempt_count=attempt)

                    conv_result = _convert_one_file(
                        file_record=file_record,
                        source_root=source_root,
                        raw_root_path=raw_root_path,
                        ocr=ocr,
                        ocr_config=ocr_config,
                        ocr_engine=ocr_engine,
                    )

                    _process_result(conv_result)

        # Step 6: Generate indexes
        if generate_index:
            click.echo("\n📋 生成目录索引...")
            try:
                _generate_indexes(db, store)
            except Exception as e:
                click.echo(f"⚠️  索引生成失败: {e}")

        # Step 7: Complete batch
        db.complete_batch(batch_id)

        # Print summary
        click.echo(f"\n{'='*50}")
        click.echo("📊 转换完成")
        click.echo(f"   ✅ 成功: {success_count}")
        click.echo(f"   ❌ 失败: {failed_count}")
        click.echo(f"   ⏭️  跳过: {skipped_count}")
        click.echo(f"   📦 批次: #{batch_id}")

        # Log summary to file (critical for Task Scheduler visibility)
        if log_file:
            _total = locals().get("total", 0)
            _root_logger.info(
                "Batch #%s complete: success=%d failed=%d skipped=%d total=%d",
                batch_id, success_count, failed_count, skipped_count, _total,
            )

        # Token usage summary for this batch
        token_summary = db.get_token_summary()
        if token_summary.get("total_tokens", 0) > 0:
            click.echo("\n   💰 Token 使用量:")
            click.echo(f"      输入: {token_summary['input_tokens']:,}")
            click.echo(f"      输出: {token_summary['output_tokens']:,}")
            click.echo(f"      合计: {token_summary['total_tokens']:,}")
            for m in token_summary.get("by_model", []):
                click.echo(f"      📎 {m['model']}: {m['total_tokens']:,} tokens ({m['call_count']} calls)")

        click.echo(f"{'='*50}")

    except KeyboardInterrupt:
        click.echo("\n⚠️  转换被中断")
        if batch_id:
            try:
                db.complete_batch(batch_id, status="interrupted")
            except Exception:
                logger.warning("Failed to mark batch as interrupted: %s", batch_id)
    finally:
        db.close()


def _generate_indexes(db: ConvertDB, store: RawStore) -> None:
    """为每个目录生成增强版 _index.md 索引文件（含 frontmatter 元数据）。"""
    from src.converter.frontmatter import parse_frontmatter

    # 文档类型对应图标
    _type_icons = {
        "policy": "📋", "process": "⚙️", "report": "📊",
        "manual": "📖", "data": "📈", "other": "📄",
        "document": "📄",
    }

    # Get all directories sorted by depth (deepest first)
    all_dirs = []
    for rel_dir_str, dir_id in _iter_directories(db):
        all_dirs.append((rel_dir_str, dir_id))

    for rel_dir_str, dir_id in all_dirs:
        files = db.get_files_by_directory(dir_id)
        subdirs = db.list_subdirectories(dir_id)

        if not files and not subdirs:
            continue

        # Build _index.md content
        lines = []
        dir_name = rel_dir_str.rsplit("/", 1)[-1] if "/" in rel_dir_str else rel_dir_str
        if dir_name == ".":
            dir_name = store.get_output_root().name
        lines.append(f"# {dir_name}\n")

        # Collect tags from successful files for summary
        success_files = [f for f in files if f.get("status") == "success"]
        if success_files:
            all_tags = set()
            for f in success_files:
                filename = f.get("filename", "")
                md_name = filename + ".md"
                md_path = store.get_output_root() / rel_dir_str / md_name
                if md_path.exists():
                    try:
                        raw = md_path.read_text(encoding="utf-8")
                        fm = parse_frontmatter(raw)
                        if fm and fm.get("tags"):
                            all_tags.update(fm["tags"])
                    except Exception:
                        logger.warning("Failed to read frontmatter tags from %s", md_path)
            if all_tags:
                tag_str = " ".join(f"`{t}`" for t in sorted(all_tags)[:20])
                lines.append(f"> 📋 标签: {tag_str}\n")

        # List subdirectories
        for subdir in subdirs:
            sub_name = subdir["name"]
            sub_file_count = subdir.get("file_count", 0)
            lines.append(f"- 📁 [{sub_name}/]({sub_name}/_index.md) ({sub_file_count} 文件)")

        # List files with frontmatter type enrichment
        for f in files:
            status = f.get("status", "unknown")
            filename = f.get("filename", "")
            ext = f.get("extension", "")
            file_size = f.get("file_size", 0)

            if status == "success":
                md_name = filename + ".md"
                # Try to read frontmatter for type
                type_label = ""
                md_path = store.get_output_root() / rel_dir_str / md_name
                if md_path.exists():
                    try:
                        raw = md_path.read_text(encoding="utf-8")
                        fm = parse_frontmatter(raw)
                        if fm and fm.get("type"):
                            icon = _type_icons.get(fm["type"], "📄")
                            type_label = f" {icon} `{fm['type']}`"
                    except Exception:
                        logger.warning("Failed to read frontmatter type from %s", md_path)
                lines.append(f"- ✅{type_label} [{filename}]({md_name}) ({_format_file_size(file_size)})")
            elif status == "failed":
                lines.append(f"- ❌ {filename} ({ext}) - 转换失败")
            elif status == "skipped":
                lines.append(f"- ⏭️ {filename} ({ext}) - 已跳过")
            else:
                lines.append(f"- ⏳ {filename} ({ext}) ({_format_file_size(file_size)})")

        lines.append("")
        index_content = "\n".join(lines)

        # Write index file
        index_path = store.map_index_path(store.source_root / rel_dir_str)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(index_content, encoding="utf-8")

        db.set_index_generated(dir_id, True)


def _iter_directories(db: ConvertDB):
    """迭代所有目录记录，yield (relative_path, dir_id)。"""
    # Use the db connection directly to list all directories
    cursor = db.conn.execute("SELECT id, relative_path FROM directories ORDER BY depth, name")
    for row in cursor.fetchall():
        yield row["relative_path"], row["id"]


# ── catalog 命令组 ──────────────────────────────

@cli.group("catalog")
def catalog():
    """转换目录管理命令"""
    pass


@catalog.command("status")
@click.argument("raw_dir", type=click.Path(exists=True))
@click.option("--detailed", "-d", is_flag=True, help="显示详细信息")
def catalog_status(raw_dir, detailed):
    """查看转换状态"""
    raw_path = Path(raw_dir).resolve()
    db_path = raw_path / "convert.db"
    if not db_path.exists():
        click.echo(f"❌ 未找到转换数据库: {db_path}")
        return

    with ConvertDB(db_path) as db:
        stats = db.get_stats()
        latest = db.get_latest_batch()

        status_counts = stats.get("status_counts", {})
        total = stats.get("file_total", 0)

        click.echo(f"\n📊 转换状态: {raw_path.name}")
        click.echo(f"{'='*40}")
        click.echo(f"  📁 目录数: {stats.get('directory_count', 0)}")
        click.echo(f"  📄 文件总数: {total}")
        click.echo(f"  ✅ 成功: {status_counts.get('success', 0)}")
        click.echo(f"  ❌ 失败: {status_counts.get('failed', 0)}")
        click.echo(f"  ⏳ 待处理: {status_counts.get('pending', 0)}")
        click.echo(f"  ⏭️  跳过: {status_counts.get('skipped', 0)}")
        click.echo(f"  🔄 转换中: {status_counts.get('converting', 0)}")

        if latest:
            click.echo(f"\n📦 最近批次: #{latest['id']}")
            click.echo(f"   类型: {latest.get('batch_type', 'N/A')}")
            click.echo(f"   状态: {latest.get('status', 'N/A')}")
            click.echo(f"   进度: {latest.get('processed', 0)}/{latest.get('total_files', 0)}")
            click.echo(f"   成功: {latest.get('success_count', 0)}")
            click.echo(f"   失败: {latest.get('failed_count', 0)}")
            click.echo(f"   开始: {latest.get('started_at', 'N/A')}")
            if latest.get("finished_at"):
                click.echo(f"   结束: {latest['finished_at']}")

        if detailed:
            # Show file extension breakdown
            click.echo("\n📋 文件类型分布:")
            ext_stats: dict[str, dict[str, int]] = {}
            for status in ("success", "failed", "pending", "skipped"):
                for f in db.get_files_by_status(status):
                    ext = f.get("extension", "unknown")
                    if ext not in ext_stats:
                        ext_stats[ext] = {}
                    ext_stats[ext][status] = ext_stats[ext].get(status, 0) + 1

            for ext, counts in sorted(ext_stats.items()):
                parts = [f"{s}:{c}" for s, c in counts.items()]
                click.echo(f"  {ext:10s} {', '.join(parts)}")


@catalog.command("failed")
@click.argument("raw_dir", type=click.Path(exists=True))
@click.option("--limit", "-l", type=int, default=20, help="显示数量")
def catalog_failed(raw_dir, limit):
    """查看失败文件"""
    raw_path = Path(raw_dir).resolve()
    db_path = raw_path / "convert.db"
    if not db_path.exists():
        click.echo(f"❌ 未找到转换数据库: {db_path}")
        return

    with ConvertDB(db_path) as db:
        failed = db.get_files_by_status("failed")[:limit]
        if not failed:
            click.echo("✅ 没有失败文件")
            return

        click.echo(f"\n❌ 失败文件 ({len(failed)} 个):\n")
        for f in failed:
            error = f.get("last_error", "未知错误")
            # Truncate long errors
            if len(error) > 100:
                error = error[:100] + "..."
            click.echo(f"  • {f['relative_path']}")
            click.echo(f"    错误: {error}")
            click.echo(f"    尝试次数: {f.get('attempt_count', 0)}")


@catalog.command("retry")
@click.argument("raw_dir", type=click.Path(exists=True))
def catalog_retry(raw_dir):
    """重试失败文件"""
    raw_path = Path(raw_dir).resolve()
    db_path = raw_path / "convert.db"
    if not db_path.exists():
        click.echo(f"❌ 未找到转换数据库: {db_path}")
        return

    with ConvertDB(db_path) as db:
        failed = db.get_files_by_status("failed")
        if not failed:
            click.echo("✅ 没有失败文件需要重试")
            return

        for f in failed:
            db.update_file_status(f["id"], "pending", attempt_count=0, last_error=None)

        click.echo(f"🔄 已重置 {len(failed)} 个失败文件为待处理状态")
        click.echo("💡 请运行 batch-convert --mode resume 重新转换")


@catalog.command("reindex")
@click.argument("raw_dir", type=click.Path(exists=True))
def catalog_reindex(raw_dir):
    """重新生成目录索引"""
    raw_path = Path(raw_dir).resolve()
    if not raw_path.exists():
        click.echo(f"❌ 目录不存在: {raw_path}")
        return

    db_path = raw_path / "convert.db"
    if not db_path.exists():
        click.echo(f"❌ 未找到转换数据库: {db_path}")
        return

    # Determine source_root from db config or directory name
    with ConvertDB(db_path) as db:
        source_root_str = db._get_config("source_root")
        if source_root_str:
            source_root = Path(source_root_str)
        else:
            # Infer from batch config
            latest = db.get_latest_batch()
            if latest and latest.get("config_json"):
                try:
                    config = json.loads(latest["config_json"])
                    source_root = Path(config.get("source", raw_path))
                except (json.JSONDecodeError, KeyError):
                    source_root = raw_path
            else:
                source_root = raw_path

        # raw_path is the output root (e.g. D:\docs\raw\DLP案件反馈)
        # raw_root is its parent (e.g. D:\docs\raw)
        raw_root = raw_path.parent
        store = RawStore(source_root, raw_root)

        click.echo("📋 重新生成目录索引...")
        _generate_indexes(db, store)

    click.echo("📋 索引重新生成完成")


@catalog.command("token")
@click.argument("raw_dir", type=click.Path(exists=True))
def catalog_token(raw_dir):
    """查看 Token 使用量统计"""
    raw_path = Path(raw_dir).resolve()
    db_path = raw_path / "convert.db"
    if not db_path.exists():
        click.echo(f"❌ 未找到转换数据库: {db_path}")
        return

    with ConvertDB(db_path) as db:
        try:
            summary = db.get_token_summary()
        except Exception:
            click.echo("❌ token_usage 表不存在（请先运行一次 batch-convert）")
            return

        click.echo(f"\n💰 Token 使用量: {raw_path.name}")
        click.echo(f"{'='*50}")

        if summary.get("total_tokens", 0) == 0:
            click.echo("  (暂无 token 使用记录)")
        else:
            click.echo(f"  输入 tokens:  {summary['input_tokens']:>12,}")
            click.echo(f"  输出 tokens:  {summary['output_tokens']:>12,}")
            click.echo(f"  合计 tokens:  {summary['total_tokens']:>12,}")

            by_model = summary.get("by_model", [])
            if by_model:
                click.echo(f"\n  {'Model':<15s} {'Calls':>6s} {'Input':>10s} {'Output':>10s} {'Total':>10s}")
                click.echo(f"  {'-'*51}")
                for m in by_model:
                    click.echo(
                        f"  {m['model']:<15s} {m['call_count']:>6d} "
                        f"{m['input_tokens']:>10,d} {m['output_tokens']:>10,d} {m['total_tokens']:>10,d}"
                    )

        # Also show OCR files count
        ocr_count = db.conn.execute(
            "SELECT COUNT(*) FROM files WHERE ocr_used=1"
        ).fetchone()[0]
        ocr_with_tokens = db.conn.execute(
            "SELECT COUNT(*) FROM files WHERE ocr_used=1 AND ocr_total_tokens > 0"
        ).fetchone()[0]
        click.echo(f"\n  OCR 文件: {ocr_count} (有 token 记录: {ocr_with_tokens})")


@catalog.command("repair")
@click.argument("raw_dir", type=click.Path(exists=True))
@click.option("--fix", "fix_types", default="all",
              help="修复类型: tables,ocr,tags,headings,all (默认: all)")
@click.option("--dry-run", is_flag=True, help="仅显示将要修复的文件，不实际修改")
@click.option("--backup", is_flag=True, help="修改前备份原文件为 .md.bak")
@click.option("--force", is_flag=True, help="即使 pipeline_version 匹配也重新修复")
def catalog_repair(raw_dir, fix_types, dry_run, backup, force):
    """回溯修复已转换的文档

    对已转换的 .md 文件应用最新的表格修复、OCR 后处理和标签提取，
    无需重新转换或调用外部 API。

    修复类型:
      tables   — 修复表格对齐问题 (适用于 Office/HTML/PDF 文档)
      ocr      — OCR 文本后处理 (适用于图片/扫描 PDF)
      tags     — 重新提取关键词标签 (适用于所有文档)
      headings — 提取文档标题结构到 .md.json (适用于所有文档)
      all      — 应用全部修复 (默认)
    """
    import hashlib

    from src.converter.headings import extract_headings
    from src.converter.ocr_postprocess import postprocess_ocr_result
    from src.converter.table_fix import fix_table_alignment
    from src.converter.tag_extractor import TagExtractor
    from src.storage.convert_db import PIPELINE_VERSION

    raw_path = Path(raw_dir).resolve()
    db_path = raw_path / "convert.db"
    if not db_path.exists():
        click.echo(f"❌ 未找到转换数据库: {db_path}")
        return

    # Parse fix types
    if fix_types == "all":
        apply_tables = True
        apply_ocr = True
        apply_tags = True
        apply_headings = True
    else:
        parts = {p.strip() for p in fix_types.split(",")}
        apply_tables = "tables" in parts
        apply_ocr = "ocr" in parts
        apply_tags = "tags" in parts
        apply_headings = "headings" in parts

    # Extensions eligible for table fix
    TABLE_EXTENSIONS = {".docx", ".doc", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".pdf"}
    # Extensions eligible for OCR postprocess
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}

    tag_extractor = TagExtractor()
    repaired = 0
    skipped = 0
    errors = 0
    skipped_reasons = {"already_current": 0, "no_fix_needed": 0}

    with ConvertDB(db_path) as db:
        files = db.get_files_by_status("success")
        if not files:
            click.echo("✅ 没有已转换的文件需要修复")
            return

        click.echo(f"\n🔧 开始修复: {raw_path.name}")
        click.echo(f"   模式: {fix_types} {'(dry-run) ' if dry_run else ''}")
        click.echo(f"   待检查文件: {len(files)} 个\n")

        for f in files:
            file_id = f["id"]
            rel_path = f["relative_path"]
            ext = f.get("extension", "").lower()
            ocr_used = bool(f.get("ocr_used", 0))
            pipeline_ver = f.get("pipeline_version", "1") or "1"

            # Skip if already at current pipeline version
            if pipeline_ver == PIPELINE_VERSION and not force:
                skipped += 1
                skipped_reasons["already_current"] += 1
                continue

            # Determine output .md path
            output_path_str = f.get("output_path")
            if output_path_str:
                md_path = Path(output_path_str)
                if not md_path.is_absolute():
                    md_path = raw_path / md_path
            else:
                md_path = raw_path / (rel_path + ".md")

            if not md_path.exists():
                errors += 1
                click.echo(f"  ⚠️ 文件不存在: {md_path}")
                continue

            # Read markdown content
            try:
                content = md_path.read_text(encoding="utf-8")
            except Exception as e:
                errors += 1
                click.echo(f"  ⚠️ 读取失败: {rel_path} — {e}")
                continue

            # Strip frontmatter before processing to avoid YAML corruption (H2 fix)
            from src.converter.frontmatter import inject_frontmatter, strip_frontmatter
            fm_existed, content = strip_frontmatter(content)

            modified = False
            new_content = content
            current_tag_result = None
            current_headings = None

            # Table fix
            if apply_tables and ext in TABLE_EXTENSIONS:
                fixed = fix_table_alignment(content)
                if fixed != content:
                    new_content = fixed
                    modified = True

            # OCR postprocess
            if apply_ocr and (ocr_used or ext in IMAGE_EXTENSIONS):
                processed = postprocess_ocr_result(new_content)
                if processed != new_content:
                    new_content = processed
                    modified = True

            # Tag extraction
            if apply_tags and new_content.strip():
                try:
                    current_tag_result = tag_extractor.extract(
                        markdown=new_content,
                        filename=rel_path,
                    )
                    modified = True  # Always update tags/metadata
                except Exception as e:
                    click.echo(f"  ⚠️ 标签提取失败: {rel_path} — {e}")

            # Headings extraction (zero LLM, regex only — always worth running)
            if apply_headings and new_content.strip():
                try:
                    current_headings = extract_headings(new_content)
                    if current_headings:
                        modified = True
                except Exception as e:
                    click.echo(f"  ⚠️ 标题提取失败: {rel_path} — {e}")
                    current_headings = None
            else:
                current_headings = None

            if not modified and not (apply_headings and current_headings):
                skipped += 1
                skipped_reasons["no_fix_needed"] += 1
                continue

            if dry_run:
                click.echo(f"  📋 [dry-run] 将修复: {rel_path}")
                repaired += 1
                continue

            # Backup if requested
            if backup:
                bak_path = Path(str(md_path) + ".bak")
                try:
                    import shutil
                    shutil.copy2(str(md_path), str(bak_path))
                except Exception as e:
                    click.echo(f"  ⚠️ 备份失败: {rel_path} — {e}")

            # Load .md.json metadata before re-injecting frontmatter
            md_json_path = Path(str(md_path) + ".json")
            metadata = {}
            if md_json_path.exists():
                try:
                    metadata = json.loads(md_json_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}

            if apply_tags and current_tag_result is not None:
                metadata["tags"] = current_tag_result.tags
                metadata["doc_type"] = current_tag_result.doc_type
                metadata["keywords"] = current_tag_result.keywords
                metadata["tag_confidence"] = current_tag_result.confidence

            if apply_headings and current_headings is not None:
                metadata["headings"] = current_headings

            # Re-inject frontmatter after processing (H2 fix)
            if fm_existed or metadata:
                fm_meta = {
                    "title": Path(rel_path).stem,
                    "doc_type": metadata.get("doc_type", "document"),
                    "source": rel_path,
                    "tags": metadata.get("tags", []),
                    "headings": metadata.get("headings", []),
                }
                new_content = inject_frontmatter(new_content, fm_meta)

            # Write modified .md file
            try:
                md_path.write_text(new_content, encoding="utf-8")
            except Exception as e:
                errors += 1
                click.echo(f"  ⚠️ 写入失败: {rel_path} — {e}")
                continue

            metadata["pipeline_version"] = PIPELINE_VERSION

            try:
                md_json_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                click.echo(f"  ⚠️ 元数据写入失败: {rel_path} — {e}")

            # Update ConvertDB
            output_size = len(new_content.encode("utf-8"))
            output_hash = hashlib.md5(new_content.encode("utf-8")).hexdigest()
            metadata_json = json.dumps(metadata, ensure_ascii=False)

            db.update_file_status(
                file_id,
                "success",
                output_size=output_size,
                output_hash=output_hash,
                metadata_json=metadata_json,
                pipeline_version=PIPELINE_VERSION,
            )

            repaired += 1
            click.echo(f"  ✅ 已修复: {rel_path}")

    # Summary
    click.echo(f"\n{'='*40}")
    click.echo("📊 修复完成:")
    click.echo(f"  ✅ 已修复: {repaired}")
    click.echo(f"  ⏭️  跳过: {skipped} (已最新: {skipped_reasons['already_current']}, 无需修复: {skipped_reasons['no_fix_needed']})")
    if errors:
        click.echo(f"  ❌ 错误: {errors}")
    if dry_run:
        click.echo("  📋 (dry-run 模式，未实际修改文件)")


@catalog.command("backfill-headings")
@click.argument("raw_dir", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="仅显示将要更新的文件，不实际修改")
@click.option("--force", is_flag=True, help="即使已有 headings 也重新提取")
def catalog_backfill_headings(raw_dir, dry_run, force):
    """为已有知识库回填标题结构元数据（无需重新转换）

    扫描 raw 目录下所有 .md 文件，提取 Markdown 标题层级结构，
    写入对应的 .md.json 元数据文件。不修改 .md 文件内容，
    不重新执行格式转换，仅升级元数据。

    适用于已有知识库增量升级，配合 Agent 的文档结构感知功能。
    """
    from src.converter.headings import extract_headings

    raw_path = Path(raw_dir).resolve()
    md_files = list(raw_path.rglob("*.md"))
    # Exclude _index.md helper files
    md_files = [f for f in md_files if f.name != "_index.md"]

    if not md_files:
        click.echo("❌ 未找到 .md 文件")
        return

    click.echo(f"\n📂 扫描目录: {raw_path.name}")
    click.echo(f"   .md 文件: {len(md_files)} 个")
    click.echo(f"   模式: {'dry-run' if dry_run else '写入'}\n")

    updated = 0
    skipped = 0
    no_headings = 0
    errors = 0

    for md_file in md_files:
        md_json_path = Path(str(md_file) + ".json")
        rel = md_file.relative_to(raw_path)

        # Read markdown content
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            errors += 1
            click.echo(f"  ⚠️ 读取失败: {rel} — {e}")
            continue

        # Extract headings
        headings = extract_headings(content)

        if not headings:
            no_headings += 1
            continue

        # Check if already has headings
        if md_json_path.exists() and not force:
            try:
                existing = json.loads(md_json_path.read_text(encoding="utf-8"))
                if existing.get("headings"):
                    skipped += 1
                    continue
            except Exception:
                logger.warning("Failed to read existing headings from %s", md_json_path)

        if dry_run:
            click.echo(f"  📋 [dry-run] {rel} → {len(headings)} 个标题")
            updated += 1
            continue

        # Load or create .md.json
        metadata = {}
        if md_json_path.exists():
            try:
                metadata = json.loads(md_json_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        # Update headings
        metadata["headings"] = headings

        # Write .md.json
        try:
            md_json_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            errors += 1
            click.echo(f"  ⚠️ 写入失败: {rel} — {e}")
            continue

        updated += 1

    # Summary
    click.echo(f"\n{'='*40}")
    click.echo("📊 标题回填完成:")
    click.echo(f"  ✅ 已更新: {updated}")
    click.echo(f"  ⏭️  已有标题跳过: {skipped}")
    click.echo(f"  📄 无标题文档: {no_headings}")
    if errors:
        click.echo(f"  ❌ 错误: {errors}")
    if dry_run:
        click.echo("  📋 (dry-run 模式，未实际修改文件)")


@catalog.command("inject-frontmatter")
@click.argument("raw_dir", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="仅显示将要更新的文件，不实际修改")
@click.option("--force", is_flag=True, help="即使已有 frontmatter 也重新生成")
def catalog_inject_frontmatter(raw_dir, dry_run, force):
    """为已有知识库注入 YAML frontmatter（无需重新转换）

    扫描 raw 目录下所有 .md 文件（排除 _index.md），
    从对应的 .md.json 元数据生成 frontmatter 并注入 .md 文件头部。
    不重新执行格式转换，仅增强 .md 文件的可读性和 OKF 兼容性。

    检测逻辑：读取 .md 文件，检查是否以 ---\\n 开头（已有 frontmatter）。
    如已有且未指定 --force，则跳过。

    \b
    示例:
        catalog inject-frontmatter "./my-raw"
        catalog inject-frontmatter "./my-raw" --dry-run
        catalog inject-frontmatter "./my-raw" --force
    """
    from src.converter.frontmatter import has_frontmatter, inject_frontmatter, strip_frontmatter

    raw_path = Path(raw_dir).resolve()
    md_files = list(raw_path.rglob("*.md"))
    # Exclude _index.md helper files
    md_files = [f for f in md_files if f.name != "_index.md"]

    if not md_files:
        click.echo("❌ 未找到 .md 文件")
        return

    click.echo(f"\n📂 扫描目录: {raw_path.name}")
    click.echo(f"   .md 文件: {len(md_files)} 个")
    click.echo(f"   模式: {'dry-run' if dry_run else '写入'}\n")

    injected = 0
    skipped_has_fm = 0
    skipped_no_json = 0
    errors = 0

    for md_file in md_files:
        md_json_path = Path(str(md_file) + ".json")
        rel = md_file.relative_to(raw_path)

        # Read markdown content
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            errors += 1
            click.echo(f"  ⚠️ 读取失败: {rel} — {e}")
            continue

        # Check if frontmatter already exists
        if has_frontmatter(content) and not force:
            skipped_has_fm += 1
            continue

        # Load metadata from .md.json
        metadata = {}
        if md_json_path.exists():
            with contextlib.suppress(Exception):
                metadata = json.loads(md_json_path.read_text(encoding="utf-8"))

        if not metadata:
            # No .md.json — generate minimal frontmatter from filename
            metadata = {
                "title": md_file.stem,
                "doc_type": "document",
                "source": md_file.name,
                "tags": [],
            }
            skipped_no_json += 1

        # Ensure title is set (md.json doesn't store a top-level title key)
        if not metadata.get("title"):
            metadata["title"] = md_file.stem

        # Ensure headings are available
        if "headings" not in metadata:
            body = strip_frontmatter(content)[1] if has_frontmatter(content) else content
            from src.converter.headings import extract_headings
            metadata["headings"] = extract_headings(body)

        # Add timestamp if missing
        if "converted_at" not in metadata:
            from datetime import datetime as _dt
            metadata["converted_at"] = _dt.now().strftime("%Y-%m-%dT%H:%M:%S")

        if dry_run:
            fm_status = "重新生成" if has_frontmatter(content) else "新增"
            click.echo(f"  📋 [dry-run] {rel} — {fm_status}")
            injected += 1
            continue

        # Inject frontmatter
        try:
            new_content = inject_frontmatter(content, metadata)
            md_file.write_text(new_content, encoding="utf-8")
            injected += 1
        except Exception as e:
            errors += 1
            click.echo(f"  ⚠️ 注入失败: {rel} — {e}")

    click.echo(f"\n{'='*40}")
    click.echo("📊 Frontmatter 注入完成:")
    click.echo(f"  ✅ 注入/更新: {injected}")
    click.echo(f"  ⏭️  已有 frontmatter 跳过: {skipped_has_fm}")
    click.echo(f"  📄 无 .md.json (最小元数据): {skipped_no_json}")
    if errors:
        click.echo(f"  ❌ 错误: {errors}")
    if dry_run:
        click.echo("  📋 (dry-run 模式，未实际修改文件)")


# ── build-index 命令 ─────────────────────────────

import hashlib as _hashlib
import shutil as _shutil


def _sample_content(content: str, file_size: int, max_chars: int = 50000, sample_threshold: int = 5 * 1024 * 1024) -> tuple[str, str]:
    """Smart content sampling for large files.

    Returns (sampled_content, strategy) where strategy is one of:
      'full'      - content fits within limit, no truncation
      'truncated' - simple head truncation (file <= threshold)
      'sampled'   - head/middle/tail sampling (file > threshold)
    """
    if len(content) <= max_chars:
        return content, "full"

    if file_size <= sample_threshold:
        return content[:max_chars] + "\n... (内容已截断)", "truncated"

    # Large files: sample beginning, middle, end
    head_size = 20000
    tail_size = 10000
    mid_size = 10000
    mid_pos = len(content) // 2

    parts = [
        content[:head_size],
        "\n\n... (前部截断) ...\n\n",
        content[mid_pos:mid_pos + mid_size],
        "\n\n... (中部截断) ...\n\n",
        content[-tail_size:],
    ]
    return "".join(parts), "sampled"


@cli.command("diff-migrate")
@click.argument("base_dir", type=click.Path(exists=True))
@click.argument("compare_dir", type=click.Path(exists=True))
@click.option("--export-new", "-e", type=click.Path(), default=None,
              help="将新增文件导出到指定目录 (保留相对路径结构)")
@click.option("--extensions", type=str, default=None,
              help="对比的文件扩展名 (逗号分隔, 如: .pdf,.docx; 默认全部支持格式)")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="导出差异报告 (.json/.csv/.md)")
@click.option("--verbose", "-v", is_flag=True, help="显示未变更文件")
@click.option("--all-files", is_flag=True, default=False,
              help="对比所有文件 (不仅限于支持的文档格式)")
def diff_migrate(base_dir, compare_dir, export_new, extensions,
                 output, verbose, all_files):
    """目录增量对比 (保护模式: 仅对比+导出, 不自动复制)

    BASE_DIR 为基准目录 (现有知识库原始内容),
    COMPARE_DIR 为对比目录 (可能包含新增内容).

    只关注对比目录中真正新增的文件:
    - 🟢 新增: 对比目录有但基准没有的全新内容
    - 📦 位移: 内容相同仅路径不同 (忽略不计)
    - 🔄 变更: 同路径但内容不同
    - 🔴 删除: 基准有但对比目录没有 (基准只增不减, 仅供参考)

    使用 --export-new 将 🟢新增 文件导出到指定目录,
    位移/变更/删除文件不会导出.

    \b
    示例:
        doc-search diff-migrate "./source/info-tech" "./new/info-tech"
        doc-search diff-migrate "./source/info-tech" "./new/info-tech" --extensions ".pdf,.docx"
        doc-search diff-migrate "./source/info-tech" "./new/info-tech" -o report.json
        doc-search diff-migrate "./source/info-tech" "./new/info-tech" -e "D:\\export" -o report.csv
    """
    base_path = Path(base_dir).resolve()
    compare_path = Path(compare_dir).resolve()

    # Parse extensions filter
    ext_set: set | None = None
    if extensions:
        ext_set = set()
        for ext in extensions.split(","):
            ext = ext.strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = "." + ext
            ext_set.add(ext)
    elif not all_files:
        ext_set = SUPPORTED_EXTENSIONS

    # Header
    ext_desc = f"{len(ext_set)} 种格式" if ext_set else "所有文件"
    click.echo(f"📁 基准目录: {base_path}")
    click.echo(f"📁 比较目录: {compare_path}")
    click.echo(f"📎 对比范围: {ext_desc}")
    click.echo("")

    # Compare
    diff_result = compare_directories(base_path, compare_path, extensions=ext_set)

    # Build display entries (sorted: added, moved, changed, deleted, then unchanged if verbose)
    display_entries = (
        diff_result.added
        + diff_result.moved
        + diff_result.changed
        + diff_result.deleted
    )
    if verbose:
        display_entries = display_entries + diff_result.unchanged

    # Status icons and colors
    status_config = {
        "added": ("🟢", "green"),
        "moved": ("📦", "blue"),
        "changed": ("🔄", "yellow"),
        "deleted": ("🔴", "red"),
        "unchanged": ("⬜", "dim"),
    }

    # Rich table or plain text
    if _RICH_AVAILABLE and console:
        table = Table(title="📋 目录差异对比", show_lines=False)
        table.add_column("状态", width=6)
        table.add_column("文件路径", style="white")
        table.add_column("基准大小", justify="right", style="dim")
        table.add_column("比较大小", justify="right", style="dim")

        for entry in display_entries:
            icon, color = status_config.get(entry.status, ("❓", "white"))
            base_size = _format_file_size(entry.base.size) if entry.base else "-"
            compare_size = _format_file_size(entry.compare.size) if entry.compare else "-"
            table.add_row(
                f"[{color}]{icon}[/{color}]",
                entry.relative_path,
                base_size,
                compare_size,
            )

        console.print(table)
    else:
        # Plain text fallback
        click.echo(f"{'状态':<8} {'文件路径':<50} {'基准大小':>10} {'比较大小':>10}")
        click.echo("-" * 82)
        for entry in display_entries:
            icon, _ = status_config.get(entry.status, ("❓", ""))
            base_size = _format_file_size(entry.base.size) if entry.base else "-"
            compare_size = _format_file_size(entry.compare.size) if entry.compare else "-"
            click.echo(f"{icon:<8} {entry.relative_path:<50} {base_size:>10} {compare_size:>10}")

    # Summary
    summary = diff_result.summary()
    click.echo("")
    click.echo(
        f"📊 对比结果: "
        f"🟢 新增: {summary['added']} | "
        f"📦 位移: {summary['moved']} | "
        f"🔄 变更: {summary['changed']} | "
        f"🔴 删除: {summary['deleted']} | "
        f"⬜ 未变: {summary['unchanged']}"
    )

    # Export new files to target directory
    if export_new:
        export_path = Path(export_new).resolve()
        added_entries = diff_result.added
        click.echo("")
        click.echo(f"📂 导出新增文件到: {export_path}")
        if not added_entries:
            click.echo("  ℹ️  无新增文件, 跳过导出")
        else:
            copied = 0
            errors: list = []
            for entry in added_entries:
                src = compare_path / entry.relative_path
                dest = export_path / entry.relative_path
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    _shutil.copy2(str(src), str(dest))
                    copied += 1
                except OSError as e:
                    errors.append((entry.relative_path, str(e)))
            click.echo(f"  ✅ 已导出: {copied}")
            if errors:
                click.echo(f"  ❌ 错误:   {len(errors)}")
                for rel_path, err_msg in errors:
                    click.echo(f"     {rel_path}: {err_msg}")

    # Export report
    if output:
        output_path = Path(output).resolve()
        suffix = output_path.suffix.lower()

        if suffix == ".json":
            report_data = []
            for entry in diff_result.entries:
                item = {
                    "relative_path": entry.relative_path,
                    "status": entry.status,
                    "base_size": entry.base.size if entry.base else None,
                    "compare_size": entry.compare.size if entry.compare else None,
                }
                if entry.moved_from:
                    item["moved_from"] = entry.moved_from
                report_data.append(item)
            output_path.write_text(
                json.dumps(report_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif suffix == ".csv":
            lines = ["relative_path,status,base_size,compare_size,moved_from"]
            for entry in diff_result.entries:
                base_size = str(entry.base.size) if entry.base else ""
                compare_size = str(entry.compare.size) if entry.compare else ""
                moved_from = entry.moved_from or ""
                lines.append(
                    f"{entry.relative_path},{entry.status},{base_size},{compare_size},"
                    f"{moved_from}"
                )
            output_path.write_text("\n".join(lines), encoding="utf-8")
        elif suffix == ".md":
            lines = [
                "# 目录差异报告",
                "",
                f"- **基准目录**: `{base_path}`",
                f"- **比较目录**: `{compare_path}`",
                f"- **新增**: {summary['added']} | **位移**: {summary['moved']} | "
                f"**变更**: {summary['changed']} | "
                f"**删除**: {summary['deleted']} | **未变**: {summary['unchanged']}",
                "",
                "| 状态 | 文件路径 | 基准大小 | 比较大小 |",
                "|------|----------|----------|----------|",
            ]
            for entry in diff_result.entries:
                icon, _ = status_config.get(entry.status, ("❓", ""))
                base_size = _format_file_size(entry.base.size) if entry.base else "-"
                compare_size = _format_file_size(entry.compare.size) if entry.compare else "-"
                lines.append(
                    f"| {icon} | {entry.relative_path} | {base_size} | {compare_size} |"
                )
            output_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            click.echo(f"⚠️  不支持的报告格式: {suffix} (支持 .json/.csv/.md)")
            return

        click.echo(f"📄 报告已导出: {output_path}")


@cli.command("build-index")
@click.argument("raw_dir", type=click.Path(exists=True))
@click.option("--max-content-size", type=int, default=50000, help="单文档最大内容字符数（默认: 50000）")
@click.option("--sample-threshold", type=int, default=5 * 1024 * 1024, help="采样阈值字节数（默认: 5MB）")
@click.option("--chunk-mode", is_flag=True, default=False, help="长文档按标题切分索引（>chunk-min-size 时生效）")
@click.option("--chunk-min-size", type=int, default=50000, help="切分阈值(字符数)，默认 50000")
def build_index(raw_dir, max_content_size, sample_threshold, chunk_mode, chunk_min_size):
    """构建 Tantivy 全文搜索索引

    扫描 raw_dir 中的 .md 文件，应用智能内容采样 (大文件自动截断/分段采样)，
    建立 Tantivy BM25 索引到 raw_dir/index/ 目录。

    启用 ``--chunk-mode`` 后，超过 ``--chunk-min-size`` 字符的长文档会按
    H2/H3 标题切分为多个独立索引条目，每条 ``doc_id`` 后缀 ``#c{N}``，
    ``title`` 格式 ``{文件名} § {标题}``。搜索天然命中 chunk 级条目，
    无需修改 Schema。

    示例:
        doc-search build-index "./my-raw"
        doc-search build-index "./my-raw" --chunk-mode
        doc-search build-index "./my-raw" --chunk-mode --chunk-min-size 30000
    """
    raw_path = Path(raw_dir).resolve()
    index_path = raw_path / "index"

    # Clean old index
    if index_path.exists():
        click.echo(f"Removing old index: {index_path}")
        _shutil.rmtree(index_path)

    index_mgr = TantivyIndexManager(index_path=index_path, use_jieba=True)
    md_files = [f for f in raw_path.rglob("*.md") if not f.name.startswith("_")]

    if not md_files:
        click.echo("No markdown files found in directory.")
        return

    click.echo(f"Indexing {len(md_files)} files from {raw_path}")

    # Statistics tracking
    count = 0
    truncated_count = 0
    sampled_count = 0
    skipped_files: list[str] = []

    # Chunk-mode helpers (imported lazily to avoid circular deps)
    if chunk_mode:
        from src.converter.headings import extract_headings
        from src.storage.chunker import split_into_chunks

    start = time.time()
    for md_file in md_files:
        try:
            file_size = md_file.stat().st_size
            content = md_file.read_text(encoding="utf-8")
            if not content.strip():
                skipped_files.append(f"{md_file.name} (empty)")
                continue

            # Strip YAML frontmatter before indexing (prevents BM25 pollution)
            from src.converter.frontmatter import strip_frontmatter
            _, content = strip_frontmatter(content)

            content, strategy = _sample_content(content, file_size, max_content_size, sample_threshold)
            if strategy == "truncated":
                truncated_count += 1
                click.echo(f"  TRUNCATED: {md_file.name} ({file_size // 1024}KB)")
            elif strategy == "sampled":
                sampled_count += 1
                click.echo(f"  SAMPLED: {md_file.name} ({file_size // 1024 // 1024}MB)")

            rel_path = md_file.relative_to(raw_path)
            base_doc_id = _hashlib.sha256(rel_path.as_posix().encode()).hexdigest()[:16]

            # ── Chunk mode: split long documents ────────────────
            if chunk_mode and len(content) > chunk_min_size:
                headings = extract_headings(content)
                chunks = split_into_chunks(content, headings, chunk_min_size)
                for idx, (chunk_title, chunk_text) in enumerate(chunks):
                    index_mgr.add_document(
                        doc_id=f"{base_doc_id}#c{idx}",
                        title=f"{md_file.stem} § {chunk_title}",
                        content=chunk_text,
                        metadata={
                            "filename": md_file.name,
                            "source_path": str(rel_path),
                            "chunk_index": idx,
                        },
                    )
                    count += 1
                click.echo(f"  CHUNKED: {md_file.name} → {len(chunks)} chunks")
            else:
                index_mgr.add_document(
                    doc_id=base_doc_id,
                    title=md_file.stem,
                    content=content,
                    metadata={
                        "filename": md_file.name,
                        "source_path": str(rel_path),
                    },
                )
                count += 1

            if count % 100 == 0:
                click.echo(f"  {count}/{len(md_files)} files...")
        except Exception as e:
            skipped_files.append(f"{md_file.name} (error: {e})")
            click.echo(f"  Error: {md_file.name}: {e}")

    index_mgr.commit()
    elapsed = time.time() - start
    stats = index_mgr.get_stats()

    # Summary
    click.echo("\n--- Index Summary ---")
    click.echo(f"Indexed: {count} docs in {elapsed:.1f}s ({elapsed / max(count, 1):.2f}s/doc)")
    click.echo(f"Truncated (<=threshold, head only): {truncated_count}")
    click.echo(f"Sampled (>threshold, head+mid+tail): {sampled_count}")
    if chunk_mode:
        click.echo(f"Chunk mode: ON (threshold {chunk_min_size} chars)")
    click.echo(f"Skipped: {len(skipped_files)}")
    for s in skipped_files:
        click.echo(f"  - {s}")
    click.echo(f"Index: {stats['num_docs']} docs at {stats['index_path']}")
    index_mgr.close()


# ── watch 命令 ──────────────────────────────────────────

@cli.command("watch")
@click.argument("raw_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--debounce", type=float, default=1.0, help="防抖间隔（秒），默认 1.0")
@click.option("--no-jieba", is_flag=True, help="禁用 jieba 中文分词")
@click.option("--log-file", type=click.Path(), default=None, help="日志文件路径（Task Scheduler 环境下推荐使用）")
@click.option("--chunk-mode", is_flag=True, default=False, help="长文档按标题切分索引")
@click.option("--chunk-min-size", type=int, default=50000, help="切分阈值(字符数)，默认 50000")
def watch_directory(raw_dir, debounce, no_jieba, log_file, chunk_mode, chunk_min_size):
    """监控 raw 目录，自动增量更新 Tantivy 索引。

    当 raw 目录中的 .md 文件被创建、修改或删除时，
    自动同步更新对应的索引。适用于批量转换后持续
    维护索引的场景。

    示例:
        doc-search watch "./my-raw"
        doc-search watch "./my-raw" --debounce 2.0
        doc-search watch "./my-raw" --log-file "D:\\logs\\watch.log"
    """
    import logging as _logging

    if log_file:
        _log_path = Path(log_file).resolve()
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _fh = _logging.FileHandler(str(_log_path), encoding="utf-8", mode="a")
        _fh.setLevel(_logging.DEBUG)
        _fh.setFormatter(_logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _logging.getLogger().addHandler(_fh)
        _logging.getLogger().setLevel(_logging.DEBUG)

    from src.watch import IndexWatcher

    raw_path = Path(raw_dir).resolve()
    index_path = raw_path / "index"

    click.echo(f"👁️  监控目录: {raw_path}")
    click.echo(f"📇 索引路径: {index_path}")
    click.echo(f"⏱️  防抖间隔: {debounce}s")
    click.echo(f"🔤 中文分词: {'禁用' if no_jieba else '启用 (jieba)'}")
    if log_file:
        click.echo(f"📝 日志文件: {log_file}")
    click.echo()
    click.echo("按 Ctrl+C 停止监控...")
    click.echo()

    watcher = IndexWatcher(
        raw_dir=str(raw_path),
        debounce_seconds=debounce,
        use_jieba=not no_jieba,
        chunk_mode=chunk_mode,
        chunk_min_size=chunk_min_size,
    )

    try:
        watcher.start(blocking=True)
    except KeyboardInterrupt:
        click.echo("\n⏹️  停止监控...")
        watcher.stop()
        stats = watcher.stats
        if stats:
            click.echo(f"📊 最终统计: {stats.summary()}")


# ── tui 命令 ──────────────────────────────────────────


@cli.command()
@click.option("--index", "-i", type=click.Path(exists=True), required=True,
              help="索引目录路径")
@click.option("--raw-dir", type=click.Path(exists=True), default=None,
              help="原始文档目录路径 (启用 GrepTool/BashTool)")
@click.option("--pi", is_flag=True, default=False,
              help="使用 Pi TUI (Node.js) 替代内置 Textual TUI")
@click.option("--api-port", type=int, default=0,
              help="HTTP API 端口 (默认自动分配, 仅 --pi 模式)")
@click.option("--model", default="glm-5.1",
              help="Pi 使用的 LLM 模型 (仅 --pi 模式, 默认 glm-5.1)")
@click.option("--thinking", default="medium",
              type=click.Choice(["off", "minimal", "low", "medium", "high", "xhigh"]),
              help="Pi thinking 级别 (仅 --pi 模式)")
@click.option("--web", is_flag=True, default=False,
              help="启动 Web 浏览器界面 (--host/--port 可配置)")
@click.option("--host", default="127.0.0.1",
              help="Web 服务绑定地址 (默认 127.0.0.1)")
@click.option("--port", type=int, default=8000,
              help="Web 服务端口 (默认 8000)")
@click.option("--open/--no-open", "open_browser", default=True,
              help="启动后自动打开浏览器 (默认 --open)")
def tui(index, raw_dir, pi, api_port, model, thinking, web, host, port, open_browser):
    """启动交互式 TUI 界面

    默认使用内置 Textual TUI。加 --pi 使用 Node.js Pi TUI。加 --web 启动浏览器界面。

    ╔══════════════════════════════════════════════════════════════╗
    ║  注意: Textual TUI 和 Pi TUI 均已弃用                      ║
    ║  推荐使用 --web 启动浏览器界面                             ║
    ╚══════════════════════════════════════════════════════════════╝

    示例:
        doc-search tui -i ./output/index --web --port 8080
        doc-search tui -i ./output/index --raw-dir ./output --web
    """
    if not web:
        click.echo("⚠️  Textual/Pi TUI 已弃用，推荐使用 --web 启动浏览器界面", err=True)

    if web:
        import uvicorn

        from src.api import app

        # Pre-set index path and raw dir as defaults for the web UI
        index_abs = str(Path(index).resolve())
        url = f"http://{host}:{port}"
        click.echo("doc-search Web 模式")
        click.echo(f"  索引: {index_abs}")
        if raw_dir:
            click.echo(f"  Raw:  {Path(raw_dir).resolve()}")
        click.echo(f"  地址: {url}")
        click.echo(f"  Web:  {url}/")
        click.echo(f"  API:  {url}/docs")

        if open_browser:
            import threading
            import webbrowser
            from urllib.parse import urlencode
            params = {"index_path": index_abs}
            if raw_dir:
                params["raw_dir"] = str(Path(raw_dir).resolve())
            browser_url = f"{url}/?{urlencode(params)}"
            def _open():
                import time
                time.sleep(1.5)
                webbrowser.open(browser_url)
            threading.Thread(target=_open, daemon=True).start()
            click.echo("  浏览器: 自动打开中...")

        click.echo("\n按 Ctrl+C 停止服务")
        uvicorn.run(app, host=host, port=port, log_level="warning")
    elif pi:
        from src.pi_bridge import PiBridge

        raw_path = Path(raw_dir).resolve() if raw_dir else Path(index).resolve().parent
        bridge = PiBridge(
            index_path=Path(index).resolve(),
            raw_dir=raw_path,
            api_port=api_port,
            model=model,
            thinking=thinking,
        )
        try:
            bridge.start()
        except KeyboardInterrupt:
            pass
        finally:
            bridge.stop()
    else:
        from src.tui import DocSearchTUI

        app = DocSearchTUI(
            index_path=Path(index).resolve(),
            raw_dir=Path(raw_dir).resolve() if raw_dir else None,
        )
        app.run()


# ── benchmark 命令 ────────────────────────────────


@cli.command()
@click.argument("queries", type=click.Path(exists=True))
@click.option("--index", "-i", type=click.Path(exists=True), required=True,
              help="索引目录或 raw 目录路径")
@click.option("--modes", type=str, default="bm25,grep", help="搜索模式 (逗号分隔, 默认: bm25,grep)")
@click.option("--limit", "-l", type=int, default=10, help="结果数量")
@click.option("--runs", type=int, default=3, help="每个查询重复次数")
@click.option("--warmup", type=int, default=1, help="预热次数")
@click.option("--output", "-o", type=click.Path(), default=None, help="输出文件路径 (根据扩展名自动选择格式: .json/.html/.md)")
def benchmark(queries, index, modes, limit, runs, warmup, output):
    """基准测试: 比较不同搜索模式的性能和质量

    查询文件格式 (JSONL, 每行一个):
        {"query": "年假如何申请", "expected_files": ["年假制度.docx.md"], "category": "hr"}

    示例:
        doc-search benchmark queries.jsonl -i "./my-index"
        doc-search benchmark queries.jsonl -i ./index --modes bm25,grep -o report.html
        doc-search benchmark queries.jsonl -i ./index --runs 5 -o report.md
    """
    from src.search.benchmark import BenchmarkRunner
    from src.search.report import BenchmarkReporter

    index_path = Path(index).resolve()

    # Auto-detect: if user points at raw dir (not index/), use parent/index logic
    if _is_tantivy_index(index_path):
        actual_index = index_path
        raw_dir = index_path.parent
    else:
        # Could be a raw directory — look for index/ sub-dir
        sub_index = index_path / "index"
        if _is_tantivy_index(sub_index):
            actual_index = sub_index
            raw_dir = index_path
        else:
            # Treat as raw dir anyway (grep-only mode will still work)
            actual_index = sub_index
            raw_dir = index_path

    # Load queries from JSONL
    try:
        query_specs = BenchmarkRunner.load_queries(Path(queries))
    except Exception as e:
        click.echo(f"❌ 加载查询文件失败: {e}")
        sys.exit(1)

    if not query_specs:
        click.echo("❌ 查询文件为空或格式错误")
        sys.exit(1)

    # Parse modes
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]

    click.echo("📊 搜索基准测试")
    click.echo(f"   索引: {actual_index}")
    click.echo(f"   Raw:  {raw_dir}")
    click.echo(f"   查询: {len(query_specs)} 个")
    click.echo(f"   模式: {', '.join(mode_list)}")
    click.echo(f"   重复: {runs} 次 | 预热: {warmup} 次")
    click.echo()

    # Run benchmark
    runner = BenchmarkRunner(index_path=actual_index, raw_dir=raw_dir)

    if _RICH_AVAILABLE:
        with console.status("[bold cyan]🔄 运行基准测试...[/bold cyan]", spinner="dots"):
            result = runner.run(query_specs, modes=mode_list, runs=runs, warmup=warmup)
    else:
        click.echo("🔄 运行基准测试...")
        result = runner.run(query_specs, modes=mode_list, runs=runs, warmup=warmup)

    click.echo(f"✅ 测试完成 (总耗时: {result.total_time:.2f}s)")
    click.echo()

    # Determine output format
    if output:
        ext = Path(output).suffix.lower()
        if ext == ".json":
            fmt = "json"
        elif ext in (".html", ".htm"):
            fmt = "html"
        elif ext in (".md", ".markdown"):
            fmt = "markdown"
        else:
            fmt = "text"
    else:
        fmt = "text"

    # Generate report
    reporter = BenchmarkReporter()
    report_str = reporter.generate(result, fmt=fmt)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_str, encoding="utf-8")
        click.echo(f"📄 报告已写入: {out_path}")
    else:
        click.echo(report_str)


# ── ab-test 命令 ──────────────────────────────────────────


@cli.command()
@click.option("--cases", type=click.Path(exists=True),
              default=lambda: str(Path(__file__).resolve().parent.parent.parent / "docs" / "qa_benchmark_cases.json"),
              help="QA 案例文件路径 (默认 docs/qa_benchmark_cases.json)")
@click.option("--index-a", type=click.Path(exists=True), required=True, help="A 臂索引路径")
@click.option("--raw-a", type=click.Path(exists=True), default=None, help="A 臂 raw 目录")
@click.option("--mode-a", type=str, default="tool_loop", help="A 臂模式 (tool_loop/pipeline)")
@click.option("--name-a", type=str, default="A", help="A 臂名称")
@click.option("--index-b", type=click.Path(exists=True), required=True, help="B 臂索引路径")
@click.option("--raw-b", type=click.Path(exists=True), default=None, help="B 臂 raw 目录")
@click.option("--mode-b", type=str, default="pipeline", help="B 臂模式 (tool_loop/pipeline)")
@click.option("--name-b", type=str, default="B", help="B 臂名称")
@click.option("--runs", type=int, default=1, help="每臂运行轮次")
@click.option("--limit", type=int, default=None, help="限制查询数量")
@click.option("--domain", type=str, default=None, help="筛选领域 (如 人事管理)")
@click.option("--difficulty", type=str, default=None, help="筛选难度 (easy/medium/hard)")
@click.option("--output", "-o", type=click.Path(), default=None, help="输出 JSON 结果路径")
@click.option("--seed", type=int, default=42, help="随机种子 (用于可重复顺序)")
def ab_test(cases, index_a, raw_a, mode_a, name_a,
            index_b, raw_b, mode_b, name_b,
            runs, limit, domain, difficulty, output, seed):
    r"""A/B 测试: 对比两个 Agent 配置的搜索质量与性能.

    从 QA 案例文件中读取测试问题，分别在 A/B 两个配置上运行，
    统计数据显著性差异。

    示例:

        doc-search ab-test --index-a ./index --mode-a tool_loop --name-a "CLI Agent" \e
            --index-b ./index --mode-b pipeline --name-b "MCP Pipeline" \e
            --runs 3 --limit 20 -o results.json
    """
    from src.search.ab_testing import (
        ABTestRunner,
        RunnerConfig,
        load_queries_from_benchmark,
    )

    click.echo("📊 A/B 测试")
    click.echo(f"   A: {name_a} [{mode_a}] @ {index_a}")
    click.echo(f"   B: {name_b} [{mode_b}] @ {index_b}")
    click.echo(f"   案例: {cases}")
    click.echo(f"   轮次: {runs}")
    if limit:
        click.echo(f"   限额: {limit}")
    click.echo()

    # 加载查询
    queries = load_queries_from_benchmark(
        cases, domain=domain, difficulty=difficulty, limit=limit,
    )
    if not queries:
        click.echo("❌ 未找到测试查询")
        return

    click.echo(f"   加载 {len(queries)} 条测试查询")

    # 定义配置
    config_a = RunnerConfig(name=name_a, mode=mode_a, index_path=index_a, raw_dir=raw_a or "")
    config_b = RunnerConfig(name=name_b, mode=mode_b, index_path=index_b, raw_dir=raw_b or "")

    # 运行
    runner = ABTestRunner(seed=seed)
    click.echo("   🔄 运行 A/B 测试...")

    import time as _time
    t0 = _time.time()
    result = runner.run(config_a, config_b, queries, runs=runs)
    elapsed = _time.time() - t0

    click.echo(f"   ✅ 完成 ({elapsed:.1f}s)")
    click.echo()
    click.echo(result.summary())

    # 输出
    if output:
        result.save_json(output)
        click.echo(f"   📄 结果已写入: {output}")


# ── stats 命令组 ──────────────────────────────────


@cli.group()
def stats():
    """查询 API 用量统计和成本分析。"""
    pass


def _find_convert_db_files(source_dir=None):
    """Find all convert.db files under raw root, optionally filtered by source_dir."""
    raw_root = Path(os.getenv("RAW_ROOT", "raw"))
    if not raw_root.exists():
        return []
    db_files = list(raw_root.rglob("convert.db"))
    if source_dir:
        db_files = [f for f in db_files if source_dir in str(f)]
    return db_files


@stats.command()
@click.option("-d", "--source-dir", help="按源目录筛选")
@click.option("--days", type=int, help="统计最近 N 天的数据")
def summary(source_dir, days):
    """总体用量汇总。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        if not raw_root.exists():
            click.echo("未找到数据目录。请先运行 batch-convert。")
        else:
            click.echo("未找到统计数据。")
        return

    totals = {
        "ocr": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
        "llm_chat": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
        "rerank": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
    }

    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                data = db.get_token_usage_summary(
                    source_dir=source_dir, days=days
                )
                for ct, row in data.get("by_type", {}).items():
                    if ct in totals:
                        totals[ct]["calls"] += row.get("call_count", 0)
                        totals[ct]["input"] += row.get("input_tokens", 0)
                        totals[ct]["output"] += row.get("output_tokens", 0)
                        totals[ct]["total"] += row.get("total_tokens", 0)
                        totals[ct]["cost"] += row.get("cost_millicents", 0)
            finally:
                db.close()
        except Exception:
            continue

    # Build Rich table
    if _RICH_AVAILABLE:
        table = Table(title="📊 API 用量统计")
        table.add_column("指标", style="bold")
        table.add_column("OCR")
        table.add_column("LLM Chat")
        table.add_column("Rerank")
        table.add_column("合计", style="bold")

        grand_calls = sum(t["calls"] for t in totals.values())
        grand_input = sum(t["input"] for t in totals.values())
        grand_output = sum(t["output"] for t in totals.values())
        grand_total_tokens = sum(t["total"] for t in totals.values())
        grand_cost = sum(t["cost"] for t in totals.values())

        rows = [
            ("调用次数", "calls", None),
            ("Input Tokens", "input", ","),
            ("Output Tokens", "output", ","),
            ("总 Tokens", "total", ","),
            ("费用 (¥)", "cost", "cents"),
        ]
        for label, key, fmt in rows:
            ocr_val = totals["ocr"][key]
            llm_val = totals["llm_chat"][key]
            rerank_val = totals["rerank"][key]
            if fmt == "cents":
                total_val = ocr_val + llm_val + rerank_val
                vals = [f"{v / 100000:.4f}" for v in (ocr_val, llm_val, rerank_val, total_val)]
            elif fmt == ",":
                vals = [f"{v:,}" for v in (ocr_val, llm_val, rerank_val, ocr_val + llm_val + rerank_val)]
            else:
                total_val = ocr_val + llm_val + rerank_val
                vals = [str(v) for v in (ocr_val, llm_val, rerank_val, total_val)]
            table.add_row(label, *vals)

        console.print(table)
    else:
        click.echo("\n📊 API 用量统计")
        click.echo("=" * 60)
        grand_calls = sum(t["calls"] for t in totals.values())
        grand_input = sum(t["input"] for t in totals.values())
        grand_output = sum(t["output"] for t in totals.values())
        grand_total_tokens = sum(t["total"] for t in totals.values())
        grand_cost = sum(t["cost"] for t in totals.values())

        for ct, label in [("ocr", "OCR"), ("llm_chat", "LLM Chat"), ("rerank", "Rerank")]:
            t = totals[ct]
            if t["calls"] > 0:
                click.echo(f"\n  {label}:")
                click.echo(f"    调用次数: {t['calls']}")
                click.echo(f"    Input: {t['input']:,} | Output: {t['output']:,} | Total: {t['total']:,}")
                click.echo(f"    费用: ¥{t['cost'] / 100000:.4f}")

        click.echo("\n  合计:")
        click.echo(f"    调用次数: {grand_calls}")
        click.echo(f"    Input: {grand_input:,} | Output: {grand_output:,} | Total: {grand_total_tokens:,}")
        click.echo(f"    费用: ¥{grand_cost / 100000:.4f}")


@stats.command()
@click.option("--days", default=30, help="统计天数 (默认30)")
@click.option("-d", "--source-dir", help="按源目录筛选")
def daily(days, source_dir):
    """每日用量趋势。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        if not raw_root.exists():
            click.echo("未找到数据目录。请先运行 batch-convert。")
        else:
            click.echo("未找到统计数据。")
        return

    all_daily = {}
    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                rows = db.get_token_usage_daily(days=days, source_dir=source_dir)
                for row in rows:
                    date = row.get("date", "")
                    if date not in all_daily:
                        all_daily[date] = {
                            "call_count": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "cost_millicents": 0,
                        }
                    all_daily[date]["call_count"] += row.get("call_count", 0)
                    all_daily[date]["input_tokens"] += row.get("input_tokens", 0)
                    all_daily[date]["output_tokens"] += row.get("output_tokens", 0)
                    all_daily[date]["total_tokens"] += row.get("total_tokens", 0)
                    all_daily[date]["cost_millicents"] += row.get("cost_millicents", 0)
            finally:
                db.close()
        except Exception:
            continue

    if not all_daily:
        click.echo("指定范围内无统计数据。")
        return

    sorted_dates = sorted(all_daily.keys(), reverse=True)

    if _RICH_AVAILABLE:
        table = Table(title=f"📊 每日用量趋势 (最近 {days} 天)")
        table.add_column("日期", style="bold")
        table.add_column("调用次数")
        table.add_column("Input Tokens")
        table.add_column("Output Tokens")
        table.add_column("总 Tokens")
        table.add_column("费用 (¥)")

        for date in sorted_dates:
            d = all_daily[date]
            table.add_row(
                date,
                str(d["call_count"]),
                f"{d['input_tokens']:,}",
                f"{d['output_tokens']:,}",
                f"{d['total_tokens']:,}",
                f"{d['cost_millicents'] / 100000:.4f}",
            )
        console.print(table)
    else:
        click.echo(f"\n📊 每日用量趋势 (最近 {days} 天)")
        click.echo("=" * 70)
        click.echo(
            f"  {'日期':<12s} {'调用':>6s} {'Input':>12s} {'Output':>12s} {'Total':>12s} {'费用':>10s}"
        )
        click.echo(f"  {'-' * 64}")
        for date in sorted_dates:
            d = all_daily[date]
            click.echo(
                f"  {date:<12s} {d['call_count']:>6d} "
                f"{d['input_tokens']:>12,} {d['output_tokens']:>12,} "
                f"{d['total_tokens']:>12,} ¥{d['cost_millicents'] / 100000:>9.4f}"
            )


@stats.command()
@click.option("-d", "--source-dir", help="按源目录筛选")
@click.option("--days", type=int, help="统计最近 N 天的数据")
def models(source_dir, days):
    """按模型分组统计。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        if not raw_root.exists():
            click.echo("未找到数据目录。请先运行 batch-convert。")
        else:
            click.echo("未找到统计数据。")
        return

    all_models = {}
    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                rows = db.get_token_usage_by_model(source_dir=source_dir, days=days)
                for row in rows:
                    model_name = row.get("model", "unknown")
                    if model_name not in all_models:
                        all_models[model_name] = {
                            "call_count": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "cost_millicents": 0,
                        }
                    all_models[model_name]["call_count"] += row.get("call_count", 0)
                    all_models[model_name]["input_tokens"] += row.get("input_tokens", 0)
                    all_models[model_name]["output_tokens"] += row.get("output_tokens", 0)
                    all_models[model_name]["total_tokens"] += row.get("total_tokens", 0)
                    all_models[model_name]["cost_millicents"] += row.get("cost_millicents", 0)
            finally:
                db.close()
        except Exception:
            continue

    if not all_models:
        click.echo("暂无模型统计数据。")
        return

    sorted_models = sorted(all_models.items(), key=lambda x: x[1]["total_tokens"], reverse=True)

    if _RICH_AVAILABLE:
        table = Table(title="📊 模型用量统计")
        table.add_column("模型", style="bold")
        table.add_column("调用次数")
        table.add_column("Input Tokens")
        table.add_column("Output Tokens")
        table.add_column("总 Tokens")
        table.add_column("费用 (¥)")

        for model_name, d in sorted_models:
            table.add_row(
                model_name,
                str(d["call_count"]),
                f"{d['input_tokens']:,}",
                f"{d['output_tokens']:,}",
                f"{d['total_tokens']:,}",
                f"{d['cost_millicents'] / 100000:.4f}",
            )
        console.print(table)
    else:
        click.echo("\n📊 模型用量统计")
        click.echo("=" * 70)
        click.echo(
            f"  {'模型':<20s} {'调用':>6s} {'Input':>12s} {'Output':>12s} {'Total':>12s} {'费用':>10s}"
        )
        click.echo(f"  {'-' * 72}")
        for model_name, d in sorted_models:
            click.echo(
                f"  {model_name:<20s} {d['call_count']:>6d} "
                f"{d['input_tokens']:>12,} {d['output_tokens']:>12,} "
                f"{d['total_tokens']:>12,} ¥{d['cost_millicents'] / 100000:>9.4f}"
            )


@stats.command()
@click.option("-f", "--format", "fmt", type=click.Choice(["json", "csv", "markdown", "html"]), default="json")
@click.option("-o", "--output", help="输出文件路径")
@click.option("--days", type=int, help="统计最近 N 天的数据")
@click.option("-d", "--source-dir", help="按源目录筛选")
def export(fmt, output, days, source_dir):
    """导出统计报告。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        if not raw_root.exists():
            click.echo("未找到数据目录。请先运行 batch-convert。")
        else:
            click.echo("未找到统计数据。")
        return

    # Aggregate data from all DBs
    summary_data = {
        "ocr": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
        "llm_chat": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
        "rerank": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
    }
    daily_data = {}
    model_data = {}

    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                # Summary
                s = db.get_token_usage_summary(source_dir=source_dir, days=days)
                for ct, row in s.get("by_type", {}).items():
                    if ct in summary_data:
                        summary_data[ct]["calls"] += row.get("call_count", 0)
                        summary_data[ct]["input"] += row.get("input_tokens", 0)
                        summary_data[ct]["output"] += row.get("output_tokens", 0)
                        summary_data[ct]["total"] += row.get("total_tokens", 0)
                        summary_data[ct]["cost"] += row.get("cost_millicents", 0)

                # Daily
                for row in db.get_token_usage_daily(days=days or 30, source_dir=source_dir):
                    date = row.get("date", "")
                    if date not in daily_data:
                        daily_data[date] = {
                            "call_count": 0, "input_tokens": 0,
                            "output_tokens": 0, "total_tokens": 0, "cost_millicents": 0,
                        }
                    for k in ("call_count", "input_tokens", "output_tokens", "total_tokens", "cost_millicents"):
                        daily_data[date][k] += row.get(k, 0)

                # Models
                for row in db.get_token_usage_by_model(source_dir=source_dir, days=days):
                    m = row.get("model", "unknown")
                    if m not in model_data:
                        model_data[m] = {
                            "call_count": 0, "input_tokens": 0,
                            "output_tokens": 0, "total_tokens": 0, "cost_millicents": 0,
                        }
                    for k in ("call_count", "input_tokens", "output_tokens", "total_tokens", "cost_millicents"):
                        model_data[m][k] += row.get(k, 0)
            finally:
                db.close()
        except Exception:
            continue

    report = {
        "summary": summary_data,
        "daily": {k: daily_data[k] for k in sorted(daily_data.keys(), reverse=True)},
        "models": {k: model_data[k] for k in sorted(model_data.keys())},
    }

    if fmt == "json":
        content = json.dumps(report, ensure_ascii=False, indent=2)
    elif fmt == "csv":
        lines = ["section,key,calls,input_tokens,output_tokens,total_tokens,cost_millicents"]
        for ct in ("ocr", "llm_chat", "rerank"):
            d = summary_data[ct]
            lines.append(
                f"summary,{ct},{d['calls']},{d['input']},{d['output']},{d['total']},{d['cost']}"
            )
        for date in sorted(daily_data.keys(), reverse=True):
            d = daily_data[date]
            lines.append(
                f"daily,{date},{d['call_count']},{d['input_tokens']},{d['output_tokens']},{d['total_tokens']},{d['cost_millicents']}"
            )
        for m in sorted(model_data.keys()):
            d = model_data[m]
            lines.append(
                f"model,{m},{d['call_count']},{d['input_tokens']},{d['output_tokens']},{d['total_tokens']},{d['cost_millicents']}"
            )
        content = "\n".join(lines)
    elif fmt == "markdown":
        lines = ["# API 用量统计报告\n"]
        lines.append("## 总体汇总\n")
        lines.append("| 类型 | 调用次数 | Input | Output | Total | 费用(¥) |")
        lines.append("|------|---------|--------|---------|--------|---------|")
        for ct in ("ocr", "llm_chat", "rerank"):
            d = summary_data[ct]
            lines.append(
                f"| {ct} | {d['calls']} | {d['input']:,} | {d['output']:,} | {d['total']:,} | ¥{d['cost'] / 100000:.4f} |"
            )
        if daily_data:
            lines.append("\n## 每日趋势\n")
            lines.append("| 日期 | 调用 | Input | Output | Total | 费用(¥) |")
            lines.append("|------|------|--------|---------|--------|---------|")
            for date in sorted(daily_data.keys(), reverse=True):
                d = daily_data[date]
                lines.append(
                    f"| {date} | {d['call_count']} | {d['input_tokens']:,} | {d['output_tokens']:,} | {d['total_tokens']:,} | ¥{d['cost_millicents'] / 100000:.4f} |"
                )
        if model_data:
            lines.append("\n## 模型统计\n")
            lines.append("| 模型 | 调用 | Input | Output | Total | 费用(¥) |")
            lines.append("|------|------|--------|---------|--------|---------|")
            for m in sorted(model_data.keys()):
                d = model_data[m]
                lines.append(
                    f"| {m} | {d['call_count']} | {d['input_tokens']:,} | {d['output_tokens']:,} | {d['total_tokens']:,} | ¥{d['cost_millicents'] / 100000:.4f} |"
                )
        content = "\n".join(lines)
    elif fmt == "html":
        h = []
        h.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        h.append("<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse;width:100%}")
        h.append("th,td{border:1px solid #ddd;padding:8px;text-align:right}th{text-align:center;background:#f5f5f5}")
        h.append("h1,h2{color:#333}</style></head><body>")
        h.append("<h1>API 用量统计报告</h1>")
        h.append("<h2>总体汇总</h2><table><tr><th>类型</th><th>调用次数</th><th>Input</th><th>Output</th><th>Total</th><th>费用(¥)</th></tr>")
        for ct in ("ocr", "llm_chat", "rerank"):
            d = summary_data[ct]
            h.append(f"<tr><td>{ct}</td><td>{d['calls']}</td><td>{d['input']:,}</td><td>{d['output']:,}</td><td>{d['total']:,}</td><td>¥{d['cost'] / 100000:.4f}</td></tr>")
        h.append("</table>")
        if daily_data:
            h.append("<h2>每日趋势</h2><table><tr><th>日期</th><th>调用</th><th>Input</th><th>Output</th><th>Total</th><th>费用(¥)</th></tr>")
            for date in sorted(daily_data.keys(), reverse=True):
                d = daily_data[date]
                h.append(f"<tr><td>{date}</td><td>{d['call_count']}</td><td>{d['input_tokens']:,}</td><td>{d['output_tokens']:,}</td><td>{d['total_tokens']:,}</td><td>¥{d['cost_millicents'] / 100000:.4f}</td></tr>")
            h.append("</table>")
        if model_data:
            h.append("<h2>模型统计</h2><table><tr><th>模型</th><th>调用</th><th>Input</th><th>Output</th><th>Total</th><th>费用(¥)</th></tr>")
            for m in sorted(model_data.keys()):
                d = model_data[m]
                h.append(f"<tr><td>{m}</td><td>{d['call_count']}</td><td>{d['input_tokens']:,}</td><td>{d['output_tokens']:,}</td><td>{d['total_tokens']:,}</td><td>¥{d['cost_millicents'] / 100000:.4f}</td></tr>")
            h.append("</table>")
        h.append("</body></html>")
        content = "\n".join(h)
    else:
        content = json.dumps(report, ensure_ascii=False, indent=2)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        click.echo(f"📄 报告已导出: {out_path}")
    else:
        click.echo(content)


@cli.command("skills")
def list_skills():
    """列出可用的外部技能"""
    from src.agent.skill_loader import list_available_skills
    skills = list_available_skills()
    if not skills:
        click.echo("未找到外部技能。安装路径: ~/.agents/skills/ 或 ~/.opencode/skills/")
        return
    click.echo(f"找到 {len(skills)} 个技能:\n")
    for name, desc in sorted(skills.items()):
        click.echo(f"  {name}: {desc}")


@cli.command("analyze")
@click.argument("query_text")
@click.option("-i", "--index", required=True, help="索引路径")
@click.option("--raw-dir", default=None, help="Raw 目录路径 (默认: 索引父目录)")
@click.option("--mode", type=click.Choice(["compare", "extract", "summarize", "table"]),
              default="extract", help="分析模式: compare=对比, extract=提取, summarize=摘要, table=表格提取")
@click.option("--doc-ids", default=None, help="文档 ID 列表 (逗号分隔, compare 需 ≥2)")
@click.option("--doc-id", default=None, help="单个文档 ID (extract/summarize/table)")
@click.option("--aspect", default=None, help="对比焦点 (compare 模式)")
@click.option("--top-k", default=3, type=int, help="自动搜索时取 top-N 文档 (默认 3)")
@click.option("-f", "--output-format", type=click.Choice(["text", "json", "markdown"]), default="text",
              help="输出格式")
@click.option("--no-log", is_flag=True, help="不记录搜索日志")
def analyze(query_text, index, raw_dir, mode, doc_ids, doc_id, aspect, output_format, no_log):
    """文档深度分析 (对比/提取/摘要/表格)。

    \b
    分析模式:
      compare    对比多个文档的异同 (--doc-ids id1,id2)
      extract    从文档中提取结构化信息 (--doc-id id1)
      summarize  生成文档摘要 (--doc-id id1)
      table      提取文档中的表格数据 (--doc-id id1)

    \b
    示例:
      # 自动搜索 + 对比 (无需 doc_id，自动 BM25 搜索 top-3 对比)
      doc-search analyze "差旅标准对比" -i ./index --mode compare

      # 自动搜索 + 提取 (自动搜索 top-1 提取信息)
      doc-search analyze "提取报销流程" -i ./index --mode extract

      # 自动搜索 + 摘要
      doc-search analyze "年假制度摘要" -i ./index --mode summarize

      # 指定文档 ID (跳过搜索)
      doc-search analyze "差旅标准" -i ./index --mode compare --doc-ids abc123,def456 --raw-dir ./raw
      doc-search analyze "报销" -i ./index --mode extract --doc-id abc123
    """
    from src.agent.analysis_agent import create_analysis_agent, search_and_analyze
    from src.utils.config import Config

    # Honor --no-log
    if no_log:
        os.environ["NO_SEARCH_LOG"] = "1"

    config = Config.from_env()

    # Parse doc_ids
    ids_list = None
    if doc_ids:
        ids_list = [d.strip() for d in doc_ids.split(",") if d.strip()]

    # Auto-search mode: no explicit doc_id/doc_ids → search first
    if not doc_id and not ids_list:
        click.echo(f"🔍 自动搜索 \"{query_text}\" → {mode} 分析...")
        resp = search_and_analyze(
            query=query_text,
            index_path=index,
            config=config,
            mode=mode,
            raw_dir=raw_dir,
            top_k=top_k,
            aspect=aspect,
        )
    else:
        # Manual mode: use provided doc_id/doc_ids
        if not raw_dir:
            raw_dir = str(Path(index).parent)

        agent = create_analysis_agent(config=config, raw_dir=raw_dir)

        if mode == "compare":
            if not ids_list or len(ids_list) < 2:
                click.echo("❌ compare 模式需要 --doc-ids 指定至少 2 个文档 ID，或不指定 --doc-ids 使用自动搜索")
                sys.exit(1)
            click.echo(f"📊 对比分析 {len(ids_list)} 个文档...")
            resp = agent.compare(doc_ids=ids_list, aspect=aspect or query_text)
        elif mode == "summarize":
            if not doc_id:
                click.echo("❌ summarize 模式需要 --doc-id，或不指定 --doc-id 使用自动搜索")
                sys.exit(1)
            click.echo("📝 生成文档摘要...")
            resp = agent.summarize(doc_id=doc_id, focus=query_text)
        elif mode == "table":
            if not doc_id:
                click.echo("❌ table 模式需要 --doc-id，或不指定 --doc-id 使用自动搜索")
                sys.exit(1)
            click.echo("📋 提取表格数据...")
            resp = agent.analyze_table(doc_id=doc_id)
        else:  # extract
            if not doc_id and ids_list:
                doc_id = ids_list[0]
            if not doc_id:
                click.echo("❌ extract 模式需要 --doc-id，或不指定使用自动搜索")
                sys.exit(1)
            click.echo("🔍 提取结构化信息...")
            resp = agent.extract(doc_id=doc_id, query=query_text)

    # Output
    if not resp.success:
        click.echo(f"❌ 分析失败: {resp.error}")
        sys.exit(1)

    if output_format == "json":
        import json as _json
        click.echo(_json.dumps({
            "answer": resp.answer,
            "sources": resp.sources,
            "tokens_used": resp.tokens_used,
            "processing_time": resp.processing_time,
        }, ensure_ascii=False, indent=2))
    elif output_format == "markdown":
        click.echo(f"## 分析结果\n\n{resp.answer}\n")
        if resp.sources:
            click.echo(f"\n---\n**来源**: {', '.join(resp.sources)}")
        click.echo(f"\n⏱ 耗时: {resp.processing_time:.1f}s | Tokens: {resp.tokens_used}")
    else:
        click.echo(f"\n{resp.answer}\n")
        if resp.sources:
            click.echo(f"📄 来源: {', '.join(resp.sources)}")
        click.echo(f"⏱ 耗时: {resp.processing_time:.1f}s | Tokens: {resp.tokens_used}")

    # Log search
    _log_search_cli(
        query_text,
        {"answer": resp.answer, "sources": resp.sources,
         "tokens_used": resp.tokens_used, "processing_time": resp.processing_time},
        "analyze",
        index_path=str(index),
    )


# ── stats budget 子命令组 ─────────────────────────────


@stats.group()
def budget():
    """预算管理和检查。"""
    pass


@budget.command("list")
def budget_list():
    """查看所有预算配置。"""
    from src.stats.budget_guard import BudgetGuard

    db_files = _find_convert_db_files()
    if not db_files:
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        if not raw_root.exists():
            click.echo("未找到数据目录。请先运行 batch-convert。")
        else:
            click.echo("未找到预算数据。")
        return

    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                guard = BudgetGuard(db)
                budgets = guard.get_budgets()
                if not budgets:
                    click.echo("暂无预算配置。使用 'doc-search stats budget set' 添加。")
                    return

                if _RICH_AVAILABLE:
                    table = Table(title="📋 预算配置")
                    table.add_column("ID", style="bold")
                    table.add_column("名称")
                    table.add_column("限额(分)")
                    table.add_column("周期")
                    table.add_column("告警阈值")
                    table.add_column("超限阻断")
                    for b in budgets:
                        table.add_row(
                            str(b["id"]),
                            b["name"],
                            str(b["limit_cents"]),
                            b["period"],
                            f"{b['alert_threshold'] * 100:.0f}%",
                            "是" if b["block_exceed"] else "否",
                        )
                    console.print(table)
                else:
                    click.echo("\n📋 预算配置")
                    click.echo("=" * 60)
                    for b in budgets:
                        click.echo(
                            f"  ID={b['id']}  {b['name']}  "
                            f"限额={b['limit_cents']}分  周期={b['period']}  "
                            f"阈值={b['alert_threshold'] * 100:.0f}%  "
                            f"阻断={'是' if b['block_exceed'] else '否'}"
                        )
            finally:
                db.close()
        except Exception:
            continue


@budget.command("set")
@click.option("--name", required=True, help="预算名称")
@click.option("--limit", "limit_cents", required=True, type=int, help="预算限额（分）")
@click.option("--period", default="monthly", type=click.Choice(["daily", "monthly", "total"]), help="预算周期")
@click.option("--alert-threshold", default=0.8, type=float, help="告警阈值 (0-1)")
@click.option("--block/--no-block", default=False, help="超限时是否阻断")
def budget_set(name, limit_cents, period, alert_threshold, block):
    """设置或更新预算。"""
    from src.stats.budget_guard import BudgetGuard

    db_files = _find_convert_db_files()
    if not db_files:
        # Create a default DB if none found
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        raw_root.mkdir(parents=True, exist_ok=True)
        db_files = [raw_root / "convert.db"]

    db_path = db_files[0]
    db = ConvertDB(db_path)
    db.open()
    try:
        guard = BudgetGuard(db)
        budget_id = guard.set_budget(
            name=name,
            limit_cents=limit_cents,
            period=period,
            alert_threshold=alert_threshold,
            block_exceed=block,
        )
        click.echo(f"✅ 预算已设置: {name} (ID={budget_id}, 限额={limit_cents}分, 周期={period})")
    finally:
        db.close()


@budget.command("check")
@click.option("-d", "--source-dir", help="按源目录筛选")
def budget_check(source_dir):
    """检查预算使用情况。"""
    from src.stats.budget_guard import BudgetGuard

    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        if not raw_root.exists():
            click.echo("未找到数据目录。请先运行 batch-convert。")
        else:
            click.echo("未找到预算数据。")
        return

    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                guard = BudgetGuard(db)
                result = guard.check_budget(source_dir=source_dir)
                if not result.alerts:
                    click.echo("暂无预算配置。使用 'doc-search stats budget set' 添加。")
                    return

                if _RICH_AVAILABLE:
                    table = Table(title="💰 预算状态")
                    table.add_column("预算名称", style="bold")
                    table.add_column("周期")
                    table.add_column("已用(分)")
                    table.add_column("限额(分)")
                    table.add_column("使用率")
                    table.add_column("状态")
                    for a in result.alerts:
                        status = "✅ 正常"
                        if a.should_block:
                            status = "🚫 已阻断"
                        elif a.is_exceeded:
                            status = "⚠️ 超限"
                        elif a.usage_percent >= 80:
                            status = "⚠️ 告警"
                        table.add_row(
                            a.budget_name,
                            a.period,
                            str(a.current_spend_cents),
                            str(a.limit_cents),
                            f"{a.usage_percent:.1f}%",
                            status,
                        )
                    console.print(table)
                else:
                    click.echo("\n💰 预算状态")
                    click.echo("=" * 60)
                    overall = "✅ 所有预算正常" if result.is_within_budget else "⚠️ 有预算超限"
                    click.echo(f"  {overall}")
                    for a in result.alerts:
                        status = "正常"
                        if a.should_block:
                            status = "超限-已阻断"
                        elif a.is_exceeded:
                            status = "超限"
                        elif a.usage_percent >= 80:
                            status = "告警"
                        click.echo(
                            f"  {a.budget_name} ({a.period}): "
                            f"{a.current_spend_cents}/{a.limit_cents}分 "
                            f"({a.usage_percent:.1f}%) [{status}]"
                        )
            finally:
                db.close()
        except Exception:
            continue


@budget.command("remove")
@click.argument("budget_id", type=int)
def budget_remove(budget_id):
    """删除指定预算。"""
    from src.stats.budget_guard import BudgetGuard

    db_files = _find_convert_db_files()
    if not db_files:
        click.echo("未找到数据库。")
        return

    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                guard = BudgetGuard(db)
                if guard.remove_budget(budget_id):
                    click.echo(f"✅ 预算 ID={budget_id} 已删除。")
                    return
            finally:
                db.close()
        except Exception:
            continue
    click.echo(f"未找到预算 ID={budget_id}。")


# ── stats diagnostics 子命令 ────────────────────────────


@stats.command()
@click.option("--days", type=int, default=7, help="统计最近 N 天的数据")
@click.option("-d", "--source-dir", help="按源目录筛选")
def diagnostics(days, source_dir):
    """查询性能诊断摘要。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        click.echo("未找到诊断数据。请先执行 Agent 搜索查询。")
        return

    aggregated = {
        "total_queries": 0,
        "success_count": 0,
        "total_ms": 0,
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "total_cache_hits": 0,
        "by_complexity": {},
    }

    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                s = db.get_diagnostics_summary(days=days, source_dir=source_dir)
                aggregated["total_queries"] += s.get("total_queries", 0)
                aggregated["success_count"] += int(s.get("success_rate", 0) * s.get("total_queries", 0) / 100)
                aggregated["total_ms"] += s.get("avg_ms", 0) * s.get("total_queries", 0)
                aggregated["total_llm_calls"] += s.get("avg_llm_calls", 0) * s.get("total_queries", 0)
                aggregated["total_tool_calls"] += s.get("cache_hit_rate", 0)  # reuse field
                for c in s.get("by_complexity", []):
                    cx = c.get("complexity", "unknown")
                    if cx not in aggregated["by_complexity"]:
                        aggregated["by_complexity"][cx] = {"count": 0, "total_ms": 0}
                    aggregated["by_complexity"][cx]["count"] += c.get("count", 0)
                    aggregated["by_complexity"][cx]["total_ms"] += c.get("avg_ms", 0) * c.get("count", 0)
            finally:
                db.close()
        except Exception:
            continue

    total = aggregated["total_queries"]
    if total == 0:
        click.echo("暂无查询诊断数据。")
        return

    click.echo("📊 查询性能诊断摘要")
    click.echo(f"   时间范围: 最近 {days} 天")
    click.echo("")
    click.echo(f"  总查询数:    {total}")
    click.echo(f"  成功率:      {aggregated['success_count'] / total * 100:.1f}%")
    avg = aggregated["total_ms"] / total if total else 0
    click.echo(f"  平均延迟:    {avg:.0f} ms ({avg / 1000:.1f}s)")
    avg_llm = aggregated["total_llm_calls"] / total if total else 0
    click.echo(f"  平均 LLM 调用: {avg_llm:.1f} 次")

    if aggregated["by_complexity"]:
        click.echo("")
        click.echo("  按复杂度:")
        for cx, data in sorted(aggregated["by_complexity"].items()):
            n = data["count"]
            click.echo(f"    {cx:<10s}  {n:>5d} 查询  avg {data['total_ms'] / n:.0f}ms" if n else f"    {cx:<10s}  0 查询")


@stats.command("slow-queries")
@click.option("--threshold", type=int, default=30000, help="慢查询阈值 (ms)")
@click.option("--limit", type=int, default=20, help="最多显示条数")
@click.option("-d", "--source-dir", help="按源目录筛选")
def slow_queries(threshold, limit, source_dir):
    """列出慢查询。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        click.echo("未找到诊断数据。")
        return

    all_slow = []
    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                rows = db.get_slow_queries(threshold_ms=threshold, limit=limit, source_dir=source_dir)
                all_slow.extend(rows)
            finally:
                db.close()
        except Exception:
            continue

    all_slow.sort(key=lambda r: r.get("total_ms", 0), reverse=True)
    all_slow = all_slow[:limit]

    if not all_slow:
        click.echo(f"无慢查询 (> {threshold}ms)。")
        return

    click.echo(f"🐢 慢查询 (> {threshold}ms, 共 {len(all_slow)} 条)")
    click.echo("")
    for i, r in enumerate(all_slow, 1):
        preview = r.get("query_preview", "")
        click.echo(
            f"  {i:>3d}. [{r.get('complexity', '?'):<7s}] "
            f"{r.get('total_ms', 0) / 1000:.1f}s  "
            f"LLM×{r.get('llm_call_count', 0)}  "
            f"{'✅' if r.get('success') else '❌'}  "
            f"{preview}"
        )


@stats.command("step-breakdown")
@click.option("--days", type=int, default=7, help="统计最近 N 天的数据")
@click.option("-d", "--source-dir", help="按源目录筛选")
def step_breakdown(days, source_dir):
    """分步延迟分析。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        click.echo("未找到诊断数据。")
        return

    merged = {}
    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                bd = db.get_step_breakdown(days=days, source_dir=source_dir)
                for step, data in bd.items():
                    if step not in merged:
                        merged[step] = {"count": 0, "total_ms": 0, "max_ms": 0}
                    merged[step]["count"] += data.get("count", 0)
                    merged[step]["total_ms"] += data.get("avg_ms", 0) * data.get("count", 0)
                    merged[step]["max_ms"] = max(merged[step]["max_ms"], data.get("max_ms", 0))
            finally:
                db.close()
        except Exception:
            continue

    if not merged:
        click.echo("暂无分步计时数据。")
        return

    click.echo("📋 分步延迟分析")
    click.echo("")
    click.echo(f"  {'步骤':<25s} {'次数':>6s} {'平均(ms)':>10s} {'最大(ms)':>10s}")
    click.echo("  " + "─" * 55)
    for step in sorted(merged, key=lambda s: merged[s]["total_ms"], reverse=True):
        d = merged[step]
        n = d["count"]
        avg = d["total_ms"] / n if n else 0
        click.echo(f"  {step:<25s} {n:>6d} {avg:>10.1f} {d['max_ms']:>10.1f}")


@stats.command("llm-calls")
@click.option("--days", type=int, default=7, help="统计最近 N 天的数据")
@click.option("-d", "--source-dir", help="按源目录筛选")
def llm_calls(days, source_dir):
    """LLM 调用统计。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        click.echo("未找到诊断数据。")
        return

    all_stats = []
    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                rows = db.get_llm_call_stats(days=days, source_dir=source_dir)
                all_stats.extend(rows)
            finally:
                db.close()
        except Exception:
            continue

    if not all_stats:
        click.echo("暂无 LLM 调用统计数据。")
        return

    # Merge by call_type
    merged = {}
    for r in all_stats:
        ct = r.get("call_type", "unknown")
        if ct not in merged:
            merged[ct] = {"call_count": 0, "total_latency": 0, "total_input": 0,
                          "total_output": 0, "total_retries": 0}
        merged[ct]["call_count"] += r.get("call_count", 0)
        merged[ct]["total_latency"] += r.get("avg_latency_ms", 0) * r.get("call_count", 0)
        merged[ct]["total_input"] += r.get("total_input_tokens", 0)
        merged[ct]["total_output"] += r.get("total_output_tokens", 0)
        merged[ct]["total_retries"] += r.get("total_retries", 0)

    click.echo("🤖 LLM 调用统计")
    click.echo("")
    click.echo(f"  {'类型':<15s} {'次数':>6s} {'平均延迟':>10s} {'Input':>12s} {'Output':>12s} {'重试':>6s}")
    click.echo("  " + "─" * 65)
    for ct in sorted(merged, key=lambda c: merged[c]["call_count"], reverse=True):
        d = merged[ct]
        n = d["call_count"]
        avg_lat = d["total_latency"] / n if n else 0
        click.echo(
            f"  {ct:<15s} {n:>6d} {avg_lat:>9.0f}ms "
            f"{d['total_input']:>12,} {d['total_output']:>12,} {d['total_retries']:>6d}"
        )


# ── stats feedback 子命令 ────────────────────────────


@stats.command()
@click.option("--days", type=int, default=7, help="统计最近 N 天的数据")
@click.option("-d", "--source-dir", help="按源目录筛选")
def feedback(days, source_dir):
    """搜索结果反馈统计。"""
    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        click.echo("未找到反馈数据。")
        return

    agg = {
        "total_up": 0,
        "total_down": 0,
        "worst_docs": {},
    }

    for db_path in db_files:
        try:
            db = ConvertDB(db_path)
            db.open()
            try:
                s = db.get_feedback_summary(days=days)
                agg["total_up"] += s.get("total_up", 0)
                agg["total_down"] += s.get("total_down", 0)
                for w in s.get("worst_rated_docs", []):
                    title = w.get("doc_title") or "(unknown)"
                    if title not in agg["worst_docs"]:
                        agg["worst_docs"][title] = {"up": 0, "down": 0}
                    agg["worst_docs"][title]["down"] += w.get("down_count", 0)
                    agg["worst_docs"][title]["up"] += w.get("up_count", 0)
            finally:
                db.close()
        except Exception:
            continue

    total = agg["total_up"] + agg["total_down"]
    if total == 0:
        click.echo("暂无反馈数据。")
        return

    rate = agg["total_up"] / total * 100 if total else 0
    click.echo(f"👍👎 搜索结果反馈统计 (最近 {days} 天)")
    click.echo("")
    click.echo(f"  总反馈:   {total}")
    click.echo(f"  👍 好评:  {agg['total_up']}")
    click.echo(f"  👎 差评:  {agg['total_down']}")
    click.echo(f"  好评率:   {rate:.1f}%")

    worst = sorted(agg["worst_docs"].items(), key=lambda x: x[1]["down"], reverse=True)
    worst = [w for w in worst if w[1]["down"] > 0][:10]
    if worst:
        click.echo("")
        click.echo("  差评最多的文档:")
        click.echo(f"    {'文档标题':<40s} {'👎':>4s} {'👍':>4s}")
        click.echo("    " + "─" * 52)
        for title, counts in worst:
            display = title[:38] + ".." if len(title) > 40 else title
            click.echo(f"    {display:<40s} {counts['down']:>4d} {counts['up']:>4d}")


# ── stats realtime 子命令 ────────────────────────────


@stats.command()
@click.option("--interval", default=5, help="刷新间隔（秒）")
@click.option("-d", "--source-dir", help="按源目录筛选")
def realtime(interval, source_dir):
    """实时监控 API 用量。"""
    import time

    db_files = _find_convert_db_files(source_dir)
    if not db_files:
        raw_root = Path(os.getenv("RAW_ROOT", "raw"))
        if not raw_root.exists():
            click.echo("未找到数据目录。请先运行 batch-convert。")
        else:
            click.echo("未找到统计数据。")
        return

    click.echo(f"🔄 实时监控 (每 {interval}s 刷新, Ctrl+C 退出)\n")

    try:
        while True:
            # Clear terminal
            os.system('cls' if os.name == 'nt' else 'clear')

            click.echo(f"🔄 实时监控 (每 {interval}s 刷新, Ctrl+C 退出)")
            click.echo(f"   上次更新: {datetime.now().strftime('%H:%M:%S')}")
            click.echo("")

            totals = {
                "ocr": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
                "llm_chat": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
                "rerank": {"calls": 0, "input": 0, "output": 0, "total": 0, "cost": 0},
            }

            for db_path in db_files:
                try:
                    db = ConvertDB(db_path)
                    db.open()
                    try:
                        data = db.get_token_usage_summary(source_dir=source_dir)
                        for ct, row in data.get("by_type", {}).items():
                            if ct in totals:
                                totals[ct]["calls"] += row.get("call_count", 0)
                                totals[ct]["input"] += row.get("input_tokens", 0)
                                totals[ct]["output"] += row.get("output_tokens", 0)
                                totals[ct]["total"] += row.get("total_tokens", 0)
                                totals[ct]["cost"] += row.get("cost_millicents", 0)
                    finally:
                        db.close()
                except Exception:
                    continue

            grand_calls = sum(t["calls"] for t in totals.values())
            grand_input = sum(t["input"] for t in totals.values())
            grand_output = sum(t["output"] for t in totals.values())
            grand_total_tokens = sum(t["total"] for t in totals.values())
            grand_cost = sum(t["cost"] for t in totals.values())

            click.echo("─" * 60)
            click.echo(
                f"  {'类型':<12s} {'调用':>8s} {'Input':>12s} "
                f"{'Output':>12s} {'Total':>12s} {'费用':>10s}"
            )
            click.echo("─" * 60)
            for ct, label in [("ocr", "OCR"), ("llm_chat", "LLM Chat"), ("rerank", "Rerank")]:
                t = totals[ct]
                if t["calls"] > 0:
                    click.echo(
                        f"  {label:<12s} {t['calls']:>8d} {t['input']:>12,} "
                        f"{t['output']:>12,} {t['total']:>12,} ¥{t['cost'] / 100000:>9.4f}"
                    )
            click.echo("─" * 60)
            click.echo(
                f"  {'合计':<12s} {grand_calls:>8d} {grand_input:>12,} "
                f"{grand_output:>12,} {grand_total_tokens:>12,} ¥{grand_cost / 100000:>9.4f}"
            )
            click.echo("─" * 60)

            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\n\n监控已停止。")


@cli.command("pdf-enhance")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default="./output", show_default=True,
              help="输出根目录")
@click.option("--dpi", type=int, default=150, show_default=True, help="PDF 渲染 DPI")
@click.option("--glm-key", envvar="GLM_API_KEY", default=None, help="GLM-OCR API Key")
@click.option("--glm-model", default="glm-ocr", show_default=True, help="GLM-OCR 模型名")
@click.option("--la-model", default="nvidia/LocateAnything-3B", show_default=True,
              help="LocateAnything 模型路径")
@click.option("--la-device", default="cuda", show_default=True, help="LA 部署设备 (cuda/cpu)")
@click.option("--la-categories", default=None,
              help="LA 补充检测类别，逗号分隔 (如: signature,checkbox,watermark)")
@click.option("--resume", is_flag=True, help="断点续传模式")
@click.option("--skip-la", is_flag=True, help="跳过 LA 增强 (仅 GLM-OCR)")
@click.option("--skip-comparison", is_flag=True, help="跳过对比报告生成")
@click.option("--parallel", type=int, default=3, show_default=True, help="GLM-OCR 并发数")
def pdf_enhance(pdf_path, output, dpi, glm_key, glm_model, la_model, la_device,
                la_categories, resume, skip_la, skip_comparison, parallel):
    """专用高精度单 PDF 处理: GLM-OCR + LocateAnything-3B 级联增强

    将单份 PDF 扫描文件经 GLM-OCR 第一轮识别 + LocateAnything-3B 第二轮补充，
    输出包含原始图片、两阶段识别结果和量化对比报告。

    \b
    输出目录结构:
      output/
      ├── images/              PDF 页面图片 (持久保留, 可复用)
      ├── glm_ocr/             GLM-OCR 第一轮结果
      ├── enhanced/            LA 增强后结果
      └── comparison_report.md  对比分析报告

    \b
    示例:
      doc-search pdf-enhance "合同.pdf" -o ./output
      doc-search pdf-enhance "合同.pdf" --resume          # 断点续传
      doc-search pdf-enhance "合同.pdf" --skip-la          # 仅 GLM-OCR
      doc-search pdf-enhance "合同.pdf" --dpi 200          # 更高分辨率
      doc-search pdf-enhance "合同.pdf" --la-device cpu    # CPU 模式
    """
    import time as _time

    from src.processor.pdf_enhance import PDFEnhancePipeline

    pdf_path = Path(pdf_path).resolve()
    output_root = Path(output).resolve()
    # Pipeline.run() creates per-PDF subdirectory: output/<pdf_stem>/
    # CLI just passes output_root, no double-nesting
    actual_output = output_root / pdf_path.stem

    click.echo(f"📄 PDF 文件: {pdf_path}")
    click.echo(f"📁 输出目录: {actual_output}")
    click.echo(f"🔧 DPI: {dpi} | 并发: {parallel} | LA: {'跳过' if skip_la else '启用'}")

    if not glm_key:
        # Try loading from .env
        glm_key = os.environ.get("GLM_API_KEY", "")
    if not glm_key and not skip_la:
        click.echo("⚠️  未配置 GLM_API_KEY，GLM-OCR 将无法调用")
        click.echo("   设置方法: --glm-key <KEY> 或 .env 文件中 GLM_API_KEY=...")
        return

    # Parse LA categories
    la_cats = None
    if la_categories:
        la_cats = [c.strip() for c in la_categories.split(",") if c.strip()]

    # Confirm overwrite (only check per-PDF subdirectory)
    if actual_output.exists() and any(actual_output.iterdir()) and not resume:
        if not click.confirm(f"输出目录 {actual_output} 非空，是否继续？"):
            return

    # Initialize pipeline
    pipeline = PDFEnhancePipeline(
        glm_api_key=glm_key,
        glm_model=glm_model,
        la_model_path=la_model,
        la_device=la_device,
        la_categories=la_cats,
        dpi=dpi,
        glm_parallel=parallel,
    )

    # Execute — pipeline.run() adds pdf_stem subdirectory internally
    start_time = _time.time()
    try:
        result = pipeline.run(
            pdf_path=pdf_path,
            output_dir=output_root,
            resume=resume,
            skip_la=skip_la,
            skip_comparison=skip_comparison,
        )
    except KeyboardInterrupt:
        click.echo("\n⚠️  用户中断 — 已保存部分结果，可用 --resume 继续")
        return
    except Exception as e:
        click.echo(f"❌ 处理失败: {e}")
        logger.exception("pdf-enhance failed")
        return

    elapsed = _time.time() - start_time

    # Display results
    click.echo("")
    click.echo("✅ " + "=" * 56)
    pdf_meta = result.get("pdf_metadata", {})
    click.echo(f"  📄 文档: {pdf_path.name} ({pdf_meta.get('total_pages', '?')} 页)")

    glm_stats = result.get("glm_stats", {})
    click.echo(f"  📊 GLM-OCR: {glm_stats.get('total_regions', 0)} 区域, "
               f"{glm_stats.get('success_pages', 0)}/{pdf_meta.get('total_pages', 0)} 页成功")

    if not skip_la:
        la_stats = result.get("la_stats", {})
        click.echo(f"  🔍 LA 补充: {la_stats.get('total_supplements', 0)} 新区域, "
                   f"{la_stats.get('special_elements', 0)} 特殊元素")

    comp = result.get("comparison", {})
    if comp and "coverage_improvement_pct" in comp:
        click.echo(f"  📈 区域增长率: {comp['coverage_improvement_pct']:.1f}%")
    if comp and "report_path" in comp:
        click.echo(f"  📄 对比报告: {comp['report_path']}")
    elif skip_la:
        click.echo(f"  📄 GLM-OCR 全文: {actual_output / 'glm_ocr' / 'full_document.md'}")

    timings = result.get("phase_timings", {})
    if timings:
        timing_str = " | ".join(f"{k}: {v:.1f}s" for k, v in timings.items())
        click.echo(f"  ⏱️  耗时: {timing_str} | 总计 {elapsed:.1f}s")

    click.echo("=" * 58)


if __name__ == "__main__":
    cli()
