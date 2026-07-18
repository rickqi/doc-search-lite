"""
Filesystem monitor that watches raw directories for .md file changes
and automatically updates the Tantivy search index.

Supports:
- File creation: adds new .md files to the index
- File modification: updates existing .md files in the index
- File deletion: removes deleted .md files from the index
- Debouncing: coalesces rapid successive events (e.g. editor saves)
- Logging: console output + optional log file for Task Scheduler
"""

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.storage.index import TantivyIndexManager

logger = logging.getLogger(__name__)

# ── Content sampling (mirrors cli.py:_sample_content) ──

MAX_CONTENT_CHARS = 50_000
SAMPLE_THRESHOLD_BYTES = 5 * 1024 * 1024  # 5 MB

# Maximum number of chunk IDs to try deleting (safety cap for _remove_from_index)
_MAX_CHUNK_IDS = 100


def _sample_content(content: str, file_size: int) -> tuple[str, str]:
    """Sample large files to keep indexing fast. Returns (content, strategy)."""
    if len(content) <= MAX_CONTENT_CHARS:
        return content, "full"

    if file_size <= SAMPLE_THRESHOLD_BYTES:
        # Small-ish file: just truncate
        return content[:MAX_CONTENT_CHARS], "truncated"

    # Large file: head + mid + tail sampling
    head_chars = MAX_CONTENT_CHARS // 2
    tail_chars = MAX_CONTENT_CHARS // 4
    mid_start = max(0, len(content) // 2 - MAX_CONTENT_CHARS // 8)
    mid_chars = MAX_CONTENT_CHARS // 4

    sampled = (
        content[:head_chars]
        + content[mid_start : mid_start + mid_chars]
        + content[-tail_chars:]
    )
    return sampled, "sampled"


def _compute_doc_id(rel_path: str) -> str:
    """Compute a stable doc_id from a relative path (matching build-index)."""
    return hashlib.sha256(rel_path.replace("\\", "/").encode()).hexdigest()[:16]


@dataclass
class IndexWatchStats:
    """Accumulated stats for the running watcher."""
    added: int = 0
    updated: int = 0
    deleted: int = 0
    errors: int = 0
    indexed_files: set[str] = field(default_factory=set)

    def summary(self) -> str:
        return (
            f"added={self.added} updated={self.updated} "
            f"deleted={self.deleted} errors={self.errors}"
        )


class _MdFileHandler(FileSystemEventHandler):
    """Watchdog event handler for .md file changes."""

    def __init__(
        self,
        raw_path: Path,
        index_mgr: TantivyIndexManager,
        debounce_seconds: float = 1.0,
        on_change: Callable[[str, str, IndexWatchStats], None] | None = None,
        chunk_mode: bool = False,
        chunk_min_size: int = 50_000,
    ):
        super().__init__()
        self._raw_path = raw_path
        self._index_mgr = index_mgr
        self._debounce = debounce_seconds
        self._on_change = on_change
        self._stats = IndexWatchStats()
        self._pending: dict[str, float] = {}  # rel_path → last event time
        self._lock = Lock()
        self._stop_event = Event()
        self._chunk_mode = chunk_mode
        self._chunk_min_size = chunk_min_size

    @property
    def stats(self) -> IndexWatchStats:
        return self._stats

    def stop(self) -> None:
        self._stop_event.set()

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        # Treat moves as delete old + create new
        self._handle(event.src_path, "deleted")
        self._handle(event.dest_path, "created")

    def _handle(self, file_path: str, action: str) -> None:
        """Debounced handler: coalesce rapid events for the same file."""
        if self._stop_event.is_set():
            return

        abs_path = Path(file_path)
        if abs_path.suffix.lower() != ".md":
            return
        if abs_path.name.startswith("_"):
            return  # Skip index/_ files

        try:
            rel_path = str(abs_path.relative_to(self._raw_path))
        except ValueError:
            return  # Not under our watch root

        normalized = rel_path.replace("\\", "/")

        with self._lock:
            now = time.time()
            last = self._pending.get(normalized, 0)
            if now - last < self._debounce:
                self._pending[normalized] = now
                return
            self._pending[normalized] = now

        # Process after debounce window
        if action == "deleted":
            self._remove_from_index(normalized)
        else:
            # Delay slightly to let file system settle
            time.sleep(0.1)
            self._add_to_index(normalized, abs_path)

        logger.debug("Watchdog: %s %s", action, normalized)

    def _add_to_index(self, rel_path: str, abs_path: Path) -> None:
        """Read .md file and add/update in the index."""
        doc_id = _compute_doc_id(rel_path)
        try:
            if not abs_path.exists():
                return

            file_size = abs_path.stat().st_size
            content = abs_path.read_text(encoding="utf-8")
            if not content.strip():
                logger.warning("Skipping empty file: %s", rel_path)
                return

            # Strip YAML frontmatter before indexing (prevents BM25 pollution)
            from src.converter.frontmatter import strip_frontmatter
            _, content = strip_frontmatter(content)

            content, strategy = _sample_content(content, file_size)
            if strategy in ("truncated", "sampled"):
                logger.info("IndexWatcher: %s %s (%d KB)", strategy.upper(), rel_path, file_size // 1024)

            title = abs_path.stem
            metadata = {
                "filename": abs_path.name,
                "source_path": rel_path,
            }

            # Try to load JSON metadata for keywords
            meta_json_path = abs_path.with_suffix(abs_path.suffix + ".json")
            if meta_json_path.exists():
                try:
                    import json
                    with open(meta_json_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    if isinstance(meta, dict):
                        if "tags" in meta:
                            metadata["keywords"] = meta["tags"]
                        if "title" in meta:
                            title = meta["title"]
                except Exception:
                    pass

            # Delete any existing chunk entries for this file (chunk-mode safe)
            self._delete_all_chunks(doc_id)

            # ── Chunk mode: split long documents ──────────────
            if self._chunk_mode and len(content) > self._chunk_min_size:
                from src.converter.headings import extract_headings
                from src.storage.chunker import split_into_chunks

                headings = extract_headings(content)
                chunks = split_into_chunks(content, headings, self._chunk_min_size)
                is_new = doc_id not in self._stats.indexed_files
                for idx, (chunk_title, chunk_text) in enumerate(chunks):
                    chunk_meta = dict(metadata)
                    chunk_meta["chunk_index"] = idx
                    self._index_mgr.add_document(
                        f"{doc_id}#c{idx}",
                        f"{title} § {chunk_title}",
                        chunk_text,
                        chunk_meta,
                    )
                if is_new:
                    self._stats.added += 1
                else:
                    self._stats.updated += 1
                self._stats.indexed_files.add(doc_id)
                logger.info(
                    "IndexWatcher: %s %s (%s, %d chunks)",
                    "ADDED" if is_new else "UPDATED", rel_path, strategy, len(chunks),
                )
            else:
                is_new = doc_id not in self._stats.indexed_files

                if is_new:
                    self._index_mgr.add_document(doc_id, title, content, metadata)
                    self._stats.added += 1
                    self._stats.indexed_files.add(doc_id)
                else:
                    self._index_mgr.update_document(doc_id, title, content, metadata)
                    self._stats.updated += 1

                logger.info("IndexWatcher: %s %s (%s)", "ADDED" if is_new else "UPDATED", rel_path, strategy)

            self._index_mgr.commit()

        except Exception as e:
            self._stats.errors += 1
            logger.error("IndexWatcher error on %s: %s", rel_path, e)

        if self._on_change:
            self._on_change(rel_path, "add" if is_new else "update", self._stats)

    def _delete_all_chunks(self, base_doc_id: str) -> None:
        """Delete the base doc_id and all chunk variants (``#c0`` … ``#cN``).

        Tantivy ``delete_documents`` is idempotent for non-existent IDs, so
        over-deleting is safe.
        """
        self._index_mgr.delete_document(base_doc_id)
        for i in range(_MAX_CHUNK_IDS):
            self._index_mgr.delete_document(f"{base_doc_id}#c{i}")

    def _remove_from_index(self, rel_path: str) -> None:
        """Remove a deleted file from the index (including all chunks)."""
        doc_id = _compute_doc_id(rel_path)
        try:
            self._delete_all_chunks(doc_id)
            self._index_mgr.commit()
            self._stats.deleted += 1
            self._stats.indexed_files.discard(doc_id)
            logger.info("IndexWatcher: DELETED %s", rel_path)
        except Exception as e:
            self._stats.errors += 1
            logger.error("IndexWatcher delete error on %s: %s", rel_path, e)

        if self._on_change:
            self._on_change(rel_path, "delete", self._stats)


def _preload_indexed_files(index_mgr: TantivyIndexManager) -> set[str]:
    """Pre-populate the set of already-indexed doc_ids from the existing index."""
    indexed: set[str] = set()
    try:
        stats = index_mgr.get_stats()
        if stats.get("num_docs", 0) > 0:
            # We can't easily enumerate all doc_ids from Tantivy,
            # but we can mark the index as "pre-populated" so we
            # know which files triggered an update vs. initial add.
            # For now, we rely on the in-memory set to track adds.
            pass
    except Exception:
        pass
    return indexed


class IndexWatcher:
    """High-level watcher that monitors a raw directory for index updates.

    Usage:
        watcher = IndexWatcher(raw_dir="/path/to/raw/mydocs")
        watcher.start()   # blocks until stop() is called
    """

    def __init__(
        self,
        raw_dir: str,
        debounce_seconds: float = 1.0,
        use_jieba: bool = True,
        max_content_chars: int = MAX_CONTENT_CHARS,
        sample_threshold: int = SAMPLE_THRESHOLD_BYTES,
        chunk_mode: bool = False,
        chunk_min_size: int = 50_000,
    ):
        self._raw_path = Path(raw_dir).resolve()
        if not self._raw_path.is_dir():
            raise ValueError(f"Not a directory: {raw_dir}")

        self._index_path = self._raw_path / "index"
        self._debounce = debounce_seconds
        self._use_jieba = use_jieba
        self._chunk_mode = chunk_mode
        self._chunk_min_size = chunk_min_size

        # Allow custom content sampling params
        global MAX_CONTENT_CHARS, SAMPLE_THRESHOLD_BYTES
        MAX_CONTENT_CHARS = max_content_chars
        SAMPLE_THRESHOLD_BYTES = sample_threshold

        self._observer: Observer | None = None
        self._handler: _MdFileHandler | None = None
        self._index_mgr: TantivyIndexManager | None = None
        self._thread: Thread | None = None

    @property
    def stats(self) -> IndexWatchStats | None:
        if self._handler:
            return self._handler.stats
        return None

    def start(self, blocking: bool = True) -> None:
        """Start watching the raw directory.

        Args:
            blocking: If True, blocks the calling thread until stop() is called.
                      If False, runs in a daemon background thread.
        """
        # Create index directory if needed
        self._index_path.mkdir(parents=True, exist_ok=True)

        # Open index in read-write mode
        self._index_mgr = TantivyIndexManager(
            index_path=self._index_path,
            use_jieba=self._use_jieba,
            readonly=False,
        )

        # Create handler and observer
        self._handler = _MdFileHandler(
            raw_path=self._raw_path,
            index_mgr=self._index_mgr,
            debounce_seconds=self._debounce,
            chunk_mode=self._chunk_mode,
            chunk_min_size=self._chunk_min_size,
        )

        self._observer = Observer()
        self._observer.schedule(self._handler, str(self._raw_path), recursive=True)
        self._observer.start()

        logger.info(
            "IndexWatcher started: %s → %s (debounce=%.1fs)",
            self._raw_path, self._index_path, self._debounce,
        )

        if blocking:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = Thread(target=self._observer.join, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop watching and clean up resources."""
        if self._handler:
            self._handler.stop()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._index_mgr:
            self._index_mgr.close()
        logger.info("IndexWatcher stopped. Final stats: %s", self.stats.summary() if self.stats else "N/A")


def start_watching(
    raw_dir: str,
    debounce_seconds: float = 1.0,
    use_jieba: bool = True,
    blocking: bool = True,
) -> IndexWatcher:
    """Convenience function to create and start an IndexWatcher.

    Args:
        raw_dir: Path to the raw directory containing .md files and index/
        debounce_seconds: Debounce window for rapid file events
        use_jieba: Enable Chinese tokenization
        blocking: Whether to block the calling thread

    Returns:
        The IndexWatcher instance (use .stop() to shut down)
    """
    watcher = IndexWatcher(
        raw_dir=raw_dir,
        debounce_seconds=debounce_seconds,
        use_jieba=use_jieba,
    )
    watcher.start(blocking=blocking)
    return watcher
