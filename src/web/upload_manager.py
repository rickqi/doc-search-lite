"""Upload job manager for file upload → convert → index pipeline.

Manages the lifecycle of upload jobs: queued → converting → indexing → done/failed.
Progress is streamed via asyncio.Queue per job (SSE-compatible).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.converter.coordinator import ConverterCoordinator
from src.storage.index import TantivyIndexManager
from src.storage.raw_store import RawStore


@dataclass
class UploadJob:
    """State for a single upload job."""

    job_id: str
    raw_dir: Path
    index_dir: Path
    files: list[Path] = field(default_factory=list)
    status: str = "queued"  # queued | converting | indexing | done | failed
    progress: dict[str, Any] = field(default_factory=lambda: {
        "stage": "queued",
        "current": 0,
        "total": 0,
        "current_file": "",
        "success_count": 0,
        "failed_count": 0,
        "doc_count": 0,
        "error": None,
    })
    event_queue: asyncio.Queue | None = None
    abort_event: asyncio.Event | None = None
    created: float = field(default_factory=time.time)


class UploadManager:
    """Thread-safe registry for upload jobs.

    Pattern matches SessionManager: dict + Lock + cleanup loop.
    """

    IDLE_TIMEOUT = 3600  # 1 hour
    MAX_JOBS = 20

    def __init__(self, storage_dir: Path):
        self._jobs: dict[str, UploadJob] = {}
        self._lock = threading.Lock()
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        raw_dir: Path,
        index_dir: Path,
        files: list[Path],
    ) -> UploadJob:
        job_id = f"up_{uuid.uuid4().hex[:12]}"
        job = UploadJob(
            job_id=job_id,
            raw_dir=raw_dir,
            index_dir=index_dir,
            files=files,
        )
        with self._lock:
            # Evict oldest if at capacity
            if len(self._jobs) >= self.MAX_JOBS:
                oldest = min(self._jobs.values(), key=lambda j: j.created)
                self._jobs.pop(oldest.job_id, None)
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> UploadJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs.pop(job_id)
                if job.abort_event:
                    job.abort_event.set()
                return True
        return False

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "file_count": len(j.files),
                    "progress": j.progress,
                    "created": j.created,
                }
                for j in sorted(self._jobs.values(), key=lambda x: x.created, reverse=True)
            ]

    def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            expired = [
                jid for jid, j in self._jobs.items()
                if now - j.created > self.IDLE_TIMEOUT
            ]
            for jid in expired:
                job = self._jobs.pop(jid)
                if job.abort_event:
                    job.abort_event.set()
                removed += 1
        return removed


# Singleton
_upload_manager: UploadManager | None = None


def get_upload_manager() -> UploadManager:
    global _upload_manager
    if _upload_manager is None:
        _upload_manager = UploadManager(Path.cwd() / "sessions" / "uploads")
    return _upload_manager


def _run_upload_job(job: UploadJob) -> None:
    """Execute upload job in background thread.

    1. Copy files to raw_dir/_uploads/
    2. Convert each file via ConverterCoordinator
    3. Add converted .md files to Tantivy index
    4. Push progress events to job.event_queue
    """
    loop = asyncio.get_event_loop()

    def push_event(event_type: str, data: dict[str, Any]):
        """Push SSE-compatible progress event to the job's queue."""
        if job.event_queue is None:
            return
        try:
            frame = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            loop.call_soon_threadsafe(
                lambda f=frame: asyncio.ensure_future(job.event_queue.put(f))
            )
        except Exception:
            pass  # Queue might be closed - this is acceptable

    try:
        # Stage 1: Copy files
        upload_dir = job.raw_dir / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        job.status = "converting"
        job.progress["stage"] = "converting"
        job.progress["total"] = len(job.files)
        push_event("progress", job.progress)

        # Stage 2: Convert
        coordinator = ConverterCoordinator(enable_ocr_fallback=True)
        store = RawStore(upload_dir, job.raw_dir)

        for i, file_path in enumerate(job.files):
            if job.abort_event and job.abort_event.is_set():
                job.status = "failed"
                job.progress["error"] = "用户取消"
                push_event("error", {"message": "用户取消"})
                return

            job.progress["current"] = i + 1
            job.progress["current_file"] = file_path.name
            push_event("progress", job.progress)

            try:
                output_dir = store.map_output_path(file_path).parent
                result = coordinator.convert(
                    source=file_path,
                    output_dir=output_dir,
                )

                if result.success:
                    metadata = result.metadata or {}
                    metadata["source_path"] = str(file_path)
                    store.save(file_path, result.markdown, metadata)
                    job.progress["success_count"] += 1
                else:
                    job.progress["failed_count"] += 1
            except Exception:
                job.progress["failed_count"] += 1

        # Stage 3: Index
        if job.abort_event and job.abort_event.is_set():
            return

        job.status = "indexing"
        job.progress["stage"] = "indexing"
        push_event("progress", job.progress)

        try:
            index_mgr = TantivyIndexManager(job.index_dir)
            md_files = list(upload_dir.rglob("*.md"))
            # Exclude _ prefixed helper files (e.g., _index.md)
            md_files = [f for f in md_files if not f.name.startswith("_")]
            doc_count = 0
            for md_file in md_files:
                if job.abort_event and job.abort_event.is_set():
                    return
                try:
                    content = md_file.read_text(encoding="utf-8")
                    # Strip YAML frontmatter before indexing
                    from src.converter.frontmatter import strip_frontmatter
                    _, content = strip_frontmatter(content)
                    doc_id = str(md_file.relative_to(upload_dir)).replace("\\", "/")
                    index_mgr.add_document(
                        doc_id=doc_id,
                        title=md_file.stem,
                        content=content,
                        metadata={
                            "filename": md_file.name,
                            "source_path": str(md_file),
                        },
                    )
                    doc_count += 1
                except Exception:
                    continue

            index_mgr.commit()
            job.progress["doc_count"] = doc_count
        except Exception as e:
            job.progress["error"] = f"索引失败: {e}"

        # Stage 4: Done
        job.status = "done"
        job.progress["stage"] = "done"
        push_event("complete", job.progress)

    except Exception as e:
        job.status = "failed"
        job.progress["error"] = str(e)
        push_event("error", {"message": str(e)})
