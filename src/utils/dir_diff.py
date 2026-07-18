"""Directory comparison and incremental copy utilities.

Scans two directories, identifies added/changed/deleted/unchanged/moved files,
and optionally copies incremental changes to a target directory.

A **moved** file is one that appears "new" by relative path in the compare
directory, but whose content hash matches an existing file in the base
directory at a different path.  Moved files are NOT considered genuine
additions — they are just relocations.

This module is a pure filesystem comparison — it does NOT depend on
ConvertDB, SQLite, or any external service.
"""

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.utils.hash import calculate_hash

# Default directories to skip during scanning
_DEFAULT_SKIP_DIRS: Set[str] = {"index", "_ocr_temp", ".git", "__pycache__"}

# Extensions that are never useful for document comparison
SKIP_EXTENSIONS: Set[str] = {
    ".crdownload", ".js", ".css", ".woff", ".ttf",
    ".m4a", ".mp3", ".wav", ".mp4",
    ".xmind", ".egg-info", ".pyc", ".pyo",
}


@dataclass
class FileMeta:
    """Metadata for a single file in a scanned directory.

    Attributes:
        relative_path: forward-slash normalized path relative to scan root
        absolute_path: full filesystem path to the file
        size: file size in bytes
        mtime: modification time as Unix timestamp
        file_hash: SHA256 hexdigest, or mtime-based pseudo-hash for large files
    """

    relative_path: str
    absolute_path: Path
    size: int
    mtime: float
    file_hash: str = ""


@dataclass
class DiffEntry:
    """A single file comparison result.

    Attributes:
        relative_path: forward-slash normalized relative path
        status: one of "added", "changed", "deleted", "unchanged"
        base: file metadata from base directory (None if added)
        compare: file metadata from compare directory (None if deleted)
    """

    relative_path: str
    status: str
    base: Optional[FileMeta] = None
    compare: Optional[FileMeta] = None
    # When status == "moved", stores the relative path of the matching
    # file in base_dir that has identical content (same hash).
    moved_from: str = ""


@dataclass
class DiffResult:
    """Complete diff between two directories.

    Attributes:
        base_dir: the base (reference) directory
        compare_dir: the directory being compared against base
        entries: all DiffEntry items from the comparison
    """

    base_dir: Path
    compare_dir: Path
    entries: List[DiffEntry] = field(default_factory=list)

    @property
    def added(self) -> List[DiffEntry]:
        """Entries with status 'added' (genuine new content in compare)."""
        return [e for e in self.entries if e.status == "added"]

    @property
    def moved(self) -> List[DiffEntry]:
        """Entries with status 'moved' (same content, different path in base)."""
        return [e for e in self.entries if e.status == "moved"]

    @property
    def changed(self) -> List[DiffEntry]:
        """Entries with status 'changed' (hash or size differs)."""
        return [e for e in self.entries if e.status == "changed"]

    @property
    def deleted(self) -> List[DiffEntry]:
        """Entries with status 'deleted' (present in base, absent in compare)."""
        return [e for e in self.entries if e.status == "deleted"]

    @property
    def unchanged(self) -> List[DiffEntry]:
        """Entries with status 'unchanged' (identical in both)."""
        return [e for e in self.entries if e.status == "unchanged"]

    @property
    def has_changes(self) -> bool:
        """True if there are any genuinely added or changed entries."""
        return bool(self.added or self.changed or self.deleted)

    def summary(self) -> Dict[str, int]:
        """Return a summary count of each status.

        Returns:
            Dict with keys: added, moved, changed, deleted, unchanged, total
        """
        return {
            "added": len(self.added),
            "moved": len(self.moved),
            "changed": len(self.changed),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
            "total": len(self.entries),
        }


@dataclass
class CopyResult:
    """Result of an incremental copy operation.

    Attributes:
        copied: relative paths of files successfully copied
        skipped: relative paths skipped (destination exists, overwrite=False)
        errors: list of (relative_path, error_message) tuples
    """

    copied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)


def scan_directory(
    root: Path,
    extensions: Optional[Set[str]] = None,
    skip_dirs: Optional[Set[str]] = None,
    large_file_threshold: int = 50 * 1024 * 1024,
) -> Dict[str, FileMeta]:
    """Recursively scan a directory and return file metadata.

    Walks ``root`` using :func:`os.walk`, collecting file metadata for each
    file. Relative paths are normalized to forward slashes for cross-platform
    consistency.

    Args:
        root: directory to scan
        extensions: if not None, only include files whose suffix.lower() is
            in this set (e.g. ``{".pdf", ".docx"}``). If None, include all
            files (subject to skip_dirs and SKIP_EXTENSIONS filtering).
        skip_dirs: set of directory names to skip during traversal. Defaults
            to ``{"index", "_ocr_temp", ".git", "__pycache__"}``.
        large_file_threshold: files larger than this (bytes) skip content
            hashing and use an mtime-based pseudo-hash instead.

    Returns:
        Dict mapping forward-slash relative path to :class:`FileMeta`.
    """
    if skip_dirs is None:
        skip_dirs = _DEFAULT_SKIP_DIRS

    root = Path(root)
    result: Dict[str, FileMeta] = {}

    if not root.exists() or not root.is_dir():
        return result

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip_dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for fname in filenames:
            file_path = Path(dirpath) / fname
            ext = file_path.suffix.lower()

            # Skip known useless extensions
            if ext in SKIP_EXTENSIONS:
                continue

            # If extensions filter is set, skip non-matching files
            if extensions is not None and ext not in extensions:
                continue

            rel_path = str(file_path.relative_to(root)).replace("\\", "/")

            try:
                file_stat = file_path.stat()
                file_size = file_stat.st_size
                mtime = file_stat.st_mtime
            except OSError:
                continue

            # Compute hash: use mtime-based pseudo-hash for large files
            if file_size > large_file_threshold:
                file_hash = f"mtime:{datetime.fromtimestamp(mtime).isoformat()}"
            else:
                try:
                    file_hash = calculate_hash(file_path)
                except OSError:
                    file_hash = ""

            result[rel_path] = FileMeta(
                relative_path=rel_path,
                absolute_path=file_path,
                size=file_size,
                mtime=mtime,
                file_hash=file_hash,
            )

    return result


def compare_directories(
    base_dir: Path,
    compare_dir: Path,
    extensions: Optional[Set[str]] = None,
    skip_dirs: Optional[Set[str]] = None,
) -> DiffResult:
    """Compare two directories and return a :class:`DiffResult`.

    Scans both directories and classifies every file.

    Classification logic:
        - **added**: in compare_dir, not in base_dir by path, AND content
          hash does not match any file in base_dir → genuine new content.
        - **moved**: in compare_dir, not in base_dir by path, BUT content
          hash matches a file in base_dir at a different path → relocation,
          not genuine new content.
        - **deleted**: in base_dir, not in compare_dir.
        - **changed**: in both at the same path, but hash or size differs.
        - **unchanged**: in both at the same path, identical hash and size.

    Args:
        base_dir: the base (reference) directory — existing knowledge base
        compare_dir: the directory to compare against base — may have new content
        extensions: optional extension filter passed to :func:`scan_directory`
        skip_dirs: optional directory-name skip set passed to
            :func:`scan_directory`

    Returns:
        :class:`DiffResult` with all entries sorted by relative path.
    """
    base_path = Path(base_dir)
    compare_path = Path(compare_dir)

    base_scan = scan_directory(base_path, extensions=extensions, skip_dirs=skip_dirs)
    compare_scan = scan_directory(compare_path, extensions=extensions, skip_dirs=skip_dirs)

    # Build a reverse hash index of base_dir for "moved" detection.
    # Maps file_hash → relative_path for every base file with a non-empty hash.
    base_hash_index: Dict[str, str] = {}
    for rel, meta in base_scan.items():
        if meta.file_hash:
            base_hash_index[meta.file_hash] = rel

    all_paths: Set[str] = set(base_scan.keys()) | set(compare_scan.keys())

    entries: List[DiffEntry] = []
    for rel_path in sorted(all_paths):
        base_meta = base_scan.get(rel_path)
        compare_meta = compare_scan.get(rel_path)

        if base_meta is None and compare_meta is not None:
            # File is in compare but not in base by path.
            # Check if it's a "moved" file (same content, different path).
            comp_hash = compare_meta.file_hash
            if comp_hash and comp_hash in base_hash_index:
                # Content matches an existing base file → it's a move, not new.
                entries.append(DiffEntry(
                    relative_path=rel_path,
                    status="moved",
                    base=None,
                    compare=compare_meta,
                    moved_from=base_hash_index[comp_hash],
                ))
            else:
                entries.append(DiffEntry(
                    relative_path=rel_path,
                    status="added",
                    base=None,
                    compare=compare_meta,
                ))
        elif base_meta is not None and compare_meta is None:
            entries.append(DiffEntry(
                relative_path=rel_path,
                status="deleted",
                base=base_meta,
                compare=None,
            ))
        elif base_meta is not None and compare_meta is not None:
            if (base_meta.file_hash == compare_meta.file_hash
                    and base_meta.size == compare_meta.size):
                entries.append(DiffEntry(
                    relative_path=rel_path,
                    status="unchanged",
                    base=base_meta,
                    compare=compare_meta,
                ))
            else:
                entries.append(DiffEntry(
                    relative_path=rel_path,
                    status="changed",
                    base=base_meta,
                    compare=compare_meta,
                ))

    return DiffResult(base_dir=base_path, compare_dir=compare_path, entries=entries)


def copy_incremental(
    diff_result: DiffResult,
    target_dir: Path,
    copy_added: bool = True,
    copy_changed: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
) -> CopyResult:
    """Copy incremental changes (added/changed files) to a target directory.

    Copies files from ``diff_result.compare_dir`` to ``target_dir``,
    preserving the relative path structure.

    Args:
        diff_result: the diff produced by :func:`compare_directories`
        target_dir: destination directory for copied files
        copy_added: if True, copy files with status "added"
        copy_changed: if True, copy files with status "changed"
        dry_run: if True, do not actually copy — just record what would
            be copied in the ``copied`` list
        overwrite: if False, skip files that already exist in target_dir

    Returns:
        :class:`CopyResult` with copied, skipped, and errors lists.
    """
    result = CopyResult()
    target_path = Path(target_dir)

    candidates: List[DiffEntry] = []
    if copy_added:
        candidates.extend(diff_result.added)
    if copy_changed:
        candidates.extend(diff_result.changed)

    for entry in candidates:
        src = diff_result.compare_dir / entry.relative_path
        dest = target_path / entry.relative_path

        if dest.exists() and not overwrite:
            result.skipped.append(entry.relative_path)
            continue

        if dry_run:
            result.copied.append(entry.relative_path)
            continue

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
            result.copied.append(entry.relative_path)
        except OSError as e:
            result.errors.append((entry.relative_path, str(e)))

    return result
