"""
Unit tests for src/utils/dir_diff.py — directory comparison utilities.

Covers scan_directory, compare_directories, copy_incremental,
and the FileMeta / DiffEntry / DiffResult / CopyResult dataclasses.

All filesystem tests use the ``tmp_path`` fixture — no hardcoded paths.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.dir_diff import (
    CopyResult,
    DiffEntry,
    DiffResult,
    FileMeta,
    compare_directories,
    copy_incremental,
    scan_directory,
)

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """Create an empty base directory for comparisons."""
    d = tmp_path / "base"
    d.mkdir()
    return d


@pytest.fixture
def compare_dir(tmp_path: Path) -> Path:
    """Create an empty compare directory for comparisons."""
    d = tmp_path / "compare"
    d.mkdir()
    return d


@pytest.fixture
def target_dir(tmp_path: Path) -> Path:
    """Create an empty target directory for copy operations."""
    d = tmp_path / "target"
    d.mkdir()
    return d


def _write_file(path: Path, content: str = "hello") -> Path:
    """Write *content* to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ── 1. scan_directory ──────────────────────────────────────────────────


class TestScanDirectory:
    """Tests for scan_directory()."""

    def test_scan_empty_directory(self, base_dir: Path):
        """An empty directory returns an empty dict."""
        result = scan_directory(base_dir)
        assert result == {}

    def test_scan_finds_all_files(self, base_dir: Path):
        """Three files at root level → three entries in the result."""
        for name in ("a.pdf", "b.docx", "c.txt"):
            _write_file(base_dir / name)
        result = scan_directory(base_dir)
        assert len(result) == 3
        for name in ("a.pdf", "b.docx", "c.txt"):
            assert name in result

    def test_scan_normalizes_paths(self, base_dir: Path):
        """Files in nested subdirs use forward-slash relative paths."""
        _write_file(base_dir / "sub" / "deep" / "file.pdf")
        _write_file(base_dir / "top.pdf")
        result = scan_directory(base_dir)
        assert "sub/deep/file.pdf" in result
        assert "top.pdf" in result
        # No backslashes in any key (Windows-safe)
        for key in result:
            assert "\\" not in key

    def test_scan_extension_filter(self, base_dir: Path):
        """Only files matching the extension filter are returned."""
        _write_file(base_dir / "a.pdf")
        _write_file(base_dir / "b.docx")
        _write_file(base_dir / "c.PDF")  # uppercase
        result = scan_directory(base_dir, extensions={".pdf"})
        # .pdf matches (case-insensitive on some systems, but at least one)
        assert any(k == "a.pdf" for k in result)
        assert "b.docx" not in result

    def test_scan_skip_dirs(self, base_dir: Path):
        """Files inside skip_dirs directories are excluded."""
        _write_file(base_dir / "keep.pdf")
        _write_file(base_dir / ".git" / "config")
        _write_file(base_dir / "__pycache__" / "x.pyc")
        result = scan_directory(base_dir, skip_dirs={".git", "__pycache__"})
        assert "keep.pdf" in result
        assert all(".git" not in k for k in result)
        assert all("__pycache__" not in k for k in result)

    def test_scan_computes_hash(self, base_dir: Path):
        """Returned FileMeta has a non-empty file_hash."""
        _write_file(base_dir / "file.pdf", "content for hashing")
        result = scan_directory(base_dir)
        meta = result["file.pdf"]
        assert meta.file_hash
        assert isinstance(meta.file_hash, str)

    def test_scan_large_file_uses_mtime(self, base_dir: Path):
        """Large files get an mtime-based hash instead of content hash."""
        big = _write_file(base_dir / "big.bin", "x")
        result = scan_directory(base_dir, large_file_threshold=0)
        meta = result["big.bin"]
        assert meta.file_hash.startswith("mtime:")

    def test_scan_filemeta_fields(self, base_dir: Path):
        """FileMeta fields (absolute_path, size, mtime) are correct."""
        f = _write_file(base_dir / "doc.pdf", "12345")
        result = scan_directory(base_dir)
        meta = result["doc.pdf"]
        assert meta.relative_path == "doc.pdf"
        assert meta.absolute_path == f.resolve()
        assert meta.size == 5
        assert meta.mtime > 0

    def test_scan_preserves_extension_case(self, base_dir: Path):
        """Extension filter comparison handles mixed case."""
        _write_file(base_dir / "A.PDF")
        _write_file(base_dir / "B.pdf")
        result = scan_directory(base_dir, extensions={".pdf"})
        # both should match (case-insensitive)
        assert len(result) == 2


# ── 2. compare_directories ─────────────────────────────────────────────


class TestCompareDirectories:
    """Tests for compare_directories()."""

    def test_compare_identical_dirs(self, base_dir: Path, compare_dir: Path):
        """Identical content in both dirs → all 'unchanged'."""
        for name in ("a.pdf", "b.txt"):
            _write_file(base_dir / name, "same")
            _write_file(compare_dir / name, "same")
        result = compare_directories(base_dir, compare_dir)
        assert len(result.entries) == 2
        assert all(e.status == "unchanged" for e in result.entries)

    def test_compare_added_files(self, base_dir: Path, compare_dir: Path):
        """Extra files in compare_dir → status 'added'."""
        _write_file(base_dir / "a.pdf", "x")
        _write_file(compare_dir / "a.pdf", "x")
        _write_file(compare_dir / "new.pdf", "y")
        result = compare_directories(base_dir, compare_dir)
        statuses = {e.relative_path: e.status for e in result.entries}
        assert statuses["new.pdf"] == "added"

    def test_compare_deleted_files(self, base_dir: Path, compare_dir: Path):
        """Extra files in base_dir → status 'deleted'."""
        _write_file(base_dir / "a.pdf", "x")
        _write_file(base_dir / "gone.pdf", "y")
        _write_file(compare_dir / "a.pdf", "x")
        result = compare_directories(base_dir, compare_dir)
        statuses = {e.relative_path: e.status for e in result.entries}
        assert statuses["gone.pdf"] == "deleted"

    def test_compare_changed_files(self, base_dir: Path, compare_dir: Path):
        """Same path, different content → status 'changed'."""
        _write_file(base_dir / "shared.pdf", "old content")
        _write_file(compare_dir / "shared.pdf", "new content")
        result = compare_directories(base_dir, compare_dir)
        statuses = {e.relative_path: e.status for e in result.entries}
        assert statuses["shared.pdf"] == "changed"

    def test_compare_mixed(self, base_dir: Path, compare_dir: Path):
        """A combination of added, deleted, changed, and unchanged."""
        _write_file(base_dir / "same.pdf", "v1")
        _write_file(compare_dir / "same.pdf", "v1")

        _write_file(base_dir / "mod.pdf", "old")
        _write_file(compare_dir / "mod.pdf", "new")

        _write_file(base_dir / "del.pdf", "only base")
        _write_file(compare_dir / "add.pdf", "only compare")

        result = compare_directories(base_dir, compare_dir)
        statuses = {e.relative_path: e.status for e in result.entries}
        assert statuses["same.pdf"] == "unchanged"
        assert statuses["mod.pdf"] == "changed"
        assert statuses["del.pdf"] == "deleted"
        assert statuses["add.pdf"] == "added"
        assert len(result.entries) == 4

    def test_compare_with_extension_filter(self, base_dir: Path, compare_dir: Path):
        """Extension filter limits which files are compared."""
        _write_file(base_dir / "a.pdf", "1")
        _write_file(base_dir / "b.txt", "1")
        _write_file(compare_dir / "a.pdf", "1")
        _write_file(compare_dir / "b.txt", "1")
        result = compare_directories(base_dir, compare_dir, extensions={".pdf"})
        paths = {e.relative_path for e in result.entries}
        assert "a.pdf" in paths
        assert "b.txt" not in paths

    def test_compare_empty_dirs(self, base_dir: Path, compare_dir: Path):
        """Two empty directories → empty DiffResult."""
        result = compare_directories(base_dir, compare_dir)
        assert len(result.entries) == 0

    def test_compare_result_properties(self, base_dir: Path, compare_dir: Path):
        """DiffResult.added/.changed/.deleted/.unchanged filter lists."""
        _write_file(base_dir / "same.pdf", "v1")
        _write_file(compare_dir / "same.pdf", "v1")
        _write_file(base_dir / "mod.pdf", "old")
        _write_file(compare_dir / "mod.pdf", "new")
        _write_file(compare_dir / "add.pdf", "new")
        _write_file(base_dir / "del.pdf", "gone")

        result = compare_directories(base_dir, compare_dir)
        added_paths = {e.relative_path for e in result.added}
        changed_paths = {e.relative_path for e in result.changed}
        deleted_paths = {e.relative_path for e in result.deleted}
        unchanged_paths = {e.relative_path for e in result.unchanged}

        assert added_paths == {"add.pdf"}
        assert changed_paths == {"mod.pdf"}
        assert deleted_paths == {"del.pdf"}
        assert unchanged_paths == {"same.pdf"}

    def test_compare_has_changes_true(self, base_dir: Path, compare_dir: Path):
        """has_changes is True when diffs exist."""
        _write_file(compare_dir / "extra.pdf", "x")
        result = compare_directories(base_dir, compare_dir)
        assert result.has_changes is True

    def test_compare_moved_file_detection(self, base_dir: Path, compare_dir: Path):
        """A file with identical content at a different path is 'moved', not 'added'."""
        content = "identical content here"
        _write_file(base_dir / "original.pdf", content)
        _write_file(compare_dir / "relocated/nested.pdf", content)

        result = compare_directories(base_dir, compare_dir)
        assert len(result.moved) == 1
        assert result.moved[0].relative_path == "relocated/nested.pdf"
        assert result.moved[0].moved_from == "original.pdf"
        assert len(result.added) == 0

    def test_compare_moved_not_triggered_by_different_content(self, base_dir: Path, compare_dir: Path):
        """Different content at a new path is 'added', not 'moved'."""
        _write_file(base_dir / "original.pdf", "content A")
        _write_file(compare_dir / "new_path.pdf", "content B")

        result = compare_directories(base_dir, compare_dir)
        assert len(result.added) == 1
        assert len(result.moved) == 0

    def test_compare_moved_summary(self, base_dir: Path, compare_dir: Path):
        """summary() counts moved files correctly."""
        content = "shared content"
        _write_file(base_dir / "a.pdf", content)
        _write_file(compare_dir / "b/c.pdf", content)  # moved
        _write_file(compare_dir / "d.pdf", "new")       # genuinely added

        result = compare_directories(base_dir, compare_dir)
        summary = result.summary()
        assert summary["added"] == 1
        assert summary["moved"] == 1

    def test_compare_has_changes_false(self, base_dir: Path, compare_dir: Path):
        """has_changes is False when dirs are identical."""
        _write_file(base_dir / "a.pdf", "x")
        _write_file(compare_dir / "a.pdf", "x")
        result = compare_directories(base_dir, compare_dir)
        assert result.has_changes is False

    def test_compare_nested_paths(self, base_dir: Path, compare_dir: Path):
        """Files in nested subdirs are compared by normalized relative path."""
        _write_file(base_dir / "sub" / "deep" / "file.pdf", "old")
        _write_file(compare_dir / "sub" / "deep" / "file.pdf", "new")
        result = compare_directories(base_dir, compare_dir)
        statuses = {e.relative_path: e.status for e in result.entries}
        assert statuses["sub/deep/file.pdf"] == "changed"


# ── 3. copy_incremental ────────────────────────────────────────────────


class TestCopyIncremental:
    """Tests for copy_incremental()."""

    def test_copy_added_files(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """Added files are copied to target, preserving relative paths."""
        _write_file(compare_dir / "new.pdf", "fresh content")
        diff = compare_directories(base_dir, compare_dir)
        result = copy_incremental(diff, target_dir)
        assert (target_dir / "new.pdf").exists()
        assert (target_dir / "new.pdf").read_text(encoding="utf-8") == "fresh content"
        assert "new.pdf" in result.copied

    def test_copy_changed_files(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """Changed files are copied to target."""
        _write_file(base_dir / "mod.pdf", "old")
        _write_file(compare_dir / "mod.pdf", "new")
        diff = compare_directories(base_dir, compare_dir)
        result = copy_incremental(diff, target_dir)
        assert (target_dir / "mod.pdf").read_text(encoding="utf-8") == "new"
        assert "mod.pdf" in result.copied

    def test_copy_dry_run(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """dry_run=True lists files but does not copy them."""
        _write_file(compare_dir / "ghost.pdf", "data")
        diff = compare_directories(base_dir, compare_dir)
        result = copy_incremental(diff, target_dir, dry_run=True)
        assert "ghost.pdf" in result.copied
        assert not (target_dir / "ghost.pdf").exists()

    def test_copy_no_overwrite(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """Existing target files are skipped (in CopyResult.skipped)."""
        _write_file(compare_dir / "exists.pdf", "new")
        _write_file(target_dir / "exists.pdf", "pre-existing")
        diff = compare_directories(base_dir, compare_dir)
        result = copy_incremental(diff, target_dir, overwrite=False)
        assert "exists.pdf" in result.skipped
        assert (target_dir / "exists.pdf").read_text(encoding="utf-8") == "pre-existing"

    def test_copy_overwrite(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """overwrite=True replaces existing target files."""
        _write_file(compare_dir / "exists.pdf", "new version")
        _write_file(target_dir / "exists.pdf", "old version")
        diff = compare_directories(base_dir, compare_dir)
        result = copy_incremental(diff, target_dir, overwrite=True)
        assert "exists.pdf" in result.copied
        assert "exists.pdf" not in result.skipped
        assert (target_dir / "exists.pdf").read_text(encoding="utf-8") == "new version"

    def test_copy_preserves_directory_structure(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """Nested subdirectory files get correct parent dirs created."""
        _write_file(compare_dir / "sub" / "deep" / "file.pdf", "nested")
        diff = compare_directories(base_dir, compare_dir)
        copy_incremental(diff, target_dir)
        assert (target_dir / "sub" / "deep" / "file.pdf").exists()

    def test_copy_errors_handled(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """Permission error on copy → entry in CopyResult.errors."""
        _write_file(compare_dir / "bad.pdf", "content")
        diff = compare_directories(base_dir, compare_dir)
        with patch("shutil.copy2", side_effect=PermissionError("denied")):
            result = copy_incremental(diff, target_dir)
        assert len(result.errors) == 1
        err_path, err_msg = result.errors[0]
        assert "bad.pdf" in err_path

    def test_copy_selective_added_only(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """copy_added=False, copy_changed=False copies nothing."""
        _write_file(compare_dir / "added.pdf", "a")
        diff = compare_directories(base_dir, compare_dir)
        result = copy_incremental(diff, target_dir, copy_added=False, copy_changed=False)
        assert result.copied == []
        assert not (target_dir / "added.pdf").exists()

    def test_copy_selective_changed_only(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """copy_added=False, copy_changed=True only copies changed files."""
        _write_file(base_dir / "mod.pdf", "old")
        _write_file(compare_dir / "mod.pdf", "new")
        _write_file(compare_dir / "extra.pdf", "extra")
        diff = compare_directories(base_dir, compare_dir)
        result = copy_incremental(diff, target_dir, copy_added=False, copy_changed=True)
        assert "mod.pdf" in result.copied
        assert "extra.pdf" not in result.copied
        assert not (target_dir / "extra.pdf").exists()

    def test_copy_source_path_from_compare(self, base_dir: Path, compare_dir: Path, target_dir: Path):
        """Copy reads from compare_dir (not base_dir) for added files."""
        _write_file(compare_dir / "src.pdf", "from compare")
        diff = compare_directories(base_dir, compare_dir)
        copy_incremental(diff, target_dir)
        assert (target_dir / "src.pdf").read_text(encoding="utf-8") == "from compare"


# ── 4. Dataclass tests ─────────────────────────────────────────────────


class TestDataclasses:
    """Tests for FileMeta, DiffEntry, DiffResult, CopyResult dataclasses."""

    def test_filemeta_defaults(self):
        """FileMeta.file_hash defaults to empty string."""
        meta = FileMeta(
            relative_path="a.pdf",
            absolute_path=Path("/tmp/a.pdf"),
            size=100,
            mtime=1234567890.0,
        )
        assert meta.file_hash == ""
        assert meta.relative_path == "a.pdf"
        assert meta.size == 100

    def test_diff_entry_fields(self):
        """DiffEntry stores base and compare FileMeta correctly."""
        base_meta = FileMeta("x.pdf", Path("/b/x.pdf"), 10, 0.0, "hash_base")
        comp_meta = FileMeta("x.pdf", Path("/c/x.pdf"), 20, 1.0, "hash_comp")
        entry = DiffEntry(
            relative_path="x.pdf",
            status="changed",
            base=base_meta,
            compare=comp_meta,
        )
        assert entry.status == "changed"
        assert entry.base is base_meta
        assert entry.compare is comp_meta
        assert base_meta.size == 10
        assert comp_meta.size == 20

    def test_diff_entry_optional_fields(self):
        """DiffEntry base/compare can be None."""
        entry = DiffEntry(relative_path="new.pdf", status="added")
        assert entry.base is None
        assert entry.compare is None

    def test_diff_result_summary(self, base_dir: Path, compare_dir: Path):
        """summary() returns correct counts per status."""
        _write_file(base_dir / "same.pdf", "v1")
        _write_file(compare_dir / "same.pdf", "v1")
        _write_file(base_dir / "mod.pdf", "old")
        _write_file(compare_dir / "mod.pdf", "new")
        _write_file(compare_dir / "add.pdf", "genuinely-new-content")
        _write_file(base_dir / "del.pdf", "x")

        result = compare_directories(base_dir, compare_dir)
        summary = result.summary()
        assert summary["unchanged"] == 1
        assert summary["changed"] == 1
        assert summary["added"] == 1
        assert summary["moved"] == 0
        assert summary["deleted"] == 1
        assert summary["total"] == 4

    def test_diff_result_default_entries(self, tmp_path: Path):
        """DiffResult.entries defaults to empty list."""
        dr = DiffResult(base_dir=tmp_path / "b", compare_dir=tmp_path / "c")
        assert dr.entries == []
        assert dr.added == []
        assert dr.moved == []
        assert dr.changed == []
        assert dr.deleted == []
        assert dr.unchanged == []
        assert dr.has_changes is False

    def test_copy_result_defaults(self):
        """Empty CopyResult has empty lists."""
        cr = CopyResult()
        assert cr.copied == []
        assert cr.skipped == []
        assert cr.errors == []

    def test_diff_result_stores_dirs(self, tmp_path: Path):
        """DiffResult stores base_dir and compare_dir paths."""
        b = tmp_path / "base"
        c = tmp_path / "cmp"
        dr = DiffResult(base_dir=b, compare_dir=c)
        assert dr.base_dir == b
        assert dr.compare_dir == c

    def test_diff_result_summary_empty(self, base_dir: Path, compare_dir: Path):
        """summary() of empty result has all-zero counts."""
        result = compare_directories(base_dir, compare_dir)
        summary = result.summary()
        assert summary["added"] == 0
        assert summary["moved"] == 0
        assert summary["changed"] == 0
        assert summary["deleted"] == 0
        assert summary["unchanged"] == 0
        assert summary["total"] == 0
