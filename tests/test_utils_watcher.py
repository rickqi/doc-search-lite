"""Unit tests for FileWatcher class."""

import time
from pathlib import Path

import pytest

from src.storage.metadata import MetadataManager
from src.utils.file_watcher import ChangeSet, FileWatcher
from src.utils.hash import calculate_hash


@pytest.fixture
def temp_source_dir(tmp_path: Path) -> Path:
    """Create a temporary source directory with test files."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    return source_dir


@pytest.fixture
def temp_index(tmp_path: Path) -> Path:
    """Create a temporary index.json file path."""
    return tmp_path / "index.json"


@pytest.fixture
def watcher() -> FileWatcher:
    """Create a FileWatcher instance with default settings."""
    return FileWatcher()


@pytest.fixture
def metadata_manager(temp_index: Path) -> MetadataManager:
    """Create a MetadataManager instance with temporary index."""
    return MetadataManager(index_path=temp_index)


class TestChangeSet:
    """Test ChangeSet dataclass."""

    def test_empty_changeset(self):
        """Test creating empty ChangeSet."""
        change_set = ChangeSet()
        assert change_set.added == []
        assert change_set.modified == []
        assert change_set.deleted == []
        assert change_set.unchanged == []
        assert change_set.has_changes is False
        assert change_set.total_changes == 0

    def test_changeset_with_files(self):
        """Test ChangeSet with files."""
        change_set = ChangeSet(
            added=[Path("/a/1.pdf"), Path("/a/2.pdf")],
            modified=[Path("/a/3.pdf")],
            deleted=[Path("/a/4.pdf")],
            unchanged=[Path("/a/5.pdf")],
        )
        assert len(change_set.added) == 2
        assert len(change_set.modified) == 1
        assert len(change_set.deleted) == 1
        assert len(change_set.unchanged) == 1
        assert change_set.has_changes is True
        assert change_set.total_changes == 4

    def test_changeset_repr(self):
        """Test ChangeSet string representation."""
        change_set = ChangeSet(
            added=[Path("/a/1.pdf")],
            modified=[Path("/a/2.pdf")],
            deleted=[Path("/a/3.pdf")],
        )
        repr_str = repr(change_set)
        assert "ChangeSet" in repr_str
        assert "added=1" in repr_str
        assert "modified=1" in repr_str
        assert "deleted=1" in repr_str


class TestFileWatcherInit:
    """Test FileWatcher initialization."""

    def test_init_defaults(self):
        """Test initialization with default settings."""
        watcher = FileWatcher()
        assert watcher.hash_algorithm == "sha256"
        assert watcher.use_mtime_check is True
        assert watcher.use_hash_check is True

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        watcher = FileWatcher(
            hash_algorithm="md5",
            use_mtime_check=False,
            use_hash_check=False,
        )
        assert watcher.hash_algorithm == "md5"
        assert watcher.use_mtime_check is False
        assert watcher.use_hash_check is False


class TestFileWatcherDetectChanges:
    """Test detect_changes method."""

    def test_detect_empty_directory(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting changes in empty directory."""
        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert change_set.has_changes is False
        assert len(change_set.added) == 0
        assert len(change_set.modified) == 0
        assert len(change_set.deleted) == 0
        assert len(change_set.unchanged) == 0

    def test_detect_new_files(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting new files not in metadata."""
        # Create test files
        file1 = temp_source_dir / "doc1.pdf"
        file2 = temp_source_dir / "doc2.pdf"
        file1.write_text("content1")
        file2.write_text("content2")

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)

        assert len(change_set.added) == 2
        assert len(change_set.modified) == 0
        assert len(change_set.deleted) == 0
        assert len(change_set.unchanged) == 0

    def test_detect_unchanged_files(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting unchanged files."""
        # Create test file
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Add to metadata
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": calculate_hash(file1),
                "modified_time": file1.stat().st_mtime,
            },
        )

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)

        assert len(change_set.added) == 0
        assert len(change_set.modified) == 0
        assert len(change_set.deleted) == 0
        assert len(change_set.unchanged) == 1

    def test_detect_modified_files(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting modified files."""
        # Create test file
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Add to metadata with old hash
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": "old_hash_value",
                "modified_time": 0,  # Old mtime
            },
        )

        # Modify the file
        time.sleep(0.1)  # Ensure mtime changes
        file1.write_text("modified content")

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)

        assert len(change_set.added) == 0
        assert len(change_set.modified) == 1
        assert len(change_set.deleted) == 0
        assert len(change_set.unchanged) == 0

    def test_detect_deleted_files(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting deleted files."""
        # Create test file
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Add to metadata
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": calculate_hash(file1),
                "modified_time": file1.stat().st_mtime,
            },
        )

        # Delete the file
        file1.unlink()

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)

        assert len(change_set.added) == 0
        assert len(change_set.modified) == 0
        assert len(change_set.deleted) == 1
        assert len(change_set.unchanged) == 0

    def test_detect_mixed_changes(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting mixed changes (new, modified, deleted, unchanged)."""
        # Create test files
        file1 = temp_source_dir / "doc1.pdf"  # Will be unchanged
        file2 = temp_source_dir / "doc2.pdf"  # Will be modified
        file3 = temp_source_dir / "doc3.pdf"  # Will be deleted (not created)
        file4 = temp_source_dir / "doc4.pdf"  # Will be new

        file1.write_text("content1")
        file2.write_text("content2")
        file4.write_text("content4")

        # Add to metadata
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": calculate_hash(file1),
                "modified_time": file1.stat().st_mtime,
            },
        )

        metadata_manager.save(
            file2,
            {
                "source_path": file2,
                "content_hash": "old_hash",
                "modified_time": 0,
            },
        )

        # file3 is in metadata but not on disk (deleted)
        metadata_manager.save(
            file3,
            {
                "source_path": file3,
                "content_hash": "deleted_hash",
                "modified_time": 0,
            },
        )

        # Modify file2
        time.sleep(0.1)
        file2.write_text("modified content2")

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)

        assert len(change_set.added) == 1  # file4
        assert len(change_set.modified) == 1  # file2
        assert len(change_set.deleted) == 1  # file3
        assert len(change_set.unchanged) == 1  # file1

    def test_detect_with_extension_filter(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting changes with extension filter."""
        # Create test files with different extensions
        file1 = temp_source_dir / "doc1.pdf"
        file2 = temp_source_dir / "doc2.docx"
        file3 = temp_source_dir / "doc3.txt"

        file1.write_text("content1")
        file2.write_text("content2")
        file3.write_text("content3")

        # Filter to only PDF files
        change_set = watcher.detect_changes(
            temp_source_dir, metadata_manager, extensions={".pdf"}
        )

        assert len(change_set.added) == 1
        assert change_set.added[0].suffix == ".pdf"

    def test_detect_with_multiple_extensions(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting changes with multiple extension filters."""
        # Create test files
        file1 = temp_source_dir / "doc1.pdf"
        file2 = temp_source_dir / "doc2.docx"
        file3 = temp_source_dir / "doc3.txt"

        file1.write_text("content1")
        file2.write_text("content2")
        file3.write_text("content3")

        # Filter to PDF and DOCX
        change_set = watcher.detect_changes(
            temp_source_dir, metadata_manager, extensions={".pdf", ".docx"}
        )

        assert len(change_set.added) == 2

    def test_detect_non_recursive(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting changes non-recursively."""
        # Create files in root and subdirectory
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        subdir = temp_source_dir / "subdir"
        subdir.mkdir()
        file2 = subdir / "doc2.pdf"
        file2.write_text("content2")

        # Non-recursive scan
        change_set = watcher.detect_changes(
            temp_source_dir, metadata_manager, recursive=False
        )

        assert len(change_set.added) == 1
        assert change_set.added[0] == file1.resolve()

    def test_detect_recursive(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting changes recursively."""
        # Create files in root and nested subdirectories
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        subdir = temp_source_dir / "subdir"
        subdir.mkdir()
        file2 = subdir / "doc2.pdf"
        file2.write_text("content2")

        nested = subdir / "nested"
        nested.mkdir()
        file3 = nested / "doc3.pdf"
        file3.write_text("content3")

        # Recursive scan (default)
        change_set = watcher.detect_changes(
            temp_source_dir, metadata_manager, recursive=True
        )

        assert len(change_set.added) == 3


class TestFileWatcherMtimeCheck:
    """Test modification time check functionality."""

    def test_mtime_only_check(
        self,
        temp_source_dir: Path,
        temp_index: Path,
        metadata_manager: MetadataManager,
    ):
        """Test using only mtime check (no hash)."""
        watcher = FileWatcher(use_mtime_check=True, use_hash_check=False)

        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Add to metadata with old mtime
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": calculate_hash(file1),
                "modified_time": 0,  # Old mtime
            },
        )

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert len(change_set.modified) == 1


class TestFileWatcherHashCheck:
    """Test content hash check functionality."""

    def test_hash_only_check(
        self,
        temp_source_dir: Path,
        temp_index: Path,
        metadata_manager: MetadataManager,
    ):
        """Test using only hash check (no mtime)."""
        watcher = FileWatcher(use_mtime_check=False, use_hash_check=True)

        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Add to metadata with correct mtime but wrong hash
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": "wrong_hash",
                "modified_time": file1.stat().st_mtime,
            },
        )

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert len(change_set.modified) == 1

    def test_same_content_different_mtime(
        self,
        temp_source_dir: Path,
        temp_index: Path,
        metadata_manager: MetadataManager,
    ):
        """Test that same content with different mtime is not marked as modified."""
        watcher = FileWatcher(use_mtime_check=True, use_hash_check=True)

        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Add to metadata with correct hash but old mtime
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": calculate_hash(file1),
                "modified_time": 0,  # Old mtime, but hash is correct
            },
        )

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)
        # File should be unchanged because hash matches
        assert len(change_set.unchanged) == 1
        assert len(change_set.modified) == 0


class TestFileWatcherUtilityMethods:
    """Test utility methods."""

    def test_get_file_hash(self, watcher: FileWatcher, temp_source_dir: Path):
        """Test get_file_hash method."""
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        hash_value = watcher.get_file_hash(file1)
        assert hash_value == calculate_hash(file1)
        assert len(hash_value) == 64  # SHA-256 produces 64 hex chars

    def test_get_file_hash_not_found(self, watcher: FileWatcher):
        """Test get_file_hash with non-existent file."""
        with pytest.raises(FileNotFoundError):
            watcher.get_file_hash(Path("/nonexistent/file.pdf"))

    def test_get_file_mtime(self, watcher: FileWatcher, temp_source_dir: Path):
        """Test get_file_mtime method."""
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        mtime = watcher.get_file_mtime(file1)
        assert isinstance(mtime, float)
        assert mtime > 0

    def test_get_file_mtime_not_found(self, watcher: FileWatcher):
        """Test get_file_mtime with non-existent file."""
        with pytest.raises(FileNotFoundError):
            watcher.get_file_mtime(Path("/nonexistent/file.pdf"))


class TestFileWatcherDifferentAlgorithms:
    """Test different hash algorithms."""

    def test_md5_algorithm(self, temp_source_dir: Path, temp_index: Path):
        """Test using MD5 hash algorithm."""
        watcher = FileWatcher(hash_algorithm="md5")
        metadata_manager = MetadataManager(index_path=temp_index)

        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Add to metadata with MD5 hash
        hash_value = calculate_hash(file1, "md5")
        metadata_manager.save(
            file1,
            {
                "source_path": file1,
                "content_hash": hash_value,
                "modified_time": file1.stat().st_mtime,
            },
        )

        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert len(change_set.unchanged) == 1


class TestFileWatcherEdgeCases:
    """Test edge cases."""

    def test_nonexistent_directory(
        self,
        watcher: FileWatcher,
        tmp_path: Path,
        metadata_manager: MetadataManager,
    ):
        """Test detecting changes in non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        change_set = watcher.detect_changes(nonexistent, metadata_manager)
        assert change_set.has_changes is False

    def test_metadata_outside_source_dir(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        temp_index: Path,
        tmp_path: Path,
    ):
        """Test that files outside source directory are not detected as deleted."""
        metadata_manager = MetadataManager(index_path=temp_index)

        # Create file in source dir
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Create file outside source dir and add to metadata
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        file2 = other_dir / "doc2.pdf"
        file2.write_text("content2")

        metadata_manager.save(
            file2,
            {
                "source_path": file2,
                "content_hash": calculate_hash(file2),
                "modified_time": file2.stat().st_mtime,
            },
        )

        # Scan source dir only
        change_set = watcher.detect_changes(temp_source_dir, metadata_manager)

        # file2 should NOT be detected as deleted
        assert len(change_set.deleted) == 0
        assert len(change_set.added) == 1  # file1

    def test_symlink_handling(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test handling of symbolic links."""
        # Create actual file
        file1 = temp_source_dir / "doc1.pdf"
        file1.write_text("content1")

        # Create symlink (on systems that support it)
        try:
            link = temp_source_dir / "link_to_doc1.pdf"
            link.symlink_to(file1)

            change_set = watcher.detect_changes(temp_source_dir, metadata_manager)
            # Both file and symlink should be detected, but symlinks resolving
            # to the same target as the original are deduplicated by resolve()
            assert len(change_set.added) in (1, 2)
        except OSError:
            # Symlinks not supported on this system
            pytest.skip("Symlinks not supported")


class TestFileWatcherIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(
        self,
        watcher: FileWatcher,
        temp_source_dir: Path,
        metadata_manager: MetadataManager,
    ):
        """Test complete workflow with multiple scans."""
        # Initial scan - empty
        change_set1 = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert change_set1.has_changes is False

        # Add files
        file1 = temp_source_dir / "doc1.pdf"
        file2 = temp_source_dir / "doc2.pdf"
        file1.write_text("content1")
        file2.write_text("content2")

        # Second scan - detect new files
        change_set2 = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert len(change_set2.added) == 2

        # Update metadata
        for f in [file1, file2]:
            metadata_manager.save(
                f,
                {
                    "source_path": f,
                    "content_hash": calculate_hash(f),
                    "modified_time": f.stat().st_mtime,
                },
            )

        # Third scan - no changes
        change_set3 = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert len(change_set3.unchanged) == 2

        # Modify file1
        time.sleep(0.1)
        file1.write_text("modified content1")

        # Fourth scan - detect modification
        change_set4 = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert len(change_set4.modified) == 1
        assert len(change_set4.unchanged) == 1

        # Delete file2
        file2.unlink()

        # Fifth scan - detect deletion
        change_set5 = watcher.detect_changes(temp_source_dir, metadata_manager)
        assert len(change_set5.deleted) == 1
        assert len(change_set5.modified) == 1  # file1 still modified from previous
