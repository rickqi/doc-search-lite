"""Unit tests for MarkdownStore storage implementation."""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.storage.base import DocumentRecord
from src.storage.markdown_store import MarkdownStore


@pytest.fixture
def temp_dirs():
    """Create temporary input and output directories."""
    input_dir = Path(tempfile.mkdtemp())
    output_dir = Path(tempfile.mkdtemp())
    yield input_dir, output_dir
    # Cleanup
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture
def store(temp_dirs):
    """Create a MarkdownStore instance."""
    input_dir, output_dir = temp_dirs
    return MarkdownStore(input_dir, output_dir)


@pytest.fixture
def sample_record(temp_dirs):
    """Create a sample DocumentRecord."""
    input_dir, _ = temp_dirs
    source_path = input_dir / "test.pdf"
    return DocumentRecord(
        id="test123",
        source_path=source_path,
        output_path=Path("output/test.md"),
        title="Test Document",
        content_hash="abc123",
        file_size=1024,
        file_mtime=datetime.now(),
        metadata={"author": "Test Author"},
        keywords=["test", "document"],
        sections=["Introduction", "Conclusion"],
    )


@pytest.fixture
def sample_content():
    """Sample markdown content."""
    return """# Test Document

This is a test document with some content.

## Introduction

The introduction section.

## Conclusion

The conclusion section.

![Test Image](images/test.png)
"""


class TestMarkdownStoreInit:
    """Tests for MarkdownStore initialization."""

    def test_init_creates_output_dir(self, temp_dirs):
        """Output directory should be created if it doesn't exist."""
        input_dir, output_dir = temp_dirs
        # Remove output dir
        shutil.rmtree(output_dir)
        assert not output_dir.exists()

        # Init should create it
        store = MarkdownStore(input_dir, output_dir)
        assert output_dir.exists()

    def test_init_with_pathlib_paths(self, temp_dirs):
        """Should work with pathlib.Path objects."""
        input_dir, output_dir = temp_dirs
        store = MarkdownStore(Path(input_dir), Path(output_dir))
        assert store.input_base == Path(input_dir).resolve()
        assert store.output_base == Path(output_dir).resolve()


class TestMirrorDirectoryStructure:
    """Tests for mirror directory structure."""

    def test_simple_filename(self, store, temp_dirs):
        """Simple filename should map to output/filename.md."""
        input_dir, _ = temp_dirs
        source_path = input_dir / "document.pdf"

        output_path = store.get_output_path(source_path)
        expected = store.output_base / "document.md"

        assert output_path == expected

    def test_nested_subdirectory(self, store, temp_dirs):
        """Nested path should mirror directory structure."""
        input_dir, _ = temp_dirs
        source_path = input_dir / "subdir" / "deep" / "file.pdf"

        output_path = store.get_output_path(source_path)
        expected = store.output_base / "subdir" / "deep" / "file.md"

        assert output_path == expected

    def test_multiple_nested_levels(self, store, temp_dirs):
        """Multiple levels should be preserved."""
        input_dir, _ = temp_dirs
        source_path = input_dir / "a" / "b" / "c" / "d" / "file.pdf"

        output_path = store.get_output_path(source_path)
        expected = store.output_base / "a" / "b" / "c" / "d" / "file.md"

        assert output_path == expected

    def test_different_extensions(self, store, temp_dirs):
        """Different source extensions should become .md."""
        input_dir, _ = temp_dirs

        for ext in [".pdf", ".docx", ".xlsx", ".txt"]:
            source_path = input_dir / f"file{ext}"
            output_path = store.get_output_path(source_path)
            assert output_path.suffix == ".md"
            assert output_path.stem == "file"


class TestSaveAndLoad:
    """Tests for save and load operations."""

    def test_save_creates_markdown_file(self, store, sample_record, sample_content):
        """Save should create a .md file."""
        result = store.save(sample_record, sample_content)

        assert result is True
        output_path = store.get_output_path(sample_record.source_path)
        assert output_path.exists()

    def test_save_creates_metadata_file(self, store, sample_record, sample_content):
        """Save should create a .json metadata file."""
        result = store.save(sample_record, sample_content)

        assert result is True
        meta_path = store.get_output_path(sample_record.source_path).with_suffix(
            ".md.json"
        )
        assert meta_path.exists()

    def test_save_creates_parent_directories(self, store, temp_dirs, sample_content):
        """Save should create parent directories if needed."""
        input_dir, _ = temp_dirs
        source_path = input_dir / "deep" / "nested" / "path" / "file.pdf"

        record = DocumentRecord(
            id="nested_test",
            source_path=source_path,
            output_path=Path("output/deep/nested/path/file.md"),
            title="Nested File",
            content_hash="xyz",
            file_size=100,
            file_mtime=datetime.now(),
        )

        result = store.save(record, sample_content)

        assert result is True
        assert (store.output_base / "deep" / "nested" / "path").exists()

    def test_load_by_id(self, store, sample_record, sample_content):
        """Load should retrieve document by ID."""
        store.save(sample_record, sample_content)

        result = store.load(sample_record.id)

        assert result is not None
        record, content = result
        assert record.id == sample_record.id
        assert content == sample_content

    def test_load_by_source(self, store, sample_record, sample_content):
        """Load by source path should work."""
        store.save(sample_record, sample_content)

        result = store.load_by_source(sample_record.source_path)

        assert result is not None
        record, content = result
        assert record.title == sample_record.title
        assert content == sample_content

    def test_load_nonexistent(self, store):
        """Load should return None for nonexistent documents."""
        result = store.load("nonexistent_id")
        assert result is None

        result = store.load_by_source(Path("/nonexistent/file.pdf"))
        assert result is None

    def test_save_updates_record_output_path(
        self, store, sample_record, sample_content
    ):
        """Save should update record's output_path."""
        store.save(sample_record, sample_content)

        assert sample_record.output_path.exists()


class TestFilenameConflicts:
    """Tests for filename conflict handling."""

    def test_conflict_appends_number(self, store, temp_dirs, sample_content):
        """Conflicting filenames should get numbered suffix."""
        input_dir, _ = temp_dirs

        # Create two records with same source path
        source_path = input_dir / "test.pdf"

        record1 = DocumentRecord(
            id="doc1",
            source_path=source_path,
            output_path=Path("output/test.md"),
            title="Document 1",
            content_hash="hash1",
            file_size=100,
            file_mtime=datetime.now(),
        )

        record2 = DocumentRecord(
            id="doc2",
            source_path=source_path,
            output_path=Path("output/test.md"),
            title="Document 2",
            content_hash="hash2",
            file_size=100,
            file_mtime=datetime.now(),
        )

        store.save(record1, sample_content)
        store.save(record2, sample_content)

        # First file should be at original path
        # Second file should be at test_1.md
        assert (store.output_base / "test.md").exists()
        assert (store.output_base / "test_1.md").exists()

    def test_multiple_conflicts(self, store, temp_dirs, sample_content):
        """Multiple conflicts should increment number."""
        input_dir, _ = temp_dirs
        source_path = input_dir / "multi.pdf"

        for i in range(5):
            record = DocumentRecord(
                id=f"doc{i}",
                source_path=source_path,
                output_path=Path("output/multi.md"),
                title=f"Document {i}",
                content_hash=f"hash{i}",
                file_size=100,
                file_mtime=datetime.now(),
            )
            store.save(record, sample_content)

        # Check all files exist with correct suffixes
        assert (store.output_base / "multi.md").exists()
        for i in range(1, 5):
            assert (store.output_base / f"multi_{i}.md").exists()


class TestNonAsciiFilenames:
    """Tests for non-ASCII filename handling."""

    def test_chinese_filename(self, store, temp_dirs, sample_content):
        """Chinese characters in filename should be handled."""
        input_dir, _ = temp_dirs
        source_path = input_dir / "中文文档.pdf"

        record = DocumentRecord(
            id="chinese_test",
            source_path=source_path,
            output_path=Path("output/中文文档.md"),
            title="中文标题",
            content_hash="hash",
            file_size=100,
            file_mtime=datetime.now(),
        )

        result = store.save(record, sample_content)
        assert result is True

        output_path = store.get_output_path(source_path)
        assert output_path.exists()

    def test_japanese_filename(self, store, temp_dirs, sample_content):
        """Japanese characters in filename should be handled."""
        input_dir, _ = temp_dirs
        source_path = input_dir / "テスト.pdf"

        record = DocumentRecord(
            id="japanese_test",
            source_path=source_path,
            output_path=Path("output/テスト.md"),
            title="テストタイトル",
            content_hash="hash",
            file_size=100,
            file_mtime=datetime.now(),
        )

        result = store.save(record, sample_content)
        assert result is True

    def test_mixed_unicode_content(self, store, sample_record):
        """Unicode content should be preserved correctly."""
        unicode_content = """# Тест Документ

Это тест на русском языке.

## 中文部分

这是一些中文内容。

## Ελληνικά

Αυτή είναι ελληνική γλώσσα.
"""
        result = store.save(sample_record, unicode_content)
        assert result is True

        loaded = store.load(sample_record.id)
        assert loaded is not None
        _, content = loaded
        assert content == unicode_content


class TestExists:
    """Tests for exists operations."""

    def test_exists_true_after_save(self, store, sample_record, sample_content):
        """Exists should return True after saving."""
        store.save(sample_record, sample_content)

        assert store.exists(sample_record.id) is True
        assert store.exists_by_source(sample_record.source_path) is True

    def test_exists_false_before_save(self, store, sample_record):
        """Exists should return False before saving."""
        assert store.exists(sample_record.id) is False
        assert store.exists_by_source(sample_record.source_path) is False

    def test_exists_false_after_delete(self, store, sample_record, sample_content):
        """Exists should return False after deletion."""
        store.save(sample_record, sample_content)
        store.delete(sample_record.id)

        assert store.exists(sample_record.id) is False


class TestDelete:
    """Tests for delete operations."""

    def test_delete_by_id(self, store, sample_record, sample_content):
        """Delete by ID should remove files."""
        store.save(sample_record, sample_content)

        result = store.delete(sample_record.id)

        assert result is True
        output_path = store.get_output_path(sample_record.source_path)
        assert not output_path.exists()

    def test_delete_by_source(self, store, sample_record, sample_content):
        """Delete by source path should remove files."""
        store.save(sample_record, sample_content)

        result = store.delete_by_source(sample_record.source_path)

        assert result is True
        output_path = store.get_output_path(sample_record.source_path)
        assert not output_path.exists()

    def test_delete_nonexistent(self, store):
        """Delete nonexistent document should return False."""
        result = store.delete("nonexistent_id")
        assert result is False

    def test_delete_removes_metadata(self, store, sample_record, sample_content):
        """Delete should also remove metadata file."""
        store.save(sample_record, sample_content)
        meta_path = store.get_output_path(sample_record.source_path).with_suffix(
            ".md.json"
        )

        assert meta_path.exists()

        store.delete(sample_record.id)

        assert not meta_path.exists()


class TestList:
    """Tests for list operations."""

    def test_list_empty(self, store):
        """List should return empty list when no documents."""
        result = store.list()
        assert result == []

    def test_list_single(self, store, sample_record, sample_content):
        """List should return single saved document."""
        store.save(sample_record, sample_content)

        result = store.list()

        assert len(result) == 1
        assert result[0].id == sample_record.id

    def test_list_multiple(self, store, temp_dirs, sample_content):
        """List should return all saved documents."""
        input_dir, _ = temp_dirs

        for i in range(5):
            source_path = input_dir / f"doc{i}.pdf"
            record = DocumentRecord(
                id=f"doc{i}",
                source_path=source_path,
                output_path=Path(f"output/doc{i}.md"),
                title=f"Document {i}",
                content_hash=f"hash{i}",
                file_size=100,
                file_mtime=datetime.now(),
            )
            store.save(record, sample_content)

        result = store.list()

        assert len(result) == 5
        ids = [r.id for r in result]
        for i in range(5):
            assert f"doc{i}" in ids

    def test_list_with_filter(self, store, temp_dirs, sample_content):
        """List should filter by status."""
        input_dir, _ = temp_dirs

        # Create active and archived documents
        for i, status in enumerate(["active", "archived", "active"]):
            source_path = input_dir / f"doc{i}.pdf"
            record = DocumentRecord(
                id=f"doc{i}",
                source_path=source_path,
                output_path=Path(f"output/doc{i}.md"),
                title=f"Document {i}",
                content_hash=f"hash{i}",
                file_size=100,
                file_mtime=datetime.now(),
                status=status,
            )
            store.save(record, sample_content)

        result = store.list(filter={"status": "active"})

        assert len(result) == 2
        for r in result:
            assert r.status == "active"


class TestSaveWithImages:
    """Tests for saving documents with images."""

    def test_save_with_images(self, store, sample_record, sample_content, temp_dirs):
        """Save with images should copy images to images directory."""
        input_dir, _ = temp_dirs

        # Create a test image
        images_dir = input_dir / "test_images"
        images_dir.mkdir()
        test_image = images_dir / "test.png"
        test_image.write_bytes(b"fake png data")

        result = store.save_with_images(sample_record, sample_content, [test_image])

        assert result is True

        # Check image was copied
        output_images_dir = store.get_images_dir(sample_record.source_path)
        copied_image = output_images_dir / "test.png"
        assert copied_image.exists()

    def test_save_with_multiple_images(
        self, store, sample_record, sample_content, temp_dirs
    ):
        """Save with multiple images should copy all images."""
        input_dir, _ = temp_dirs

        # Create multiple test images
        images_dir = input_dir / "test_images"
        images_dir.mkdir()
        images = []
        for i in range(3):
            img = images_dir / f"image{i}.png"
            img.write_bytes(f"fake png data {i}".encode())
            images.append(img)

        result = store.save_with_images(sample_record, sample_content, images)

        assert result is True

        output_images_dir = store.get_images_dir(sample_record.source_path)
        for i in range(3):
            assert (output_images_dir / f"image{i}.png").exists()


class TestMetadataPersistence:
    """Tests for metadata persistence."""

    def test_metadata_preserves_all_fields(self, store, sample_record, sample_content):
        """All DocumentRecord fields should be preserved."""
        sample_record.metadata = {
            "author": "Test Author",
            "date": "2024-01-15",
            "custom_field": "custom_value",
        }
        sample_record.keywords = ["keyword1", "keyword2", "keyword3"]
        sample_record.sections = ["Section A", "Section B"]

        store.save(sample_record, sample_content)

        result = store.load(sample_record.id)
        assert result is not None

        record, _ = result
        assert record.metadata == sample_record.metadata
        assert record.keywords == sample_record.keywords
        assert record.sections == sample_record.sections

    def test_metadata_datetime_preservation(self, store, sample_record, sample_content):
        """Datetime fields should be preserved correctly."""
        now = datetime.now()
        sample_record.file_mtime = now
        sample_record.created_at = now
        sample_record.updated_at = now

        store.save(sample_record, sample_content)

        result = store.load(sample_record.id)
        assert result is not None

        record, _ = result
        # Compare ISO strings to avoid microsecond differences
        assert record.file_mtime.isoformat() == now.isoformat()
        assert record.created_at.isoformat() == now.isoformat()


class TestGetOutputPath:
    """Tests for get_output_path helper method."""

    def test_get_output_path_basic(self, store, temp_dirs):
        """get_output_path should return correct path."""
        input_dir, _ = temp_dirs
        source = input_dir / "file.pdf"

        result = store.get_output_path(source)

        expected = store.output_base / "file.md"
        assert result == expected

    def test_get_images_dir(self, store, temp_dirs):
        """get_images_dir should return correct path."""
        input_dir, _ = temp_dirs
        source = input_dir / "file.pdf"

        result = store.get_images_dir(source)

        expected = store.output_base / "images"
        assert result == expected

    def test_get_images_dir_nested(self, store, temp_dirs):
        """get_images_dir for nested file should be in correct location."""
        input_dir, _ = temp_dirs
        source = input_dir / "sub" / "file.pdf"

        result = store.get_images_dir(source)

        expected = store.output_base / "sub" / "images"
        assert result == expected
