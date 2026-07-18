"""Unit tests for MetadataManager class."""

import json
from pathlib import Path

import pytest

from src.storage.metadata import MetadataManager


@pytest.fixture
def temp_index(tmp_path):
    """Create a temporary index.json file path."""
    return tmp_path / "index.json"


@pytest.fixture
def sample_metadata():
    """Create sample metadata dictionary."""
    return {
        "source_path": Path("/path/to/source.pdf"),
        "output_path": Path("/path/to/output.md"),
        "title": "Test Document",
        "keywords": ["test", "document", "sample"],
        "modified_time": "2024-01-15T10:30:00",
        "content_hash": "abc123def456",
        "convert_time": "2024-01-15T11:00:00",
        "file_size": 1024,
        "converter": "pdf2md",
    }


@pytest.fixture
def manager(temp_index):
    """Create a MetadataManager instance with temporary index."""
    return MetadataManager(index_path=temp_index)


class TestMetadataManagerInit:
    """Test MetadataManager initialization."""

    def test_init_with_path(self, temp_index):
        """Test initialization with custom path."""
        manager = MetadataManager(index_path=temp_index)
        assert manager.index_path == temp_index
        assert manager.get_count() == 0

    def test_init_default_path(self, tmp_path):
        """Test initialization with default path."""
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            manager = MetadataManager()
            assert manager.index_path == tmp_path / "index.json"
        finally:
            os.chdir(original_cwd)

    def test_init_loads_existing_index(self, temp_index, sample_metadata):
        """Test that initialization loads existing index.json."""
        # Create existing index.json with string paths (as JSON would have)
        metadata_copy = sample_metadata.copy()
        metadata_copy["source_path"] = str(sample_metadata["source_path"])
        metadata_copy["output_path"] = str(sample_metadata["output_path"])
        # Use the string representation of Path as the key (consistent with how MetadataManager saves)
        doc_path = Path("/path/to/doc1.pdf")
        test_data = {
            str(doc_path): metadata_copy,
        }
        with temp_index.open("w", encoding="utf-8") as f:
            json.dump(test_data, f)

        manager = MetadataManager(index_path=temp_index)
        assert manager.get_count() == 1
        assert manager.exists(doc_path)

    def test_init_handles_corrupted_index(self, temp_index):
        """Test that initialization handles corrupted index.json gracefully."""
        # Write invalid JSON
        with temp_index.open("w", encoding="utf-8") as f:
            f.write("{ invalid json")

        manager = MetadataManager(index_path=temp_index)
        assert manager.get_count() == 0  # Should start fresh


class TestMetadataManagerSave:
    """Test save method."""

    def test_save_new_metadata(self, manager, sample_metadata):
        """Test saving new metadata."""
        doc_path = Path("/path/to/doc.pdf")
        result = manager.save(doc_path, sample_metadata)

        assert result is True
        assert manager.exists(doc_path)
        assert manager.index_path.exists()

    def test_save_with_string_path(self, manager, sample_metadata):
        """Test saving with string path instead of Path."""
        result = manager.save("/path/to/doc.pdf", sample_metadata)

        assert result is True
        assert manager.exists(Path("/path/to/doc.pdf"))

    def test_save_updates_existing(self, manager, sample_metadata):
        """Test saving updates existing metadata."""
        doc_path = Path("/path/to/doc.pdf")
        manager.save(doc_path, sample_metadata)

        # Update with new metadata
        updated_metadata = sample_metadata.copy()
        updated_metadata["title"] = "Updated Title"
        manager.save(doc_path, updated_metadata)

        loaded = manager.load(doc_path)
        assert loaded["title"] == "Updated Title"
        assert "last_updated" in loaded

    def test_save_creates_parent_dir(self, tmp_path, sample_metadata):
        """Test that save creates parent directories."""
        nested_index = tmp_path / "nested" / "dir" / "index.json"
        manager = MetadataManager(index_path=nested_index)

        result = manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        assert result is True
        assert nested_index.exists()

    def test_save_converts_path_to_string(self, manager, sample_metadata):
        """Test that Path objects are converted to strings for JSON."""
        doc_path = Path("/path/to/doc.pdf")
        manager.save(doc_path, sample_metadata)

        # Read raw JSON to verify paths are strings
        with manager.index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # The key should be the string representation of the path
        metadata = data[str(doc_path)]
        assert isinstance(metadata["source_path"], str)
        assert isinstance(metadata["output_path"], str)


class TestMetadataManagerLoad:
    """Test load method."""

    def test_load_existing_metadata(self, manager, sample_metadata):
        """Test loading existing metadata."""
        doc_path = Path("/path/to/doc.pdf")
        manager.save(doc_path, sample_metadata)

        loaded = manager.load(doc_path)

        assert loaded is not None
        assert loaded["title"] == "Test Document"
        assert loaded["keywords"] == ["test", "document", "sample"]
        assert isinstance(loaded["source_path"], Path)
        assert isinstance(loaded["output_path"], Path)

    def test_load_with_string_path(self, manager, sample_metadata):
        """Test loading with string path."""
        manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        loaded = manager.load("/path/to/doc.pdf")

        assert loaded is not None
        assert loaded["title"] == "Test Document"

    def test_load_nonexistent(self, manager):
        """Test loading metadata that doesn't exist."""
        loaded = manager.load(Path("/nonexistent/path.pdf"))
        assert loaded is None

    def test_load_converts_strings_to_paths(self, manager, sample_metadata):
        """Test that string paths are converted back to Path objects."""
        manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        loaded = manager.load(Path("/path/to/doc.pdf"))

        assert isinstance(loaded["source_path"], Path)
        assert isinstance(loaded["output_path"], Path)
        # Use Path equality instead of string comparison to avoid platform differences
        assert loaded["source_path"] == Path("/path/to/source.pdf")


class TestMetadataManagerExists:
    """Test exists method."""

    def test_exists_true(self, manager, sample_metadata):
        """Test exists returns True for existing metadata."""
        manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        assert manager.exists(Path("/path/to/doc.pdf")) is True

    def test_exists_false(self, manager):
        """Test exists returns False for non-existent metadata."""
        assert manager.exists(Path("/nonexistent/path.pdf")) is False

    def test_exists_with_string_path(self, manager, sample_metadata):
        """Test exists with string path."""
        manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        assert manager.exists("/path/to/doc.pdf") is True


class TestMetadataManagerDelete:
    """Test delete method."""

    def test_delete_existing(self, manager, sample_metadata):
        """Test deleting existing metadata."""
        doc_path = Path("/path/to/doc.pdf")
        manager.save(doc_path, sample_metadata)

        result = manager.delete(doc_path)

        assert result is True
        assert manager.exists(doc_path) is False
        assert manager.get_count() == 0

    def test_delete_nonexistent(self, manager):
        """Test deleting non-existent metadata."""
        result = manager.delete(Path("/nonexistent/path.pdf"))
        assert result is False

    def test_delete_with_string_path(self, manager, sample_metadata):
        """Test delete with string path."""
        manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        result = manager.delete("/path/to/doc.pdf")

        assert result is True
        assert manager.exists(Path("/path/to/doc.pdf")) is False


class TestMetadataManagerListAll:
    """Test list_all method."""

    def test_list_all_empty(self, manager):
        """Test listing when no metadata exists."""
        result = manager.list_all()
        assert result == []

    def test_list_all_multiple(self, manager, sample_metadata):
        """Test listing all metadata entries."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)
        manager.save(Path("/path/doc2.pdf"), sample_metadata)
        manager.save(Path("/path/doc3.pdf"), sample_metadata)

        result = manager.list_all()

        assert len(result) == 3
        assert all("title" in entry for entry in result)

    def test_list_all_converts_paths(self, manager, sample_metadata):
        """Test that list_all converts string paths to Path objects."""
        manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        result = manager.list_all()

        assert len(result) == 1
        assert isinstance(result[0]["source_path"], Path)
        assert isinstance(result[0]["output_path"], Path)


class TestMetadataManagerQuery:
    """Test query method."""

    def test_query_exact_match(self, manager, sample_metadata):
        """Test query with exact match filter."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)
        manager.save(Path("/path/doc2.pdf"), sample_metadata)

        # Modify one
        doc2_metadata = sample_metadata.copy()
        doc2_metadata["title"] = "Different Title"
        manager.save(Path("/path/doc2.pdf"), doc2_metadata)

        results = manager.query({"title": "Test Document"})

        assert len(results) == 1
        assert results[0]["title"] == "Test Document"

    def test_query_contains_string(self, manager, sample_metadata):
        """Test query with contains operator."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        results = manager.query({"title__contains": "Test"})

        assert len(results) == 1

    def test_query_contains_case_insensitive(self, manager, sample_metadata):
        """Test that contains is case-insensitive."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        results = manager.query({"title__contains": "test"})

        assert len(results) == 1

    def test_query_contains_no_match(self, manager, sample_metadata):
        """Test query with contains that doesn't match."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        results = manager.query({"title__contains": "Nonexistent"})

        assert len(results) == 0

    def test_query_greater_than(self, manager, sample_metadata):
        """Test query with greater than operator."""
        sample_metadata["file_size"] = 1024
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        small_meta = sample_metadata.copy()
        small_meta["file_size"] = 512
        manager.save(Path("/path/doc2.pdf"), small_meta)

        results = manager.query({"file_size__gt": "700"})

        assert len(results) == 1
        assert results[0]["file_size"] == 1024

    def test_query_less_than(self, manager, sample_metadata):
        """Test query with less than operator."""
        sample_metadata["file_size"] = 1024
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        small_meta = sample_metadata.copy()
        small_meta["file_size"] = 512
        manager.save(Path("/path/doc2.pdf"), small_meta)

        results = manager.query({"file_size__lt": "700"})

        assert len(results) == 1
        assert results[0]["file_size"] == 512

    def test_query_multiple_filters(self, manager, sample_metadata):
        """Test query with multiple filters."""
        sample_metadata["file_size"] = 1024
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        diff_meta = sample_metadata.copy()
        diff_meta["title"] = "Different"
        diff_meta["file_size"] = 1024
        manager.save(Path("/path/doc2.pdf"), diff_meta)

        results = manager.query({"title": "Test Document", "file_size__gt": "500"})

        assert len(results) == 1
        assert results[0]["title"] == "Test Document"

    def test_query_nonexistent_field(self, manager, sample_metadata):
        """Test query on field that doesn't exist."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        results = manager.query({"nonexistent_field": "value"})

        assert len(results) == 0

    def test_query_converts_paths(self, manager, sample_metadata):
        """Test that query results have Path objects."""
        manager.save(Path("/path/to/doc.pdf"), sample_metadata)

        results = manager.query({"title": "Test Document"})

        assert len(results) == 1
        assert isinstance(results[0]["source_path"], Path)


class TestMetadataManagerGetCount:
    """Test get_count method."""

    def test_get_count_empty(self, manager):
        """Test get_count when empty."""
        assert manager.get_count() == 0

    def test_get_count_multiple(self, manager, sample_metadata):
        """Test get_count with multiple entries."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)
        manager.save(Path("/path/doc2.pdf"), sample_metadata)
        manager.save(Path("/path/doc3.pdf"), sample_metadata)

        assert manager.get_count() == 3

    def test_get_count_after_delete(self, manager, sample_metadata):
        """Test get_count updates after deletion."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)
        manager.save(Path("/path/doc2.pdf"), sample_metadata)
        manager.delete(Path("/path/doc1.pdf"))

        assert manager.get_count() == 1


class TestMetadataManagerClear:
    """Test clear method."""

    def test_clear(self, manager, sample_metadata):
        """Test clearing all metadata."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)
        manager.save(Path("/path/doc2.pdf"), sample_metadata)

        result = manager.clear()

        assert result is True
        assert manager.get_count() == 0


class TestMetadataManagerRebuildIndex:
    """Test rebuild_index method."""

    def test_rebuild_index(self, manager, sample_metadata, tmp_path):
        """Test rebuilding index at new location."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        new_index = tmp_path / "new_index.json"
        result = manager.rebuild_index(new_index)

        assert result is True
        assert new_index.exists()
        assert manager.index_path == new_index

    def test_rebuild_index_preserves_data(self, manager, sample_metadata, tmp_path):
        """Test that rebuild_index preserves metadata."""
        manager.save(Path("/path/doc1.pdf"), sample_metadata)

        new_index = tmp_path / "new_index.json"
        manager.rebuild_index(new_index)

        loaded = manager.load(Path("/path/doc1.pdf"))
        assert loaded["title"] == "Test Document"


class TestMetadataManagerIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, manager):
        """Test complete workflow: save, load, query, delete."""
        metadata1 = {
            "source_path": Path("/path/doc1.pdf"),
            "output_path": Path("/path/doc1.md"),
            "title": "Document One",
            "keywords": ["one", "test"],
            "modified_time": "2024-01-15T10:00:00",
            "content_hash": "hash1",
            "convert_time": "2024-01-15T11:00:00",
            "file_size": 1024,
            "converter": "pdf",
        }

        metadata2 = {
            "source_path": Path("/path/doc2.pdf"),
            "output_path": Path("/path/doc2.md"),
            "title": "Document Two",
            "keywords": ["two", "test"],
            "modified_time": "2024-01-15T10:30:00",
            "content_hash": "hash2",
            "convert_time": "2024-01-15T11:30:00",
            "file_size": 2048,
            "converter": "pdf",
        }

        # Save both
        assert manager.save(Path("/path/doc1.pdf"), metadata1) is True
        assert manager.save(Path("/path/doc2.pdf"), metadata2) is True

        # Verify count
        assert manager.get_count() == 2

        # Query specific
        results = manager.query({"title": "Document One"})
        assert len(results) == 1

        # List all
        all_docs = manager.list_all()
        assert len(all_docs) == 2

        # Delete one
        assert manager.delete(Path("/path/doc1.pdf")) is True
        assert manager.get_count() == 1

    def test_persistence_across_instances(self, temp_index, sample_metadata):
        """Test that metadata persists across manager instances."""
        # First instance saves
        manager1 = MetadataManager(index_path=temp_index)
        manager1.save(Path("/path/doc.pdf"), sample_metadata)

        # Second instance loads
        manager2 = MetadataManager(index_path=temp_index)
        assert manager2.exists(Path("/path/doc.pdf"))
        loaded = manager2.load(Path("/path/doc.pdf"))
        assert loaded is not None
        assert loaded["title"] == "Test Document"
