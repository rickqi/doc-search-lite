"""JSON-based metadata manager for document metadata storage."""

import contextlib
import json
from datetime import datetime
from pathlib import Path


class MetadataManager:
    """Manages document metadata in JSON format with index.json file.

    Metadata entries support the following conventional fields:
        - tags: List[str] — Domain tags extracted by TagExtractor (e.g. ["消保审查", "产品条款"])
        - doc_type: str — Document type classification (e.g. "regulation", "insurance_product")
        - keywords: List[str] — Key terms extracted from the document
    """

    def __init__(self, index_path: Path | None = None):
        """
        Initialize the metadata manager.

        Args:
            index_path: Path to the index.json file. If None, uses current dir/index.json
        """
        if index_path is None:
            index_path = Path.cwd() / "index.json"

        self.index_path = Path(index_path)
        self._metadata: dict[str, dict] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load metadata from index.json file."""
        if self.index_path.exists():
            try:
                with self.index_path.open("r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            except (OSError, json.JSONDecodeError):
                # If file is corrupted, start with empty metadata
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_index(self) -> bool:
        """
        Save metadata to index.json file.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            with self.index_path.open("w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)
            return True
        except OSError:
            return False

    def save(self, path: Path, metadata: dict) -> bool:
        """
        Save metadata for a document path.

        Args:
            path: Original file path (used as key)
            metadata: Dictionary containing metadata fields

        Returns:
            True if successful, False otherwise
        """
        if not isinstance(path, Path):
            path = Path(path)

        key = str(path)

        # Convert Path objects to strings for JSON serialization
        metadata_copy = metadata.copy()
        for field in ["source_path", "output_path"]:
            if field in metadata_copy and isinstance(metadata_copy[field], Path):
                metadata_copy[field] = str(metadata_copy[field])

        # Add/modify timestamp
        metadata_copy["last_updated"] = datetime.now().isoformat()

        self._metadata[key] = metadata_copy
        return self._save_index()

    def load(self, path: Path) -> dict | None:
        """
        Load metadata for a document path.

        Args:
            path: Original file path

        Returns:
            Metadata dictionary if found, None otherwise
        """
        if not isinstance(path, Path):
            path = Path(path)

        key = str(path)
        metadata = self._metadata.get(key)

        if metadata is None:
            return None

        # Convert string paths back to Path objects
        metadata_copy = metadata.copy()
        for field in ["source_path", "output_path"]:
            if field in metadata_copy and isinstance(metadata_copy[field], str):
                metadata_copy[field] = Path(metadata_copy[field])

        return metadata_copy

    def exists(self, path: Path) -> bool:
        """
        Check if metadata exists for a document path.

        Args:
            path: Original file path

        Returns:
            True if metadata exists, False otherwise
        """
        if not isinstance(path, Path):
            path = Path(path)

        return str(path) in self._metadata

    def delete(self, path: Path) -> bool:
        """
        Delete metadata for a document path.

        Args:
            path: Original file path

        Returns:
            True if deleted, False if not found
        """
        if not isinstance(path, Path):
            path = Path(path)

        key = str(path)

        if key in self._metadata:
            del self._metadata[key]
            return self._save_index()

        return False

    def list_all(self) -> list[dict]:
        """
        List all metadata entries.

        Returns:
            List of metadata dictionaries
        """
        result = []

        for metadata in self._metadata.values():
            # Convert string paths back to Path objects
            metadata_copy = metadata.copy()
            for field in ["source_path", "output_path"]:
                if field in metadata_copy and isinstance(metadata_copy[field], str):
                    metadata_copy[field] = Path(metadata_copy[field])
            result.append(metadata_copy)

        return result

    def query(self, filters: dict) -> list[dict]:
        """
        Query metadata entries based on filter criteria.

        Args:
            filters: Dictionary of filter criteria
                     Supported operators:
                     - Exact match: {"field": "value"}
                     - Contains (string): {"field__contains": "substring"}
                     - Greater than: {"field__gt": "value"}
                     - Less than: {"field__lt": "value"}

        Returns:
            List of matching metadata dictionaries
        """
        results = []

        for metadata in self._metadata.values():
            if self._matches_filters(metadata, filters):
                # Convert string paths back to Path objects
                metadata_copy = metadata.copy()
                for field in ["source_path", "output_path"]:
                    if field in metadata_copy and isinstance(metadata_copy[field], str):
                        metadata_copy[field] = Path(metadata_copy[field])
                results.append(metadata_copy)

        return results

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        """
        Check if metadata matches all filter criteria.

        Args:
            metadata: Metadata dictionary to check
            filters: Filter criteria

        Returns:
            True if matches all filters, False otherwise
        """
        for filter_key, filter_value in filters.items():
            # Check for operators
            if "__contains" in filter_key:
                field = filter_key.split("__contains")[0]
                if field not in metadata:
                    return False
                if (
                    not isinstance(metadata[field], str)
                    or filter_value.lower() not in metadata[field].lower()
                ):
                    return False
            elif "__gt" in filter_key:
                field = filter_key.split("__gt")[0]
                if field not in metadata:
                    return False
                try:
                    if float(metadata[field]) <= float(filter_value):
                        return False
                except (ValueError, TypeError):
                    return False
            elif "__lt" in filter_key:
                field = filter_key.split("__lt")[0]
                if field not in metadata:
                    return False
                try:
                    if float(metadata[field]) >= float(filter_value):
                        return False
                except (ValueError, TypeError):
                    return False
            else:
                # Exact match
                if filter_key not in metadata:
                    return False
                if metadata[filter_key] != filter_value:
                    return False

        return True

    def get_count(self) -> int:
        """
        Get total count of metadata entries.

        Returns:
            Number of entries
        """
        return len(self._metadata)

    def clear(self) -> bool:
        """
        Clear all metadata entries.

        Returns:
            True if successful, False otherwise
        """
        self._metadata.clear()
        return self._save_index()

    def query_by_tags(self, tags: list[str]) -> list[dict]:
        """Query metadata entries that contain any of the specified tags.

        Args:
            tags: List of tag strings to match against metadata "tags" field.

        Returns:
            List of matching metadata dictionaries.
        """
        results = []
        tag_set = set(t.lower() for t in tags) if tags else set()

        for metadata in self._metadata.values():
            entry_tags = metadata.get("tags", [])
            if not isinstance(entry_tags, list):
                continue
            entry_tag_set = set(t.lower() for t in entry_tags)
            if tag_set and entry_tag_set & tag_set:
                # Convert string paths back to Path objects
                metadata_copy = metadata.copy()
                for field in ["source_path", "output_path"]:
                    if field in metadata_copy and isinstance(metadata_copy[field], str):
                        metadata_copy[field] = Path(metadata_copy[field])
                results.append(metadata_copy)

        return results

    def rebuild_index(self, new_index_path: Path) -> bool:
        """
        Rebuild index file at a new location.

        Args:
            new_index_path: New path for index.json

        Returns:
            True if successful, False otherwise
        """
        old_path = self.index_path
        self.index_path = Path(new_index_path)
        success = self._save_index()

        if success and old_path != self.index_path:
            # Optionally remove old file
            with contextlib.suppress(FileNotFoundError):
                old_path.unlink()

        return success
