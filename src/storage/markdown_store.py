"""Markdown storage implementation with mirror directory structure."""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from src.storage.base import DocumentRecord, Storage

logger = logging.getLogger(__name__)


class MarkdownStore(Storage):
    """
    Storage implementation that saves documents as Markdown files.

    Maintains a mirror directory structure where:
    - Source: input/subdir/file.pdf
    - Output: output/input/subdir/file.md
    - Images: output/input/subdir/images/

    Metadata is stored alongside each .md file as a .json file.
    """

    IMAGES_DIR_NAME = "images"
    METADATA_SUFFIX = ".json"

    def __init__(self, input_base: Path, output_base: Path):
        """
        Initialize the MarkdownStore.

        Args:
            input_base: Base directory for source documents
            output_base: Base directory for output Markdown files
        """
        self.input_base = Path(input_base).resolve()
        self.output_base = Path(output_base).resolve()
        self._doc_id_index: dict[str, Path] | None = None  # lazy-built cache
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        self.output_base.mkdir(parents=True, exist_ok=True)

    def _build_doc_id_index(self) -> dict[str, Path]:
        """Build a mapping from doc_id to metadata file path."""
        index = {}
        for meta_path in self.output_base.rglob(f"*{self.METADATA_SUFFIX}"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                doc_id = data.get("id")
                if doc_id:
                    index[doc_id] = meta_path
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return index

    def _get_doc_id_index(self) -> dict[str, Path]:
        """Get the doc_id index, building it lazily if needed."""
        if self._doc_id_index is None:
            self._doc_id_index = self._build_doc_id_index()
        return self._doc_id_index

    def _invalidate_index(self) -> None:
        """Invalidate the doc_id index cache."""
        self._doc_id_index = None

    def _get_relative_source_path(self, source_path: Path) -> Path:
        """
        Get the relative path from input_base to source_path.

        Args:
            source_path: Absolute or relative path to source file

        Returns:
            Relative path from input_base
        """
        source_path = Path(source_path).resolve()
        try:
            return source_path.relative_to(self.input_base)
        except ValueError:
            # source_path is not under input_base, use just the filename
            return Path(source_path.name)

    def _get_output_md_path(self, source_path: Path) -> Path:
        """
        Get the output Markdown file path for a source file.

        Args:
            source_path: Path to source file

        Returns:
            Path to output .md file
        """
        rel_path = self._get_relative_source_path(source_path)
        # Change extension to .md
        md_path = rel_path.with_suffix(".md")
        return self.output_base / md_path

    def _get_output_metadata_path(self, source_path: Path) -> Path:
        """
        Get the output metadata file path for a source file.

        Args:
            source_path: Path to source file

        Returns:
            Path to output .json metadata file
        """
        md_path = self._get_output_md_path(source_path)
        return md_path.with_suffix(md_path.suffix + self.METADATA_SUFFIX)

    def _get_output_images_dir(self, source_path: Path) -> Path:
        """
        Get the output images directory for a source file.

        Args:
            source_path: Path to source file

        Returns:
            Path to images directory
        """
        md_path = self._get_output_md_path(source_path)
        return md_path.parent / self.IMAGES_DIR_NAME

    def _generate_doc_id(self, source_path: Path) -> str:
        """
        Generate a unique document ID based on source path.

        Args:
            source_path: Path to source file

        Returns:
            Unique document ID string
        """
        rel_path = self._get_relative_source_path(source_path)
        # Use hash of the relative path string for uniqueness (use as_posix for cross-platform)
        path_str = rel_path.as_posix()
        return hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:16]

    def _resolve_filename_conflict(self, target_path: Path) -> Path:
        """
        Resolve filename conflicts by appending a number.

        Args:
            target_path: Target file path that may conflict

        Returns:
            Path with conflict resolved (may be same as input)
        """
        if not target_path.exists():
            return target_path

        parent = target_path.parent
        stem = target_path.stem
        suffix = target_path.suffix

        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
            # Safety limit
            if counter > 10000:
                raise RuntimeError(f"Too many filename conflicts for {target_path}")

    def _serialize_record(self, record: DocumentRecord) -> dict:
        """
        Serialize DocumentRecord to a JSON-compatible dictionary.

        Args:
            record: DocumentRecord to serialize

        Returns:
            JSON-compatible dictionary
        """
        return {
            "id": record.id,
            "source_path": str(record.source_path),
            "output_path": str(record.output_path),
            "title": record.title,
            "content_hash": record.content_hash,
            "file_size": record.file_size,
            "file_mtime": record.file_mtime.isoformat(),
            "metadata": record.metadata,
            "keywords": record.keywords,
            "sections": record.sections,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "convert_count": record.convert_count,
            "last_convert_time": record.last_convert_time,
            "last_converter": record.last_converter,
            "status": record.status,
        }

    def _deserialize_record(self, data: dict) -> DocumentRecord:
        """
        Deserialize dictionary to DocumentRecord.

        Args:
            data: Dictionary from JSON

        Returns:
            DocumentRecord instance
        """
        return DocumentRecord(
            id=data["id"],
            source_path=Path(data["source_path"]),
            output_path=Path(data["output_path"]),
            title=data["title"],
            content_hash=data["content_hash"],
            file_size=data["file_size"],
            file_mtime=datetime.fromisoformat(data["file_mtime"]),
            metadata=data.get("metadata", {}),
            keywords=data.get("keywords", []),
            sections=data.get("sections", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            convert_count=data.get("convert_count", 0),
            last_convert_time=data.get("last_convert_time", 0.0),
            last_converter=data.get("last_converter", ""),
            status=data.get("status", "active"),
        )

    def save(self, record: DocumentRecord, content: str) -> bool:
        """
        Save a document record and its Markdown content.

        Args:
            record: DocumentRecord to save
            content: Markdown text content

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get output paths
            md_path = self._get_output_md_path(record.source_path)
            meta_path = self._get_output_metadata_path(record.source_path)

            # Handle filename conflict
            if md_path.exists():
                md_path = self._resolve_filename_conflict(md_path)
                meta_path = md_path.with_suffix(md_path.suffix + self.METADATA_SUFFIX)

            # Create parent directories
            md_path.parent.mkdir(parents=True, exist_ok=True)

            # Write Markdown content (handle non-ASCII)
            md_path.write_text(content, encoding="utf-8")

            # Update record with actual output path
            record.output_path = md_path
            record.updated_at = datetime.now()

            # Write metadata
            metadata = self._serialize_record(record)
            metadata = self._sanitize_for_json(metadata)
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Invalidate cache after save
            self._invalidate_index()
            return True
        except Exception as e:
            logger.error("Error saving document %s: %s", record.source_path, e)
            return False

    def save_with_images(
        self, record: DocumentRecord, content: str, images: list[Path]
    ) -> bool:
        """
        Save a document record with its Markdown content and images.

        Args:
            record: DocumentRecord to save
            content: Markdown text content
            images: List of image file paths to copy

        Returns:
            True if successful, False otherwise
        """
        try:
            # First save the markdown and metadata
            if not self.save(record, content):
                return False

            # Copy images if any
            if images:
                images_dir = self._get_output_images_dir(record.source_path)
                images_dir.mkdir(parents=True, exist_ok=True)

                for img_path in images:
                    img_path = Path(img_path)
                    if img_path.exists():
                        dest_path = images_dir / img_path.name
                        # Handle image filename conflict
                        if dest_path.exists():
                            dest_path = self._resolve_filename_conflict(dest_path)
                        shutil.copy2(img_path, dest_path)

            return True
        except Exception as e:
            logger.error("Error saving document with images %s: %s", record.source_path, e)
            return False

    def load(self, doc_id: str) -> tuple[DocumentRecord, str] | None:
        """
        Load a document record and its content by ID.

        Args:
            doc_id: Document ID to load

        Returns:
            Tuple of (DocumentRecord, content) if found, None otherwise
        """
        idx = self._get_doc_id_index()
        meta_path = idx.get(doc_id)
        if meta_path is None or not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            record = self._deserialize_record(data)
            md_path = meta_path.with_suffix("")  # Remove .json
            if md_path.exists():
                content = md_path.read_text(encoding="utf-8")
                return (record, content)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def load_by_source(self, source_path: Path) -> tuple[DocumentRecord, str] | None:
        """
        Load a document record and its content by source path.

        Args:
            source_path: Path to source file

        Returns:
            Tuple of (DocumentRecord, content) if found, None otherwise
        """
        md_path = self._get_output_md_path(source_path)
        meta_path = self._get_output_metadata_path(source_path)

        if not md_path.exists() or not meta_path.exists():
            return None

        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            record = self._deserialize_record(data)
            content = md_path.read_text(encoding="utf-8")
            return (record, content)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def exists(self, doc_id: str) -> bool:
        """
        Check if a document exists by ID.

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        idx = self._get_doc_id_index()
        meta_path = idx.get(doc_id)
        return meta_path is not None and meta_path.exists()

    def exists_by_source(self, source_path: Path) -> bool:
        """
        Check if a document exists by source path.

        Args:
            source_path: Path to source file

        Returns:
            True if document exists, False otherwise
        """
        md_path = self._get_output_md_path(source_path)
        return md_path.exists()

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        idx = self._get_doc_id_index()
        meta_path = idx.get(doc_id)
        if meta_path is None or not meta_path.exists():
            return False
        try:
            md_path = meta_path.with_suffix("")
            if md_path.exists():
                md_path.unlink()
            meta_path.unlink()
            images_dir = md_path.parent / self.IMAGES_DIR_NAME
            if images_dir.exists() and self._is_dir_empty(images_dir):
                shutil.rmtree(images_dir)
            # Invalidate cache after deletion
            self._invalidate_index()
            return True
        except OSError:
            return False

    def delete_by_source(self, source_path: Path) -> bool:
        """
        Delete a document by source path.

        Args:
            source_path: Path to source file

        Returns:
            True if successful, False otherwise
        """
        result = self.load_by_source(source_path)
        if result is None:
            return False

        record, _ = result
        return self.delete(record.id)

    def _is_dir_empty(self, path: Path) -> bool:
        """Check if a directory is empty."""
        return not any(path.iterdir())

    def list(self, filter: dict | None = None) -> list[DocumentRecord]:
        """
        List all documents, optionally filtered.

        Args:
            filter: Optional dict of field→value criteria (matches doc_type, tags, source_path)

        Returns:
            List of DocumentRecord objects
        """
        records = []

        for meta_path in self.output_base.rglob(f"*{self.METADATA_SUFFIX}"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                record = self._deserialize_record(data)

                # Apply filter if provided
                if filter and not self._matches_filter(record, filter):
                    continue

                records.append(record)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        return records

    def _matches_filter(self, record: DocumentRecord, filter: dict) -> bool:
        """
        Check if a record matches filter criteria.

        Args:
            record: DocumentRecord to check
            filter: Filter criteria dictionary

        Returns:
            True if record matches filter, False otherwise
        """
        for key, value in filter.items():
            if hasattr(record, key):
                record_value = getattr(record, key)
                if record_value != value:
                    return False
            elif key in record.metadata:
                if record.metadata[key] != value:
                    return False
            else:
                return False

        return True

    @staticmethod
    def _sanitize_for_json(obj):
        """Recursively sanitize values for JSON serialization."""
        if isinstance(obj, dict):
            return {k: MarkdownStore._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [MarkdownStore._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                return obj.decode("latin-1")
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def get_output_path(self, source_path: Path) -> Path:
        """
        Get the expected output path for a source file.

        Args:
            source_path: Path to source file

        Returns:
            Expected output path for the Markdown file
        """
        return self._get_output_md_path(source_path)

    def get_images_dir(self, source_path: Path) -> Path:
        """
        Get the images directory for a source file.

        Args:
            source_path: Path to source file

        Returns:
            Path to images directory
        """
        return self._get_output_images_dir(source_path)
