"""File watcher for incremental update detection using hash and modification time."""

from dataclasses import dataclass, field
from pathlib import Path

from src.storage.metadata import MetadataManager
from src.utils.hash import calculate_hash


@dataclass
class ChangeSet:
    """Represents the result of change detection.

    Attributes:
        added: List of newly added file paths
        modified: List of modified file paths
        deleted: List of deleted file paths
        unchanged: List of unchanged file paths
    """

    added: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes detected.

        Returns:
            True if there are any added, modified, or deleted files
        """
        return bool(self.added or self.modified or self.deleted)

    @property
    def total_changes(self) -> int:
        """Get total number of changes.

        Returns:
            Number of added + modified + deleted files
        """
        return len(self.added) + len(self.modified) + len(self.deleted)

    def __repr__(self) -> str:
        """Return string representation of ChangeSet."""
        return (
            f"ChangeSet(added={len(self.added)}, modified={len(self.modified)}, "
            f"deleted={len(self.deleted)}, unchanged={len(self.unchanged)})"
        )


class FileWatcher:
    """Watch for file changes using modification time and content hash.

    This class detects file changes by comparing files on disk with stored
    metadata. It uses modification time for quick checks and content hash
    for accurate change detection.
    """

    def __init__(
        self,
        hash_algorithm: str = "sha256",
        use_mtime_check: bool = True,
        use_hash_check: bool = True,
    ):
        """Initialize FileWatcher.

        Args:
            hash_algorithm: Hash algorithm to use (default: "sha256")
            use_mtime_check: Whether to use modification time for quick check
            use_hash_check: Whether to use content hash for accurate check
        """
        self.hash_algorithm = hash_algorithm
        self.use_mtime_check = use_mtime_check
        self.use_hash_check = use_hash_check

    def detect_changes(
        self,
        source_dir: Path,
        metadata_manager: MetadataManager,
        extensions: set[str] | None = None,
        recursive: bool = True,
    ) -> ChangeSet:
        """Detect file changes in source directory.

        Args:
            source_dir: Directory to scan for files
            metadata_manager: MetadataManager instance for checking stored metadata
            extensions: Optional set of file extensions to include (e.g., {'.pdf', '.docx'})
                       If None, all files are included
            recursive: Whether to scan subdirectories recursively

        Returns:
            ChangeSet with lists of added, modified, deleted, and unchanged files
        """
        source_dir = Path(source_dir)

        # Get all files currently on disk
        if recursive:
            disk_files = self._scan_directory_recursive(source_dir, extensions)
        else:
            disk_files = self._scan_directory_flat(source_dir, extensions)

        # Get all files in metadata
        metadata_files = self._get_metadata_files(metadata_manager, source_dir)

        # Categorize files
        change_set = ChangeSet()

        for file_path in disk_files:
            if file_path not in metadata_files:
                # New file not in metadata
                change_set.added.append(file_path)
            else:
                # File exists in metadata, check if modified
                if self._is_file_modified(file_path, metadata_manager):
                    change_set.modified.append(file_path)
                else:
                    change_set.unchanged.append(file_path)

        # Check for deleted files (in metadata but not on disk)
        for file_path in metadata_files:
            if file_path not in disk_files:
                change_set.deleted.append(file_path)

        return change_set

    def _scan_directory_recursive(
        self, directory: Path, extensions: set[str] | None
    ) -> set[Path]:
        """Scan directory recursively for files.

        Args:
            directory: Directory to scan
            extensions: Optional set of extensions to filter

        Returns:
            Set of file paths found
        """
        files: set[Path] = set()

        if not directory.exists():
            return files

        for path in directory.rglob("*"):
            if path.is_file() and (extensions is None or path.suffix.lower() in extensions):
                files.add(path.resolve())

        return files

    def _scan_directory_flat(
        self, directory: Path, extensions: set[str] | None
    ) -> set[Path]:
        """Scan directory non-recursively for files.

        Args:
            directory: Directory to scan
            extensions: Optional set of extensions to filter

        Returns:
            Set of file paths found
        """
        files: set[Path] = set()

        if not directory.exists():
            return files

        for path in directory.iterdir():
            if path.is_file() and (extensions is None or path.suffix.lower() in extensions):
                files.add(path.resolve())

        return files

    def _get_metadata_files(
        self, metadata_manager: MetadataManager, source_dir: Path
    ) -> set[Path]:
        """Get all file paths stored in metadata.

        Args:
            metadata_manager: MetadataManager instance
            source_dir: Source directory to filter by

        Returns:
            Set of file paths from metadata that are under source_dir
        """
        files: set[Path] = set()
        source_dir = source_dir.resolve()

        for metadata in metadata_manager.list_all():
            source_path = metadata.get("source_path")
            if source_path:
                resolved_path = Path(source_path).resolve()
                # Only include files under the source directory
                try:
                    resolved_path.relative_to(source_dir)
                    files.add(resolved_path)
                except ValueError:
                    # File is not under source_dir
                    pass

        return files

    def _is_file_modified(
        self, file_path: Path, metadata_manager: MetadataManager
    ) -> bool:
        """Check if a file has been modified.

        Uses modification time for quick check and content hash for accurate check.

        Args:
            file_path: Path to the file
            metadata_manager: MetadataManager instance

        Returns:
            True if file is modified, False otherwise
        """
        metadata = metadata_manager.load(file_path)
        if metadata is None:
            return True

        # Quick check using modification time
        if self.use_mtime_check:
            stored_mtime = metadata.get("modified_time")
            if stored_mtime is not None:
                try:
                    current_mtime = file_path.stat().st_mtime
                    if float(current_mtime) != float(stored_mtime):
                        # Mtime changed, might be modified
                        # Fall through to hash check if enabled
                        if not self.use_hash_check:
                            return True
                except (OSError, ValueError):
                    pass

        # Accurate check using content hash
        if self.use_hash_check:
            stored_hash = metadata.get("content_hash")
            if stored_hash:
                try:
                    current_hash = calculate_hash(file_path, self.hash_algorithm)
                    return current_hash != stored_hash
                except (OSError, FileNotFoundError):
                    return True

        return False

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate hash for a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal hash string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        return calculate_hash(file_path, self.hash_algorithm)

    def get_file_mtime(self, file_path: Path) -> float:
        """Get modification time for a file.

        Args:
            file_path: Path to the file

        Returns:
            Modification time as float (Unix timestamp)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.stat().st_mtime
