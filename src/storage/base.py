"""Base classes and data models for storage management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentRecord:
    """Data model for a processed document record."""

    id: str
    source_path: Path
    output_path: Path
    title: str
    content_hash: str
    file_size: int
    file_mtime: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    convert_count: int = 0
    last_convert_time: float = 0.0
    last_converter: str = ""
    status: str = "active"


@dataclass
class SearchHit:
    """Data model for a single search result hit."""

    doc_id: str
    title: str
    score: float
    excerpt: str
    highlights: List[str] = field(default_factory=list)
    source_path: Optional[Path] = None
    section: str = ""
    page: int = -1

    def __post_init__(self):
        """Ensure source_path is a Path object."""
        if self.source_path is not None and not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)


@dataclass
class SearchResult:
    """Data model for search results."""

    hits: List[SearchHit]
    total: int
    query: str
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    offset: int = 0
    limit: int = 10
    has_more: bool = False

    def __post_init__(self):
        """Calculate has_more based on total and offset/limit."""
        self.has_more = (self.offset + self.limit) < self.total


class Storage(ABC):
    """Abstract base class for document storage implementations."""

    @abstractmethod
    def save(self, record: DocumentRecord, content: str) -> bool:
        """
        Save a document record and its content.

        Args:
            record: DocumentRecord to save
            content: Document text content

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def load(self, doc_id: str) -> Optional[Tuple[DocumentRecord, str]]:
        """
        Load a document record and its content by ID.

        Args:
            doc_id: Document ID to load

        Returns:
            Tuple of (DocumentRecord, content) if found, None otherwise
        """
        pass

    @abstractmethod
    def exists(self, doc_id: str) -> bool:
        """
        Check if a document exists.

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def list(self, filter: Optional[Dict] = None) -> List[DocumentRecord]:
        """
        List all documents, optionally filtered.

        Args:
            filter: Optional filter criteria

        Returns:
            List of DocumentRecord objects
        """
        pass


class IndexManager(ABC):
    """Abstract base class for search index management."""

    @abstractmethod
    def add_document(
        self, doc_id: str, title: str, content: str, metadata: Dict
    ) -> bool:
        """
        Add a document to the search index.

        Args:
            doc_id: Document ID
            title: Document title
            content: Document text content
            metadata: Document metadata

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def update_document(
        self, doc_id: str, title: str, content: str, metadata: Dict
    ) -> bool:
        """
        Update a document in the search index.

        Args:
            doc_id: Document ID
            title: Document title
            content: Document text content
            metadata: Document metadata

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the search index.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict] = None,
    ) -> SearchResult:
        """
        Search the index for documents.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            offset: Number of results to skip
            filters: Optional filter criteria

        Returns:
            SearchResult object with matching hits
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        pass
