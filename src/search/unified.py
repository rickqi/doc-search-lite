"""Unified search result models for cross-source result merging.

Provides data models used by HybridSearcher (BM25 + Grep RRF fusion)
and MultiIndexSearcher (fan-out + cross-index merge).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SearchSource(str, Enum):
    """Origin of a search result."""
    BM25 = "bm25"
    GREP = "grep"


@dataclass
class UnifiedSearchResult:
    """Single search result from any source, with normalized scoring.

    Attributes:
        doc_id: Document identifier (may be namespaced for multi-index).
        source_path: Path to the source file (relative or absolute).
        title: Document title or filename.
        snippet: Short text excerpt / match preview.
        highlights: Highlighted query terms (from BM25).
        raw_score: Original relevance score from the source.
        normalized_score: Min-max normalized score in [0, 1].
        rrf_score: Reciprocal Rank Fusion score.
        rank: Final rank after merge (assigned after sorting).
        search_source: Which source produced this result.
        index_name: Name of the index this result came from (multi-index).
        grep_matches: Number of grep matches in this file.
        grep_line_matches: Detailed line-level grep match info.
        retrieval_time: Time taken to retrieve this result (seconds).
    """

    doc_id: str
    source_path: Optional[Path] = None
    title: str = ""
    snippet: str = ""
    highlights: List[str] = field(default_factory=list)
    raw_score: float = 0.0
    normalized_score: float = 0.0
    rrf_score: float = 0.0
    rank: int = 0
    search_source: SearchSource = SearchSource.BM25
    index_name: str = ""
    grep_matches: int = 0
    grep_line_matches: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_time: float = 0.0

    def __post_init__(self):
        """Normalize source_path to Path object."""
        if self.source_path is not None and isinstance(self.source_path, str):
            self.source_path = Path(self.source_path)

    @property
    def score(self) -> float:
        """Compatibility alias — returns raw_score (BM25 relevance, same scale as SearchPreview)."""
        return self.raw_score if self.raw_score else self.rrf_score


@dataclass
class UnifiedSearchResults:
    """Container for merged search results from one or more sources.

    Attributes:
        results: Ordered list of unified results (best first).
        total: Total number of unique results before limit.
        query: The original query string.
        sources_used: List of source names that contributed results.
        execution_time: Total wall-clock time for the search (seconds).
        bm25_count: Number of results contributed by BM25.
        grep_count: Number of results contributed by Grep.
    """

    results: List[UnifiedSearchResult]
    total: int
    query: str
    sources_used: List[str]
    execution_time: float = 0.0
    bm25_count: int = 0
    grep_count: int = 0

    # PaginatedResults compatibility (SearchTool expects these)
    offset: int = 0
    limit: int = 10
    has_more: bool = False
