"""MultiIndexSearcher — Fan-out search across multiple Tantivy indexes.

Architecture:
    Fan-out: ThreadPoolExecutor(N) — parallel create_searcher + search per index
    Normalization: Min-max normalize scores within each index to [0, 1]
    Merge: RRF across all indexes
    Namespacing: doc_ids become ``"{index_name}::{original_doc_id}"``
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.search.bm25_search import BM25Searcher, PaginatedResults, create_searcher
from src.search.query_router import QueryRouter
from src.search.unified import SearchSource, UnifiedSearchResult, UnifiedSearchResults


class SearcherPool:
    """Cache BM25Searcher instances to avoid repeated jieba/Tantivy loading.

    Class-level singleton pool shared across all MultiIndexSearcher instances.
    Uses TTL-based expiry and LRU eviction when at capacity.
    """

    _instances: dict[str, BM25Searcher] = {}
    _timestamps: dict[str, float] = {}
    _ttl: float = 600.0  # 10 min TTL
    _max_size: int = 16

    @classmethod
    def get(
        cls,
        index_path: Path,
        use_jieba: bool = True,
        readonly: bool = True,
    ) -> BM25Searcher:
        """Get a cached BM25Searcher or create a new one.

        Args:
            index_path: Path to the Tantivy index directory.
            use_jieba: Whether to use jieba for Chinese tokenization.
            readonly: If True, skip IndexWriter creation.

        Returns:
            Cached or newly created BM25Searcher instance.
        """
        key = str(Path(index_path).resolve())
        now = time.time()

        # Return cached instance if still valid
        if key in cls._instances:
            if now - cls._timestamps.get(key, 0) < cls._ttl:
                # Update access time for LRU
                cls._timestamps[key] = now
                return cls._instances[key]
            # TTL expired — remove stale entry
            del cls._instances[key]
            del cls._timestamps[key]

        # Evict oldest entry if at capacity
        if len(cls._instances) >= cls._max_size:
            oldest_key = min(cls._timestamps, key=cls._timestamps.get)  # type: ignore[arg-type]
            del cls._instances[oldest_key]
            del cls._timestamps[oldest_key]

        # Create and cache new searcher
        searcher = create_searcher(
            index_path=index_path, use_jieba=use_jieba, readonly=readonly
        )
        cls._instances[key] = searcher
        cls._timestamps[key] = now
        return searcher

    @classmethod
    def evict(cls, index_path: Path) -> bool:
        """Remove a specific cached searcher by index path.

        Args:
            index_path: Path to the index directory to evict.

        Returns:
            True if an entry was removed, False if not found.
        """
        key = str(Path(index_path).resolve())
        if key in cls._instances:
            del cls._instances[key]
            del cls._timestamps[key]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Remove all cached searchers."""
        cls._instances.clear()
        cls._timestamps.clear()

    @classmethod
    def size(cls) -> int:
        """Return the number of cached searchers."""
        return len(cls._instances)


class MultiIndexSearcher:
    """Search across multiple Tantivy indexes with cross-index RRF merge.

    Args:
        index_paths: List of paths to Tantivy index directories.
        query_router: Optional QueryRouter to filter indexes before searching.
            When provided, only indexes matching the query are searched.
            When None (default), all indexes are searched as before.
    """

    RRF_K: int = 60

    def __init__(
        self,
        index_paths: list[Path],
        query_router: QueryRouter | None = None,
    ):
        self._index_paths = [Path(p) for p in index_paths]
        self._router = query_router

    # ── Public API ─────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        per_index_limit: int = 20,
        offset: int = 0,
    ) -> UnifiedSearchResults:
        """Execute parallel search across all indexes and RRF-merge.

        Args:
            query: Search query string.
            limit: Maximum final results to return.
            per_index_limit: Results to request from each individual index.

        Returns:
            UnifiedSearchResults with cross-index RRF-merged results.
        """
        start_time = time.time()

        # Determine which indexes to search via optional router
        target_paths = self._resolve_target_indexes(query)
        index_results: list[tuple[str, list[UnifiedSearchResult]]] = []

        with ThreadPoolExecutor(max_workers=len(target_paths) or 1) as pool:
            futures = {}
            for idx_path in target_paths:
                index_name = idx_path.parent.name if idx_path.name == "index" else idx_path.name
                futures[
                    pool.submit(self._search_single_index, idx_path, query, per_index_limit, index_name)
                ] = index_name

            for future in as_completed(futures):
                index_name = futures[future]
                try:
                    results = future.result()
                    if results:
                        index_results.append((index_name, results))
                except Exception:
                    # Silently skip indexes that fail (e.g., corrupt, locked)
                    pass

        # Normalize scores within each index
        for _name, results in index_results:
            self._minmax_normalize(results)

        # RRF merge across indexes
        merged = self._cross_index_rrf(index_results)

        # Sort, limit, rank
        merged.sort(key=lambda r: r.rrf_score, reverse=True)
        total_unique = len(merged)
        has_more = total_unique > limit + offset
        merged = merged[offset:offset + limit]
        for rank, r in enumerate(merged, offset + 1):
            r.rank = rank

        elapsed = time.time() - start_time

        sources_used = [name for name, _ in index_results]

        return UnifiedSearchResults(
            results=merged,
            total=len({r.doc_id for _, results in index_results for r in results}),
            query=query,
            sources_used=sources_used,
            execution_time=elapsed,
            bm25_count=sum(len(results) for _, results in index_results),
            offset=offset,
            limit=limit,
            has_more=has_more,
        )

    # ── Document retrieval ──────────────────────────────────────

    def get_full_content(self, doc_id: str):
        """Get full document content by doc_id, stripping multi-index prefix.

        Multi-index doc_ids are namespaced as ``{index_name}::{original_doc_id}``.
        This method strips the prefix and delegates to the appropriate
        single-index searcher via SearcherPool.

        Args:
            doc_id: Namespaced doc_id (e.g. ``"L3_诊疗指南::1839047957c89d7f"``).

        Returns:
            FullSearchResult with complete content, or None if not found.
        """
        if "::" not in doc_id:
            return None

        index_name, original_id = doc_id.split("::", 1)

        # Find the matching index path
        for idx_path in self._index_paths:
            name = idx_path.parent.name if idx_path.name == "index" else idx_path.name
            if name == index_name:
                try:
                    searcher = SearcherPool.get(index_path=idx_path, use_jieba=True, readonly=True)
                    return searcher.get_full_content(original_id)
                except Exception:
                    pass

        return None

    # ── Internal helpers ───────────────────────────────────────

    def _resolve_target_indexes(self, query: str) -> list[Path]:
        """Determine which indexes to search based on optional router.

        Args:
            query: Search query string.

        Returns:
            List of index paths to search. Falls back to all indexes
            when no router is set or the router returns no matches.
        """
        if not self._router:
            return self._index_paths

        # Map index paths to string keys for router matching
        # Derive index name consistently with search() logic
        path_to_name: dict[str, Path] = {}
        for idx_path in self._index_paths:
            name = idx_path.parent.name if idx_path.name == "index" else idx_path.name
            path_to_name[name] = idx_path

        # Route using index names as keys
        matched_names = self._router.route(query)

        if not matched_names:
            return self._index_paths

        # Resolve matched names back to paths; fallback to all if none resolve
        resolved = []
        for name in matched_names:
            if name in path_to_name:
                resolved.append(path_to_name[name])

        return resolved if resolved else self._index_paths

    @staticmethod
    def _search_single_index(
        index_path: Path,
        query: str,
        limit: int,
        index_name: str,
    ) -> list[UnifiedSearchResult]:
        """Search a single index and return UnifiedSearchResult list."""
        searcher = SearcherPool.get(index_path=index_path, use_jieba=True, readonly=True)
        paginated: PaginatedResults = searcher.search(query, limit=limit)

        results: list[UnifiedSearchResult] = []
        for preview in paginated.results:
            results.append(
                UnifiedSearchResult(
                    doc_id=f"{index_name}::{preview.doc_id}",
                    source_path=preview.source_path,
                    title=preview.title,
                    snippet=preview.snippet,
                    highlights=preview.highlights,
                    raw_score=preview.score,
                    search_source=SearchSource.BM25,
                    index_name=index_name,
                    retrieval_time=paginated.execution_time,
                )
            )
        return results

    @staticmethod
    def _minmax_normalize(results: list[UnifiedSearchResult]) -> None:
        """Min-max normalize raw_score to [0, 1] in-place."""
        if not results:
            return
        scores = [r.raw_score for r in results]
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            for r in results:
                r.normalized_score = 1.0
        else:
            for r in results:
                r.normalized_score = (r.raw_score - min_s) / (max_s - min_s)

    def _cross_index_rrf(
        self,
        index_results: list[tuple[str, list[UnifiedSearchResult]]],
    ) -> list[UnifiedSearchResult]:
        """RRF merge results from multiple indexes.

        Uses (index_name, original doc_id deduped by source_path) as key.
        """
        key_to_result: dict[str, UnifiedSearchResult] = {}
        rrf_scores: dict[str, float] = {}

        for index_name, results in index_results:
            for rank, r in enumerate(results, 1):
                # Dedup key: index_name + source_path or doc_id
                path_str = str(r.source_path) if r.source_path else r.doc_id
                key = f"{index_name}::{path_str}"

                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank)

                if key not in key_to_result:
                    key_to_result[key] = r
                else:
                    # Enrich with additional data
                    existing = key_to_result[key]
                    if not existing.title and r.title:
                        existing.title = r.title
                    if not existing.snippet and r.snippet:
                        existing.snippet = r.snippet

        for key, r in key_to_result.items():
            r.rrf_score = rrf_scores[key]

        return list(key_to_result.values())
