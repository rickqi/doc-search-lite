"""HybridSearcher — Parallel BM25 + Grep with Reciprocal Rank Fusion.

Architecture:
    Phase 1: ThreadPoolExecutor(2) — parallel BM25 + Grep
    Phase 2: Convert results to UnifiedSearchResult
    Phase 3: RRF merge (dedup by source_path)
    Phase 4: Sort, limit, assign ranks
"""

import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from src.agent.tool_types import ToolResult
from src.search.bm25_search import BM25Searcher, PaginatedResults
from src.search.unified import SearchSource, UnifiedSearchResult, UnifiedSearchResults

# ── Search Profile System ───────────────────────────────────────
# Weight presets for different document types.  Each profile defines
# bm25/grep RRF weights and a recommended title_boost for BM25 field
# boosting.  Profiles are opt-in — the default ("general") preserves
# the historic bm25=1.0, grep=0.5 weights.

PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "legal": {"bm25": 1.0, "grep": 0.3, "title_boost": 2.0},
    "technical": {"bm25": 1.0, "grep": 0.5, "title_boost": 1.5},
    "faq": {"bm25": 0.6, "grep": 0.8, "title_boost": 1.0},
    "general": {"bm25": 1.0, "grep": 0.5, "title_boost": 1.5},
}


class HybridSearcher:
    """Search combining BM25 index and Grep raw-file search via RRF.

    Args:
        bm25_searcher: Configured BM25Searcher instance (readonly).
        grep_raw_dir: Directory containing raw .md files for GrepTool.
        bm25_weight: Weight multiplier for BM25 RRF contributions.
            Overrides the profile default when explicitly provided.
        grep_weight: Weight multiplier for Grep RRF contributions.
            Overrides the profile default when explicitly provided.
        profile: Search profile name (e.g. ``"legal"``, ``"technical"``).
            Controls default weights and title boosting.  ``"general"``
            is the default and preserves backward-compatible weights.
    """

    RRF_K: int = 60  # Standard RRF constant

    def __init__(
        self,
        bm25_searcher: BM25Searcher,
        grep_raw_dir: Path,
        bm25_weight: Optional[float] = None,
        grep_weight: Optional[float] = None,
        profile: str = "general",
    ):
        self._bm25 = bm25_searcher
        self._grep_raw_dir = Path(grep_raw_dir)
        self._profile = profile

        # Resolve weights: explicit param > profile > historical default
        pw = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["general"])
        self._bm25_weight = bm25_weight if bm25_weight is not None else pw["bm25"]
        self._grep_weight = grep_weight if grep_weight is not None else pw["grep"]

    # ── Profile helpers ─────────────────────────────────────────

    @property
    def profile(self) -> str:
        """Active search profile name."""
        return self._profile

    @classmethod
    def get_profile_weights(cls, profile: str) -> dict[str, float]:
        """Return weight dict for *profile*, falling back to ``"general"``."""
        return PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["general"]).copy()

    # ── Public API ─────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        bm25_limit: int = 20,
        grep_limit: int = 50,
    ) -> UnifiedSearchResults:
        """Execute parallel BM25 + Grep search and RRF-merge results.

        Args:
            query: Search query string.
            limit: Maximum final results to return.
            bm25_limit: How many results to request from BM25.
            grep_limit: How many line matches to request from Grep.

        Returns:
            UnifiedSearchResults with RRF-merged, ranked results.
        """
        start_time = time.time()

        bm25_results: Optional[PaginatedResults] = None
        grep_result: Optional[ToolResult] = None

        # Phase 1: Parallel retrieval
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self._bm25.search, query, bm25_limit): "bm25",
                pool.submit(self._run_grep, query, grep_limit): "grep",
            }
            for future in as_completed(futures):
                label = futures[future]
                if label == "bm25":
                    bm25_results = future.result()
                else:
                    grep_result = future.result()

        # Phase 2: Convert to UnifiedSearchResult lists
        bm25_unified = self._convert_bm25(bm25_results) if bm25_results else []
        grep_unified = self._convert_grep(grep_result) if grep_result else []

        bm25_count = len(bm25_unified)
        grep_count = len(grep_unified)

        # Phase 3: RRF merge
        merged = self._rrf_merge(bm25_unified, grep_unified)

        # Phase 4: Sort, limit, rank
        merged.sort(key=lambda r: r.rrf_score, reverse=True)
        merged = merged[:limit]
        for rank, r in enumerate(merged, 1):
            r.rank = rank

        elapsed = time.time() - start_time

        sources_used: List[str] = []
        if bm25_count > 0:
            sources_used.append("bm25")
        if grep_count > 0:
            sources_used.append("grep")

        return UnifiedSearchResults(
            results=merged,
            total=len(set(
                (str(r.source_path) if r.source_path else r.doc_id)
                for r in bm25_unified + grep_unified
            )),
            query=query,
            sources_used=sources_used,
            execution_time=elapsed,
            bm25_count=bm25_count,
            grep_count=grep_count,
        )

    # ── Internal helpers ───────────────────────────────────────

    def _run_grep(self, query: str, grep_limit: int) -> ToolResult:
        """Execute GrepTool search (callable for ThreadPoolExecutor)."""
        from src.agent.tools.grep import GrepTool

        grep_tool = GrepTool(raw_dir=self._grep_raw_dir, max_results=grep_limit)
        return grep_tool.execute(
            pattern=query,
            case_sensitive=False,
            max_results=grep_limit,
            file_filter="*.md",
        )

    def _convert_bm25(self, paginated: PaginatedResults) -> List[UnifiedSearchResult]:
        """Convert BM25 SearchPreview results to UnifiedSearchResult."""
        results: List[UnifiedSearchResult] = []
        for preview in paginated.results:
            results.append(
                UnifiedSearchResult(
                    doc_id=preview.doc_id,
                    source_path=preview.source_path,
                    title=preview.title,
                    snippet=preview.snippet,
                    highlights=preview.highlights,
                    raw_score=preview.score,
                    search_source=SearchSource.BM25,
                    retrieval_time=paginated.execution_time,
                )
            )
        return results

    def _convert_grep(self, tool_result: ToolResult) -> List[UnifiedSearchResult]:
        """Convert GrepTool output to UnifiedSearchResult (one per file).

        GrepTool returns result.data as ``"file:line: content\\n..."``.
        We aggregate by file, count matches, and compute a synthetic score.
        """
        if not tool_result.success or tool_result.data == "No matches found.":
            return []

        # Aggregate matches by file
        file_data: dict[str, dict] = defaultdict(
            lambda: {"matches": 0, "lines": [], "first_match": ""}
        )

        output_lines = tool_result.data.split("\n") if isinstance(tool_result.data, str) else []
        for line in output_lines:
            if line.startswith("  "):
                continue  # skip context lines
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            file_key = parts[0]
            line_no = parts[1].strip()
            content = parts[2].strip()

            file_data[file_key]["matches"] += 1
            file_data[file_key]["lines"].append({
                "file": file_key,
                "line_no": int(line_no) if line_no.isdigit() else 0,
                "content": content[:200],
            })
            if not file_data[file_key]["first_match"]:
                file_data[file_key]["first_match"] = content[:200]

        grep_time = tool_result.metadata.get("execution_time", 0)

        results: List[UnifiedSearchResult] = []
        for file_key, data in file_data.items():
            match_count = data["matches"]
            # Synthetic score: log1p(match_count) / log1p(50)  ∈ (0, 1]
            synthetic_score = math.log1p(match_count) / math.log1p(50)

            # Derive a doc_id from file path
            doc_id = file_key.replace("\\", "/").replace("/", "_").rstrip(".md")

            results.append(
                UnifiedSearchResult(
                    doc_id=doc_id,
                    source_path=Path(file_key),
                    title=Path(file_key).stem,
                    snippet=data["first_match"],
                    raw_score=synthetic_score,
                    search_source=SearchSource.GREP,
                    grep_matches=match_count,
                    grep_line_matches=data["lines"],
                    retrieval_time=grep_time,
                )
            )

        return results

    def _rrf_merge(
        self,
        bm25_results: List[UnifiedSearchResult],
        grep_results: List[UnifiedSearchResult],
    ) -> List[UnifiedSearchResult]:
        """Reciprocal Rank Fusion merge using source_path as dedup key.

        Each source contributes:  weight / (RRF_K + rank)  to the doc's score.
        When the same document appears in both sources, data is enriched.
        """
        path_to_result: dict[str, UnifiedSearchResult] = {}
        rrf_scores: dict[str, float] = {}

        # BM25 contributions
        for rank, r in enumerate(bm25_results, 1):
            key = str(r.source_path) if r.source_path else r.doc_id
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self._bm25_weight / (self.RRF_K + rank)
            if key not in path_to_result:
                path_to_result[key] = r
            else:
                # Enrich: fill missing title from BM25
                if not path_to_result[key].title and r.title:
                    path_to_result[key].title = r.title

        # Grep contributions
        for rank, r in enumerate(grep_results, 1):
            key = str(r.source_path) if r.source_path else r.doc_id
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self._grep_weight / (self.RRF_K + rank)
            if key not in path_to_result:
                path_to_result[key] = r
            else:
                # Enrich with grep match details
                path_to_result[key].grep_matches = r.grep_matches
                path_to_result[key].grep_line_matches = r.grep_line_matches
                # If the existing entry has no snippet but grep does, use grep's
                if not path_to_result[key].snippet and r.snippet:
                    path_to_result[key].snippet = r.snippet

        # Assign RRF scores
        for key, r in path_to_result.items():
            r.rrf_score = rrf_scores[key]

        return list(path_to_result.values())
