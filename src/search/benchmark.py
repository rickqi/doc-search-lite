"""Benchmark runner for comparing search modes.

Runs queries against BM25 (Tantivy) and Grep (regex) search backends,
collecting latency, result count, hit rate, and MRR metrics.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QuerySpec:
    """A single benchmark query with optional ground truth."""

    query: str
    expected_files: List[str] = field(default_factory=list)
    category: str = ""


@dataclass
class ModeResult:
    """Result from running one query in one search mode."""

    mode: str
    query: str
    success: bool
    latency: float = 0.0
    result_count: int = 0
    result_files: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    hit_rate: float = 0.0  # % of expected_files found
    mrr: float = 0.0  # Mean Reciprocal Rank
    error: str = ""
    run_index: int = 0  # which repetition (0-based)


@dataclass
class BenchmarkResult:
    """Complete benchmark results across all queries and modes."""

    queries: List[QuerySpec]
    results: List[ModeResult]
    modes_tested: List[str]
    index_path: str
    total_time: float = 0.0


class BenchmarkRunner:
    """Run search benchmarks comparing different modes.

    Supported modes:
      - "bm25":  Tantivy BM25 full-text search via BM25Searcher
      - "grep":  Python regex search via GrepTool (DCI paradigm)

    Example::

        runner = BenchmarkRunner(index_path=Path("./index"))
        queries = [QuerySpec(query="年假", expected_files=["年假制度.docx.md"])]
        result = runner.run(queries, modes=["bm25", "grep"], runs=3, warmup=1)
    """

    def __init__(self, index_path: Path, raw_dir: Optional[Path] = None):
        self._index_path = Path(index_path)
        self._raw_dir = Path(raw_dir) if raw_dir else self._index_path.parent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        queries: List[QuerySpec],
        modes: Optional[List[str]] = None,
        runs: int = 1,
        warmup: int = 1,
    ) -> BenchmarkResult:
        """Run all queries against all modes, collect metrics.

        Args:
            queries: List of QuerySpec with query text and optional ground truth.
            modes: Search modes to test (default: ["bm25", "grep"]).
            runs: Number of repetitions per query/mode combination.
            warmup: Number of warmup iterations (results discarded).

        Returns:
            BenchmarkResult with all collected ModeResult entries.
        """
        if modes is None:
            modes = ["bm25", "grep"]

        start = time.time()
        results: List[ModeResult] = []

        # Warmup — discard results, just prime jieba / file system caches
        if warmup > 0 and queries:
            self._warmup(queries[0].query, modes, warmup)

        for query_spec in queries:
            for mode in modes:
                for run_idx in range(runs):
                    result = self._run_single(query_spec, mode, run_idx)
                    results.append(result)

        return BenchmarkResult(
            queries=queries,
            results=results,
            modes_tested=modes,
            index_path=str(self._index_path),
            total_time=time.time() - start,
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, query: str, modes: List[str], count: int) -> None:
        """Warmup runs to initialize jieba etc."""
        dummy_spec = QuerySpec(query=query)
        for _ in range(count):
            for mode in modes:
                try:
                    self._run_single(dummy_spec, mode)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Single execution
    # ------------------------------------------------------------------

    def _run_single(self, spec: QuerySpec, mode: str, run_index: int = 0) -> ModeResult:
        """Run a single query in a single mode and return metrics."""
        start = time.perf_counter()
        try:
            if mode == "bm25":
                files, scores = self._search_bm25(spec.query)
            elif mode == "grep":
                files, scores = self._search_grep(spec.query)
            else:
                return ModeResult(
                    mode=mode,
                    query=spec.query,
                    success=False,
                    error=f"Unknown mode: {mode}",
                    run_index=run_index,
                )

            latency = time.perf_counter() - start

            hit_rate, mrr = self._calc_relevance(files, spec.expected_files)

            return ModeResult(
                mode=mode,
                query=spec.query,
                success=True,
                latency=latency,
                result_count=len(files),
                result_files=files[:10],
                scores=scores[:10],
                hit_rate=hit_rate,
                mrr=mrr,
                run_index=run_index,
            )
        except Exception as exc:
            return ModeResult(
                mode=mode,
                query=spec.query,
                success=False,
                latency=time.perf_counter() - start,
                error=str(exc),
                run_index=run_index,
            )

    # ------------------------------------------------------------------
    # Relevance metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_relevance(
        result_files: List[str],
        expected_files: List[str],
    ) -> Tuple[float, float]:
        """Calculate hit_rate and MRR given result file list and ground truth.

        Returns (hit_rate, mrr).  Both are 0.0 when expected_files is empty.
        """
        if not expected_files:
            return 0.0, 0.0

        hits = 0
        mrr = 0.0
        for i, f in enumerate(result_files):
            for ef in expected_files:
                if ef in f:
                    hits += 1
                    if mrr == 0.0:
                        mrr = 1.0 / (i + 1)
                    break

        hit_rate = hits / len(expected_files)
        return hit_rate, mrr

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------

    def _search_bm25(self, query: str, limit: int = 20) -> Tuple[List[str], List[float]]:
        """Run BM25 search, return (files, scores)."""
        from src.search.bm25_search import create_searcher

        searcher = create_searcher(self._index_path, use_jieba=True, readonly=True)
        results = searcher.search(query, limit=limit)
        files = [
            str(r.source_path.name) if r.source_path else ""
            for r in results.results
        ]
        scores = [r.score for r in results.results]
        return files, scores

    def _search_grep(self, query: str, limit: int = 20) -> Tuple[List[str], List[float]]:
        """Run Grep search, return (files, [])."""
        from src.agent.tools.grep import GrepTool

        grep = GrepTool(raw_dir=self._raw_dir, max_results=limit)
        result = grep.execute(pattern=query, case_sensitive=False, file_filter="*.md")
        if not result.success or result.data == "No matches found.":
            return [], []

        # Parse "file:line: content" format — deduplicate files
        files: List[str] = []
        seen: set = set()
        for line in result.data.split("\n"):
            if line.startswith("  ") or not line:
                continue
            parts = line.split(":", 2)
            if parts:
                fname = Path(parts[0]).name
                if fname not in seen:
                    seen.add(fname)
                    files.append(fname)
        return files, []

    # ------------------------------------------------------------------
    # Query loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_queries(path: Path) -> List[QuerySpec]:
        """Load queries from a JSONL file.

        Each line is a JSON object with fields:
          - query (required)
          - expected_files (optional, list of filename substrings)
          - category (optional)

        Blank lines and lines starting with ``#`` are skipped.
        """
        queries: List[QuerySpec] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                data = json.loads(line)
                queries.append(
                    QuerySpec(
                        query=data["query"],
                        expected_files=data.get("expected_files", []),
                        category=data.get("category", ""),
                    )
                )
        return queries

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_by_mode(
        results: List[ModeResult],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate ModeResult list into per-mode summary statistics.

        Returns ``{mode: {avg_latency, avg_hit_rate, avg_mrr, avg_result_count, success_rate}}``.
        Only successful results are included in latency / hit_rate / mrr averages.
        """
        buckets: Dict[str, List[ModeResult]] = {}
        for r in results:
            buckets.setdefault(r.mode, []).append(r)

        summary: Dict[str, Dict[str, float]] = {}
        for mode, mode_results in buckets.items():
            successful = [r for r in mode_results if r.success]
            total = len(mode_results)
            ok = len(successful)

            avg_latency = sum(r.latency for r in successful) / ok if ok else 0.0
            avg_hit_rate = sum(r.hit_rate for r in successful) / ok if ok else 0.0
            avg_mrr = sum(r.mrr for r in successful) / ok if ok else 0.0
            avg_result_count = sum(r.result_count for r in successful) / ok if ok else 0.0
            success_rate = ok / total if total else 0.0

            summary[mode] = {
                "avg_latency": avg_latency,
                "avg_hit_rate": avg_hit_rate,
                "avg_mrr": avg_mrr,
                "avg_result_count": avg_result_count,
                "success_rate": success_rate,
            }

        return summary
