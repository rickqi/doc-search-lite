"""Query performance diagnostics collector.

Records step timings and LLM call details during query execution,
then persists to SQLite for analysis.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""

    call_type: str
    call_sequence: int = 0
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    retry_count: int = 0
    model: str = ""
    cache_hit: bool = False
    cached_tokens: int = 0


class DiagnosticsCollector:
    """Per-query performance diagnostics collector.

    Usage::

        collector = DiagnosticsCollector(db)
        collector.start_query("query text", "medium", session_id="abc123")
        collector.record_step("classify", 0.5)
        collector.record_llm_call("expand", "glm-4", latency_ms=1200.0)
        collector.finish(success=True, model="glm-4", provider="glm")
    """

    def __init__(self, db: Any = None):
        """Initialize collector.

        Args:
            db: ConvertDB instance (optional — if None, no data is persisted)
        """
        self._db = db
        self._steps: Dict[str, float] = {}
        self._llm_calls: List[LLMCallRecord] = []
        self._step_starts: Dict[str, float] = {}
        self._tool_call_count: int = 0
        self._tool_cache_hits: int = 0
        self._query_hash: str = ""
        self._query_preview: str = ""
        self._complexity: str = ""
        self._session_id: str = ""
        self._start_time: float = 0.0
        self._finished: bool = False

    @property
    def is_active(self) -> bool:
        """Whether the collector is currently tracking a query."""
        return self._start_time > 0 and not self._finished

    def start_query(self, query: str, complexity: str, session_id: str = "") -> None:
        """Start tracking a query."""
        self._query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        self._query_preview = query[:50]
        self._complexity = complexity
        self._session_id = session_id or ""
        self._start_time = time.time()
        self._finished = False
        self._steps.clear()
        self._llm_calls.clear()
        self._step_starts.clear()
        self._tool_call_count = 0
        self._tool_cache_hits = 0

    def record_step(self, name: str, duration_ms: float) -> None:
        """Record timing for a pipeline step."""
        self._steps[name] = duration_ms

    def record_step_start(self, name: str) -> None:
        """Mark the start of a step."""
        self._step_starts[name] = time.time()

    def record_step_end(self, name: str) -> float:
        """Complete a step started with record_step_start(). Returns duration_ms."""
        if name in self._step_starts:
            duration_ms = (time.time() - self._step_starts[name]) * 1000
            self._steps[name] = duration_ms
            del self._step_starts[name]
            return duration_ms
        return 0.0

    def record_llm_call(
        self,
        call_type: str,
        model: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        retry_count: int = 0,
        cache_hit: bool = False,
        cached_tokens: int = 0,
        ttft_ms: Optional[float] = None,
    ) -> None:
        """Record a single LLM API call."""
        self._llm_calls.append(LLMCallRecord(
            call_type=call_type,
            call_sequence=len(self._llm_calls) + 1,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens or input_tokens + output_tokens,
            retry_count=retry_count,
            model=model,
            cache_hit=cache_hit,
            cached_tokens=cached_tokens,
        ))

    def record_tool_call(self, cache_hit: bool = False) -> None:
        """Record a tool invocation."""
        self._tool_call_count += 1
        if cache_hit:
            self._tool_cache_hits += 1

    def finish(
        self,
        success: bool,
        error_type: str = None,
        model: str = "",
        provider: str = "",
        search_count: int = 0,
        read_count: int = 0,
        result_count: int = 0,
        coverage_score: Optional[float] = None,
        feedback_rounds: int = 0,
        final_sufficient: bool = False,
        search_mode: str = "agent",
        source_dir: str = "",
    ) -> Optional[int]:
        """Finish tracking and persist to DB. Returns diagnostic_id or None."""
        self._finished = True
        if not self._db:
            return None

        total_ms = int((time.time() - self._start_time) * 1000) if self._start_time else 0
        llm_total_ms = sum(c.latency_ms for c in self._llm_calls)
        llm_input = sum(c.input_tokens for c in self._llm_calls)
        llm_output = sum(c.output_tokens for c in self._llm_calls)
        llm_retries = sum(c.retry_count for c in self._llm_calls)
        tool_ms = sum(v for k, v in self._steps.items() if k in ("search", "read", "grep", "rerank"))
        step_json = json.dumps(self._steps, ensure_ascii=False) if self._steps else None

        try:
            diag_id = self._db.add_query_diagnostic(
                session_id=self._session_id,
                query_hash=self._query_hash,
                query_preview=self._query_preview,
                complexity=self._complexity,
                total_ms=total_ms,
                success=1 if success else 0,
                error_type=error_type,
                llm_call_count=len(self._llm_calls),
                llm_total_ms=int(llm_total_ms),
                llm_input_tokens=llm_input,
                llm_output_tokens=llm_output,
                llm_retry_count=llm_retries,
                tool_call_count=self._tool_call_count,
                tool_total_ms=int(tool_ms),
                tool_cache_hits=self._tool_cache_hits,
                step_timings=step_json,
                model=model,
                provider=provider,
                search_count=search_count,
                read_count=read_count,
                result_count=result_count,
                coverage_score=coverage_score,
                feedback_rounds=feedback_rounds,
                final_sufficient=1 if final_sufficient else 0,
                search_mode=search_mode,
                source_dir=source_dir,
            )

            for call in self._llm_calls:
                self._db.add_llm_call_log(
                    diagnostic_id=diag_id,
                    call_type=call.call_type,
                    call_sequence=call.call_sequence,
                    latency_ms=int(call.latency_ms),
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    total_tokens=call.total_tokens,
                    retry_count=call.retry_count,
                    model=call.model,
                    cache_hit=1 if call.cache_hit else 0,
                    cached_tokens=call.cached_tokens,
                )

            logger.debug(f"Diagnostics persisted: diag_id={diag_id}, total_ms={total_ms}")
            return diag_id
        except Exception as e:
            logger.warning(f"Failed to persist diagnostics: {e}")
            return None

    def get_step_timings(self) -> Dict[str, float]:
        """Get current step timings."""
        return dict(self._steps)

    def get_llm_stats(self) -> Dict[str, Any]:
        """Get aggregated LLM stats."""
        return {
            "call_count": len(self._llm_calls),
            "latency_total_ms": sum(c.latency_ms for c in self._llm_calls),
            "input_tokens": sum(c.input_tokens for c in self._llm_calls),
            "output_tokens": sum(c.output_tokens for c in self._llm_calls),
            "retry_count": sum(c.retry_count for c in self._llm_calls),
            "cache_hits": sum(1 for c in self._llm_calls if c.cache_hit),
        }
