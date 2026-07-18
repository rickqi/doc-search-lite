"""Local reranker using BAAI/bge-reranker-v2-m3 cross-encoder.

Drop-in replacement for ZhipuAIReranker. Runs entirely on local GPU/CPU —
no API key, no network calls, zero per-query cost.

Model: BAAI/bge-reranker-v2-m3 (~568MB, multilingual, supports Chinese)
Dependency: sentence-transformers (pip install sentence-transformers)

Usage::

    reranker = LocalReranker()
    results = reranker.rerank(
        query="年假有多少天",
        documents=["doc1 text...", "doc2 text..."],
        top_n=5,
    )

Environment:
    RERANKER_MODEL: Model name (default: BAAI/bge-reranker-v2-m3)
    RERANKER_DEVICE: Device (default: auto-detect GPU)
"""

import logging
import os
import time
from typing import List, Optional

from src.search.reranker import RerankResult, RerankUsage, MAX_DOCUMENTS, MAX_DOC_LENGTH, MAX_QUERY_LENGTH

logger = logging.getLogger(__name__)

# Lazy-loaded model instance (singleton — avoid reloading on every query)
_model_instance = None
_model_name = None


def _get_model(model_name: str, device: Optional[str] = None):
    """Lazy-load the cross-encoder model (singleton pattern)."""
    global _model_instance, _model_name

    if _model_instance is not None and _model_name == model_name:
        return _model_instance

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    logger.info("Loading reranker model: %s (first load takes ~10s)...", model_name)
    start = time.time()

    if device is None:
        device = os.environ.get("RERANKER_DEVICE", "")

    kwargs = {}
    if device:
        kwargs["device"] = device

    _model_instance = CrossEncoder(model_name, **kwargs)
    _model_name = model_name
    logger.info("Reranker model loaded in %.1fs", time.time() - start)
    return _model_instance


class LocalReranker:
    """Reranker using local BAAI/bge-reranker-v2-m3 cross-encoder.

    Same interface as ZhipuAIReranker — drop-in replacement.

    The model loads lazily on first ``rerank()`` call (~10s warmup).
    Subsequent calls use the cached model instance (singleton).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        usage_tracker=None,
        **kwargs,  # Absorb unused params (api_key, timeout, max_retries) for interface compat
    ):
        """Initialize the local reranker.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-reranker-v2-m3)
            device: torch device ("cuda", "cpu", or None for auto)
            usage_tracker: Optional UsageTracker (local rerank is free, but tracked for stats)
        """
        self._model_name = model_name or os.environ.get(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self._device = device
        self._usage_tracker = usage_tracker
        self._total_usage = RerankUsage()
        self._available = True  # Optimistic; verified on first rerank()

    # ------------------------------------------------------------------
    # Public helpers (same interface as ZhipuAIReranker)
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the reranker is ready."""
        return self._available

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed (always 0 for local — no API tokens)."""
        return self._total_usage.total_tokens

    @property
    def usage(self) -> RerankUsage:
        """Cumulative usage snapshot."""
        return RerankUsage(
            prompt_tokens=self._total_usage.prompt_tokens,
            total_tokens=self._total_usage.total_tokens,
        )

    # ------------------------------------------------------------------
    # Main entry point (same interface as ZhipuAIReranker)
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to the query.

        Args:
            query: Search query (truncated to 4096 chars).
            documents: Document texts to rank (max 128, each max 4096 chars).
            top_n: Number of top results to return.

        Returns:
            List of RerankResult sorted by relevance_score descending.
            Falls back to original order on any failure.
        """
        if not documents:
            return []

        # Validate and truncate inputs
        query = query[:MAX_QUERY_LENGTH]
        documents = [d[:MAX_DOC_LENGTH] for d in documents[:MAX_DOCUMENTS]]
        top_n = min(top_n, len(documents))

        try:
            model = _get_model(self._model_name, self._device)
        except ImportError as e:
            logger.error("Local reranker dependency missing: %s", e)
            self._available = False
            return self._fallback_order(documents, top_n)
        except Exception as e:
            logger.error("Failed to load reranker model: %s", e)
            self._available = False
            return self._fallback_order(documents, top_n)

        try:
            # Cross-encoder scoring: predict relevance for each (query, doc) pair
            pairs = [(query, doc) for doc in documents]
            start = time.time()
            scores = model.predict(pairs)
            elapsed = time.time() - start

            logger.debug(
                "Local rerank: %d docs in %.3fs (%.1f docs/s)",
                len(documents), elapsed, len(documents) / max(elapsed, 0.001),
            )

            # Build results sorted by score descending
            scored = [(i, float(scores[i])) for i in range(len(documents))]
            scored.sort(key=lambda x: x[1], reverse=True)

            results = [
                RerankResult(index=idx, relevance_score=round(score, 6))
                for idx, score in scored[:top_n]
            ]

            # Track usage (local is free, but record for diagnostics)
            if self._usage_tracker:
                self._usage_tracker.record_rerank(
                    model=self._model_name,
                    input_tokens=0,
                    total_tokens=0,
                )

            return results

        except Exception as e:
            logger.error("Local rerank failed: %s", e)
            return self._fallback_order(documents, top_n)

    @staticmethod
    def _fallback_order(
        documents: List[str],
        top_n: int,
    ) -> List[RerankResult]:
        """Fallback: return documents in original order with decreasing scores."""
        return [
            RerankResult(index=i, relevance_score=round(1.0 - i * 0.01, 4))
            for i in range(min(top_n, len(documents)))
        ]


def create_reranker(
    config=None,
    usage_tracker=None,
    reranker_type: Optional[str] = None,
):
    """Factory: create the appropriate reranker based on configuration.

    Args:
        config: Configuration object (for cloud API key)
        usage_tracker: Optional UsageTracker
        reranker_type: Override type ("zhipu" or "local").
            If None, reads from RERANKER_TYPE env var (default: "zhipu").

    Returns:
        ZhipuAIReranker or LocalReranker instance.
    """
    rtype = reranker_type or os.environ.get("RERANKER_TYPE", "zhipu")

    if rtype == "local":
        logger.info("Using LOCAL reranker (bge-reranker-v2-m3)")
        return LocalReranker(usage_tracker=usage_tracker)

    # Default: cloud ZhipuAI
    from src.search.reranker import ZhipuAIReranker
    api_key = ""
    if config and hasattr(config, "glm_api_key"):
        api_key = config.glm_api_key
    return ZhipuAIReranker(api_key=api_key, usage_tracker=usage_tracker)
