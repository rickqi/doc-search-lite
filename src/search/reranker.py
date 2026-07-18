"""Cloud-based reranker using ZhipuAI Rerank API.

Reranks BM25 search results using the ZhipuAI cross-encoder rerank model.
No local model inference required — all computation happens on ZhipuAI's servers.

Uses only stdlib (urllib.request) to avoid adding new dependencies.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_DOCUMENTS = 128
MAX_DOC_LENGTH = 4096
MAX_QUERY_LENGTH = 4096


@dataclass
class RerankResult:
    """A single reranked result."""

    index: int  # Original index in the input list
    relevance_score: float
    document: str | None = None


@dataclass
class RerankUsage:
    """Token usage tracking for rerank API calls."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class ZhipuAIReranker:
    """Reranker using ZhipuAI's cloud Rerank API.

    Usage::

        reranker = ZhipuAIReranker(api_key="...")
        results = reranker.rerank(
            query="年假有多少天",
            documents=["doc1 text...", "doc2 text..."],
            top_n=5,
        )
        for r in results:
            print(f"Score: {r.relevance_score:.4f} | Original index: {r.index}")

    Graceful degradation: if the API key is missing or the call fails,
    documents are returned in their original order with synthetic scores.
    """

    API_URL = "https://open.bigmodel.cn/api/paas/v4/rerank"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        usage_tracker=None,
    ):
        self._api_key = api_key or os.environ.get("GLM_API_KEY", "")
        self._timeout = timeout
        self._max_retries = max_retries
        self._total_usage = RerankUsage()
        self._usage_tracker = usage_tracker

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the reranker is configured (has an API key)."""
        return bool(self._api_key)

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed across all rerank calls."""
        return self._total_usage.total_tokens

    @property
    def usage(self) -> RerankUsage:
        """Cumulative token usage snapshot."""
        return RerankUsage(
            prompt_tokens=self._total_usage.prompt_tokens,
            total_tokens=self._total_usage.total_tokens,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to the query.

        Args:
            query: Search query (max 4096 chars, truncated if longer).
            documents: Document texts to rank (max 128, each max 4096 chars).
            top_n: Number of top results to return.

        Returns:
            List of :class:`RerankResult` sorted by *relevance_score*
            descending.  Falls back to original order on any failure.
        """
        if not documents:
            return []

        if not self.available:
            logger.warning(
                "Reranker not available (no API key), returning original order"
            )
            return self._fallback_order(documents, top_n)

        # Validate and truncate inputs
        query = query[:MAX_QUERY_LENGTH]
        documents = [d[:MAX_DOC_LENGTH] for d in documents[:MAX_DOCUMENTS]]
        top_n = min(top_n, len(documents))

        for attempt in range(self._max_retries + 1):
            try:
                return self._call_api(query, documents, top_n)
            except Exception as exc:
                if attempt == self._max_retries:
                    logger.error(
                        "Rerank API failed after %d attempts: %s",
                        attempt + 1,
                        exc,
                    )
                    return self._fallback_order(documents, top_n)
                backoff = 0.5 * (2 ** attempt)
                logger.debug("Rerank attempt %d failed, retrying in %.1fs", attempt + 1, backoff)
                time.sleep(backoff)

        # Should not be reached, but just in case
        return self._fallback_order(documents, top_n)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        """Make the actual HTTP call to the ZhipuAI Rerank API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "model": "rerank",
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.API_URL,
            data=payload,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")

        data = json.loads(body)

        # Track token usage
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        self._total_usage.prompt_tokens += prompt_tokens
        self._total_usage.total_tokens += total_tokens

        # Track via UsageTracker if available
        if self._usage_tracker:
            self._usage_tracker.record_rerank(
                model="rerank",
                input_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )

        # Parse results
        results: list[RerankResult] = []
        for r in data.get("results", []):
            results.append(
                RerankResult(
                    index=r["index"],
                    relevance_score=r["relevance_score"],
                )
            )

        return results

    @staticmethod
    def _fallback_order(
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        """Fallback: return documents in original order with decreasing scores."""
        return [
            RerankResult(index=i, relevance_score=round(1.0 - i * 0.01, 4))
            for i in range(min(top_n, len(documents)))
        ]
