"""Unit tests for ZhipuAIReranker class.

Tests cover:
- Initialization and properties
- Rerank with mocked API success
- Fallback order on API failure
- No API key behavior
- Empty documents handling
- Input truncation for long documents
- Token usage tracking
- Retry logic
- _fallback_order scoring
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.search.reranker import MAX_DOC_LENGTH, MAX_DOCUMENTS, RerankResult, RerankUsage, ZhipuAIReranker


class TestZhipuAIRerankerInit:
    """Test ZhipuAIReranker initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        reranker = ZhipuAIReranker(api_key="test-key")
        assert reranker._api_key == "test-key"
        assert reranker._timeout == 30.0
        assert reranker._max_retries == 2

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        reranker = ZhipuAIReranker(
            api_key="custom-key",
            timeout=60.0,
            max_retries=5,
        )
        assert reranker._api_key == "custom-key"
        assert reranker._timeout == 60.0
        assert reranker._max_retries == 5

    def test_init_no_api_key(self):
        """Test initialization without API key uses env var."""
        with patch.dict("os.environ", {"GLM_API_KEY": "env-key"}, clear=False):
            reranker = ZhipuAIReranker()
            assert reranker._api_key == "env-key"

    def test_init_no_key_no_env(self):
        """Test initialization with no key and no env var."""
        with patch.dict("os.environ", {}, clear=True):
            reranker = ZhipuAIReranker()
            assert reranker._api_key == ""


class TestZhipuAIRerankerProperties:
    """Test ZhipuAIReranker properties."""

    def test_available_with_api_key(self):
        """Test available is True with API key."""
        reranker = ZhipuAIReranker(api_key="test-key")
        assert reranker.available is True

    def test_available_without_api_key(self):
        """Test available is False without API key."""
        with patch.dict("os.environ", {}, clear=True):
            reranker = ZhipuAIReranker(api_key="")
            assert reranker.available is False

    def test_tokens_used_initial_zero(self):
        """Test tokens_used starts at 0."""
        reranker = ZhipuAIReranker(api_key="test-key")
        assert reranker.tokens_used == 0

    def test_usage_initial_zero(self):
        """Test usage starts with zero values."""
        reranker = ZhipuAIReranker(api_key="test-key")
        usage = reranker.usage
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_returns_rerank_usage(self):
        """Test usage property returns RerankUsage instance."""
        reranker = ZhipuAIReranker(api_key="test-key")
        assert isinstance(reranker.usage, RerankUsage)


class TestZhipuAIRerankerSuccess:
    """Test successful rerank API calls."""

    @pytest.fixture
    def reranker(self):
        """Create reranker with test API key."""
        return ZhipuAIReranker(api_key="test-key")

    def _mock_urlopen_response(self, results_data, usage_data=None):
        """Create a mock urlopen response."""
        mock_resp = MagicMock()
        response_body = {
            "results": results_data,
            "usage": usage_data or {"prompt_tokens": 50, "total_tokens": 50},
        }
        mock_resp.read.return_value = json.dumps(response_body).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_success(self, mock_urlopen, reranker):
        """Test successful rerank returns sorted results."""
        mock_urlopen.return_value = self._mock_urlopen_response(
            results_data=[
                {"index": 1, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.75},
            ],
            usage_data={"prompt_tokens": 72, "total_tokens": 72},
        )

        results = reranker.rerank("查询", ["文档1", "文档2"], top_n=2)

        assert len(results) == 2
        assert results[0].index == 1
        assert results[0].relevance_score == 0.99
        assert results[1].index == 0
        assert results[1].relevance_score == 0.75

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_success_returns_rerank_results(self, mock_urlopen, reranker):
        """Test that results are RerankResult instances."""
        mock_urlopen.return_value = self._mock_urlopen_response(
            results_data=[{"index": 0, "relevance_score": 0.9}],
        )

        results = reranker.rerank("查询", ["文档1"])
        assert len(results) == 1
        assert isinstance(results[0], RerankResult)

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_sends_correct_payload(self, mock_urlopen, reranker):
        """Test that rerank sends correct API payload."""
        mock_urlopen.return_value = self._mock_urlopen_response(
            results_data=[{"index": 0, "relevance_score": 0.9}],
        )

        reranker.rerank("测试查询", ["文档1", "文档2"], top_n=2)

        # Check the request was made
        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert "Bearer test-key" in req.headers.get("Authorization", "")

        # Check payload
        body = json.loads(req.data.decode("utf-8"))
        assert body["query"] == "测试查询"
        assert body["documents"] == ["文档1", "文档2"]
        assert body["top_n"] == 2
        assert body["model"] == "rerank"

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_top_n_limit(self, mock_urlopen, reranker):
        """Test that top_n limits returned results."""
        mock_urlopen.return_value = self._mock_urlopen_response(
            results_data=[
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
            ],
        )

        results = reranker.rerank(
            "查询", ["文档1", "文档2", "文档3"], top_n=2
        )
        assert len(results) == 2


class TestZhipuAIRerankerFallback:
    """Test fallback behavior on failures."""

    def test_rerank_no_api_key_returns_fallback(self):
        """Test fallback order when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            reranker = ZhipuAIReranker(api_key="")
            results = reranker.rerank("查询", ["文档1", "文档2"], top_n=2)

            assert len(results) == 2
            assert results[0].index == 0
            assert results[1].index == 1
            # Scores should decrease by 0.01
            assert results[0].relevance_score == 1.0
            assert results[1].relevance_score == 0.99

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_api_error_returns_fallback(self, mock_urlopen):
        """Test fallback order when API call raises error."""
        mock_urlopen.side_effect = Exception("Network error")

        reranker = ZhipuAIReranker(api_key="test-key", max_retries=0)
        results = reranker.rerank("查询", ["文档1", "文档2"], top_n=2)

        assert len(results) == 2
        assert results[0].index == 0
        assert results[1].index == 1

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_http_error_returns_fallback(self, mock_urlopen):
        """Test fallback on HTTP error."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.test.com", 500, "Server Error", {}, None
        )

        reranker = ZhipuAIReranker(api_key="test-key", max_retries=0)
        results = reranker.rerank("查询", ["文档1"], top_n=1)

        assert len(results) == 1
        assert results[0].index == 0

    def test_rerank_empty_documents_returns_empty(self):
        """Test that empty documents returns empty list."""
        reranker = ZhipuAIReranker(api_key="test-key")
        results = reranker.rerank("查询", [])
        assert results == []

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_fallback_preserves_count(self, mock_urlopen):
        """Test fallback returns correct number of results."""
        mock_urlopen.side_effect = Exception("Error")

        reranker = ZhipuAIReranker(api_key="test-key", max_retries=0)
        docs = ["文档1", "文档2", "文档3"]
        results = reranker.rerank("查询", docs, top_n=2)

        assert len(results) == 2


class TestZhipuAIRerankerInputTruncation:
    """Test input truncation for long documents."""

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_truncates_long_documents(self, mock_urlopen):
        """Test that long documents are truncated to MAX_DOC_LENGTH."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"prompt_tokens": 100, "total_tokens": 100},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reranker = ZhipuAIReranker(api_key="test-key")
        long_doc = "x" * (MAX_DOC_LENGTH + 1000)

        reranker.rerank("查询", [long_doc], top_n=1)

        # Verify the payload was truncated
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert len(body["documents"][0]) == MAX_DOC_LENGTH

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_truncates_long_query(self, mock_urlopen):
        """Test that long query is truncated."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"prompt_tokens": 100, "total_tokens": 100},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reranker = ZhipuAIReranker(api_key="test-key")
        long_query = "q" * 5000

        reranker.rerank(long_query, ["文档1"], top_n=1)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert len(body["query"]) <= 4096

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_limits_document_count(self, mock_urlopen):
        """Test that documents beyond MAX_DOCUMENTS are dropped."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"prompt_tokens": 100, "total_tokens": 100},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reranker = ZhipuAIReranker(api_key="test-key")
        docs = [f"文档{i}" for i in range(MAX_DOCUMENTS + 10)]

        reranker.rerank("查询", docs, top_n=1)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert len(body["documents"]) == MAX_DOCUMENTS


class TestZhipuAIRerankerTokenTracking:
    """Test token usage tracking."""

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_tracks_tokens(self, mock_urlopen):
        """Test that tokens_used accumulates across calls."""
        mock_resp1 = MagicMock()
        mock_resp1.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"prompt_tokens": 50, "total_tokens": 100},
        }).encode("utf-8")
        mock_resp1.__enter__ = lambda s: mock_resp1
        mock_resp1.__exit__ = MagicMock(return_value=False)

        mock_resp2 = MagicMock()
        mock_resp2.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.8}],
            "usage": {"prompt_tokens": 30, "total_tokens": 60},
        }).encode("utf-8")
        mock_resp2.__enter__ = lambda s: mock_resp2
        mock_resp2.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [mock_resp1, mock_resp2]

        reranker = ZhipuAIReranker(api_key="test-key")

        # First call
        reranker.rerank("查询1", ["文档1"], top_n=1)
        assert reranker.tokens_used == 100

        # Second call
        reranker.rerank("查询2", ["文档2"], top_n=1)
        assert reranker.tokens_used == 160

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_usage_snapshot(self, mock_urlopen):
        """Test that usage property returns correct snapshot."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"prompt_tokens": 50, "total_tokens": 100},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reranker = ZhipuAIReranker(api_key="test-key")
        reranker.rerank("查询", ["文档"], top_n=1)

        usage = reranker.usage
        assert usage.prompt_tokens == 50
        assert usage.total_tokens == 100

    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_missing_usage_defaults_to_zero(self, mock_urlopen):
        """Test that missing usage in response defaults to 0."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reranker = ZhipuAIReranker(api_key="test-key")
        reranker.rerank("查询", ["文档"], top_n=1)

        assert reranker.tokens_used == 0


class TestZhipuAIRerankerRetry:
    """Test retry logic."""

    @patch("src.search.reranker.time.sleep")
    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_retry_then_succeed(self, mock_urlopen, mock_sleep):
        """Test first call fails, second succeeds."""
        mock_fail_resp = MagicMock()
        mock_fail_resp.__enter__ = lambda s: mock_fail_resp
        mock_fail_resp.__exit__ = MagicMock(return_value=False)

        mock_success_resp = MagicMock()
        mock_success_resp.read.return_value = json.dumps({
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"prompt_tokens": 50, "total_tokens": 50},
        }).encode("utf-8")
        mock_success_resp.__enter__ = lambda s: mock_success_resp
        mock_success_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            Exception("Temporary error"),
            mock_success_resp,
        ]

        reranker = ZhipuAIReranker(api_key="test-key", max_retries=1)
        results = reranker.rerank("查询", ["文档"], top_n=1)

        assert len(results) == 1
        assert results[0].relevance_score == 0.9
        # Should have slept for backoff
        mock_sleep.assert_called_once()

    @patch("src.search.reranker.time.sleep")
    @patch("src.search.reranker.urllib.request.urlopen")
    def test_rerank_all_retries_fail(self, mock_urlopen, mock_sleep):
        """Test that all retries exhausted returns fallback."""
        mock_urlopen.side_effect = Exception("Persistent error")

        reranker = ZhipuAIReranker(api_key="test-key", max_retries=2)
        results = reranker.rerank("查询", ["文档1", "文档2"], top_n=2)

        # Should return fallback order
        assert len(results) == 2
        assert results[0].index == 0
        assert results[1].index == 1
        # Should have retried max_retries times
        assert mock_urlopen.call_count == 3  # initial + 2 retries


class TestFallbackOrder:
    """Test _fallback_order static method."""

    def test_fallback_order_scores(self):
        """Test scores decrease by 0.01."""
        results = ZhipuAIReranker._fallback_order(["a", "b", "c"], 3)
        assert results[0].relevance_score == 1.0
        assert results[1].relevance_score == 0.99
        assert results[2].relevance_score == 0.98

    def test_fallback_order_indices(self):
        """Test indices match document positions."""
        results = ZhipuAIReranker._fallback_order(["a", "b"], 2)
        assert results[0].index == 0
        assert results[1].index == 1

    def test_fallback_order_top_n_limits(self):
        """Test top_n limits results."""
        results = ZhipuAIReranker._fallback_order(["a", "b", "c"], 2)
        assert len(results) == 2

    def test_fallback_order_more_top_n_than_docs(self):
        """Test top_n > len(documents) is handled."""
        results = ZhipuAIReranker._fallback_order(["a"], 5)
        assert len(results) == 1

    def test_fallback_order_returns_rerank_results(self):
        """Test that results are RerankResult instances."""
        results = ZhipuAIReranker._fallback_order(["a"], 1)
        assert isinstance(results[0], RerankResult)


class TestRerankResultDataclass:
    """Test RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Test creating RerankResult."""
        result = RerankResult(index=0, relevance_score=0.95)
        assert result.index == 0
        assert result.relevance_score == 0.95
        assert result.document is None

    def test_rerank_result_with_document(self):
        """Test RerankResult with document text."""
        result = RerankResult(index=1, relevance_score=0.8, document="some text")
        assert result.document == "some text"


class TestRerankUsageDataclass:
    """Test RerankUsage dataclass."""

    def test_rerank_usage_default_values(self):
        """Test RerankUsage default values."""
        usage = RerankUsage()
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0

    def test_rerank_usage_custom_values(self):
        """Test RerankUsage with custom values."""
        usage = RerankUsage(prompt_tokens=100, total_tokens=200)
        assert usage.prompt_tokens == 100
        assert usage.total_tokens == 200


class TestConstants:
    """Test module-level constants."""

    def test_max_documents(self):
        """Test MAX_DOCUMENTS constant."""
        assert MAX_DOCUMENTS == 128

    def test_max_doc_length(self):
        """Test MAX_DOC_LENGTH constant."""
        assert MAX_DOC_LENGTH == 4096
