"""Tests for LocalReranker — local cross-encoder reranking."""

from unittest.mock import MagicMock, patch

import pytest


class TestLocalRerankerBasics:
    """Basic LocalReranker tests."""

    def test_init_defaults(self):
        from src.search.local_reranker import LocalReranker

        r = LocalReranker()
        assert r.available is True
        assert r.tokens_used == 0
        assert "bge-reranker" in r._model_name

    def test_init_custom_model(self):
        from src.search.local_reranker import LocalReranker

        r = LocalReranker(model_name="custom/model")
        assert r._model_name == "custom/model"

    def test_usage_property(self):
        from src.search.local_reranker import LocalReranker

        r = LocalReranker()
        usage = r.usage
        assert usage.total_tokens == 0
        assert usage.prompt_tokens == 0


class TestLocalRerankerExecute:
    """LocalReranker.rerank() tests."""

    def test_empty_documents(self):
        from src.search.local_reranker import LocalReranker

        r = LocalReranker()
        result = r.rerank("query", [], top_n=5)
        assert result == []

    def test_fallback_on_import_error(self):
        """Should fallback when sentence-transformers not installed."""
        from src.search.local_reranker import LocalReranker

        r = LocalReranker()
        with patch("src.search.local_reranker._get_model", side_effect=ImportError("not installed")):
            result = r.rerank("query", ["doc1", "doc2"], top_n=2)
        assert len(result) == 2
        assert result[0].index == 0
        assert result[0].relevance_score > result[1].relevance_score
        assert r.available is False

    def test_fallback_on_model_error(self):
        """Should fallback when model loading fails."""
        from src.search.local_reranker import LocalReranker

        r = LocalReranker()
        with patch("src.search.local_reranker._get_model", side_effect=RuntimeError("GPU error")):
            result = r.rerank("query", ["doc1", "doc2"], top_n=2)
        assert len(result) == 2
        assert r.available is False

    def test_successful_rerank(self):
        """Should rerank documents by predicted scores."""
        from src.search.local_reranker import LocalReranker

        mock_model = MagicMock()
        # Mock: doc1 is more relevant (score 0.9) than doc2 (score 0.3)
        mock_model.predict.return_value = [0.3, 0.9]
        r = LocalReranker()
        with patch("src.search.local_reranker._get_model", return_value=mock_model):
            result = r.rerank("query", ["doc1", "doc2"], top_n=2)
        assert len(result) == 2
        assert result[0].index == 1  # doc2 has higher score
        assert result[0].relevance_score == pytest.approx(0.9, abs=0.001)
        assert result[1].index == 0  # doc1 has lower score
        assert result[1].relevance_score == pytest.approx(0.3, abs=0.001)

    def test_top_n_limit(self):
        """Should return only top_n results."""
        from src.search.local_reranker import LocalReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3, 0.7]
        r = LocalReranker()
        with patch("src.search.local_reranker._get_model", return_value=mock_model):
            result = r.rerank("query", ["d0", "d1", "d2", "d3"], top_n=2)
        assert len(result) == 2
        assert result[0].index == 1  # highest score 0.9
        assert result[1].index == 3  # second highest 0.7

    def test_predict_exception_fallback(self):
        """Should fallback when model.predict raises."""
        from src.search.local_reranker import LocalReranker

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("inference error")
        r = LocalReranker()
        with patch("src.search.local_reranker._get_model", return_value=mock_model):
            result = r.rerank("query", ["doc1", "doc2"], top_n=2)
        assert len(result) == 2
        assert result[0].index == 0  # fallback original order


class TestCreateReranker:
    """Factory function tests."""

    def test_default_creates_zhipu(self):
        import os

        from src.search.local_reranker import create_reranker
        from src.search.reranker import ZhipuAIReranker
        old = os.environ.pop("RERANKER_TYPE", None)
        try:
            r = create_reranker(config=None)
            assert isinstance(r, ZhipuAIReranker)
        finally:
            if old:
                os.environ["RERANKER_TYPE"] = old

    def test_local_type_creates_local(self):
        import os

        from src.search.local_reranker import LocalReranker, create_reranker
        old = os.environ.get("RERANKER_TYPE")
        os.environ["RERANKER_TYPE"] = "local"
        try:
            r = create_reranker(config=None)
            assert isinstance(r, LocalReranker)
        finally:
            if old is None:
                os.environ.pop("RERANKER_TYPE", None)
            else:
                os.environ["RERANKER_TYPE"] = old

    def test_explicit_type_overrides_env(self):
        import os

        from src.search.local_reranker import LocalReranker, create_reranker
        old = os.environ.get("RERANKER_TYPE")
        os.environ["RERANKER_TYPE"] = "zhipu"
        try:
            r = create_reranker(config=None, reranker_type="local")
            assert isinstance(r, LocalReranker)
        finally:
            if old is None:
                os.environ.pop("RERANKER_TYPE", None)
            else:
                os.environ["RERANKER_TYPE"] = old
