"""
Tests for Dify External Knowledge Base API integration.

Covers:
- ScoreNormalizer: sigmoid normalization
- KnowledgeBaseMapping: config loading, resolve, error handling
- MetadataConditionEvaluator: filter operators
- Pydantic model validation
- /retrieval endpoint (route registration + auth + error formats)
"""

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.web.dify_retrieval import (
    DifyErrorCode,
    DifyErrorResponse,
    DifyRecord,
    DifyRetrievalRequest,
    DifyRetrievalResponse,
    DifyRetrievalSetting,
    KnowledgeBaseConfig,
    KnowledgeBaseMapping,
    MetadataCondition,
    MetadataConditionEvaluator,
    MetadataConditionItem,
    ScoreNormalizer,
)

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════
# ScoreNormalizer
# ═══════════════════════════════════════════════════════════════


class TestScoreNormalizer:
    """Tests for BM25 → 0-1 sigmoid normalization."""

    def test_positive_score_returns_between_0_1(self):
        norm = ScoreNormalizer(k=0.1)
        result = norm.normalize(10.0)
        assert 0.0 < result < 1.0

    def test_zero_score_returns_zero(self):
        norm = ScoreNormalizer(k=0.1)
        assert norm.normalize(0.0) == 0.0

    def test_negative_score_returns_zero(self):
        norm = ScoreNormalizer(k=0.1)
        assert norm.normalize(-5.0) == 0.0

    def test_high_score_approaches_one(self):
        norm = ScoreNormalizer(k=0.1)
        result = norm.normalize(100.0)
        assert result > 0.99

    def test_monotonic(self):
        """Higher BM25 score → higher normalized score."""
        norm = ScoreNormalizer(k=0.1)
        low = norm.normalize(5.0)
        high = norm.normalize(20.0)
        assert high > low

    def test_batch_normalize_single(self):
        norm = ScoreNormalizer(k=0.1)
        result = norm.normalize_batch([5.0])
        assert len(result) == 1
        assert 0.0 < result[0] < 1.0

    def test_batch_normalize_multiple(self):
        norm = ScoreNormalizer(k=0.1)
        scores = [1.0, 10.0, 50.0]
        result = norm.normalize_batch(scores)
        assert len(result) == 3
        assert all(0.0 <= s <= 1.0 for s in result)
        # Higher raw score → higher normalized score
        assert result[0] < result[1] < result[2]

    def test_batch_normalize_empty(self):
        norm = ScoreNormalizer(k=0.1)
        assert norm.normalize_batch([]) == []

    def test_batch_normalize_identical_scores(self):
        norm = ScoreNormalizer(k=0.1)
        result = norm.normalize_batch([5.0, 5.0, 5.0])
        assert len(result) == 3
        assert all(s == result[0] for s in result)

    def test_custom_k_parameter(self):
        """Higher k → steeper curve → scores closer to extremes."""
        norm_flat = ScoreNormalizer(k=0.01)
        norm_steep = ScoreNormalizer(k=1.0)
        # With small k, mid-score stays near 0.5
        mid_flat = norm_flat.normalize(5.0)
        mid_steep = norm_steep.normalize(5.0)
        # Steeper k pushes 5.0 closer to 1.0
        assert mid_steep > mid_flat


# ═══════════════════════════════════════════════════════════════
# KnowledgeBaseMapping
# ═══════════════════════════════════════════════════════════════


class TestKnowledgeBaseMapping:
    """Tests for knowledge_id → index_path mapping."""

    def _write_config(self, data: dict) -> Path:
        """Write a temp JSON config and return its path."""
        f = NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, f)
        f.close()
        return Path(f.name)

    def test_load_and_resolve(self):
        path = self._write_config({
            "test-kb": {
                "index_path": "/tmp/test-index",
                "raw_dir": "/tmp/test-raw",
                "description": "Test KB",
            }
        })
        try:
            mapping = KnowledgeBaseMapping(config_path=path)
            config = mapping.resolve("test-kb")
            assert config.index_path == Path("/tmp/test-index")
            assert config.raw_dir == Path("/tmp/test-raw")
            assert config.description == "Test KB"
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_nonexistent_kb(self):
        path = self._write_config({})
        try:
            mapping = KnowledgeBaseMapping(config_path=path)
            with pytest.raises(KeyError, match="nonexistent"):
                mapping.resolve("nonexistent")
        finally:
            path.unlink(missing_ok=True)

    def test_load_without_raw_dir(self):
        path = self._write_config({
            "kb-only-index": {"index_path": "/tmp/idx"}
        })
        try:
            mapping = KnowledgeBaseMapping(config_path=path)
            config = mapping.resolve("kb-only-index")
            assert config.index_path == Path("/tmp/idx")
            assert config.raw_dir is None
            assert config.description == ""
        finally:
            path.unlink(missing_ok=True)

    def test_list_knowledge_bases(self):
        path = self._write_config({"kb1": {"index_path": "/a"}, "kb2": {"index_path": "/b"}})
        try:
            mapping = KnowledgeBaseMapping(config_path=path)
            assert set(mapping.list_knowledge_bases()) == {"kb1", "kb2"}
        finally:
            path.unlink(missing_ok=True)

    def test_contains(self):
        path = self._write_config({"kb1": {"index_path": "/a"}})
        try:
            mapping = KnowledgeBaseMapping(config_path=path)
            assert "kb1" in mapping
            assert "kb2" not in mapping
        finally:
            path.unlink(missing_ok=True)

    def test_load_invalid_json_raises(self):
        """Non-JSON content should raise."""
        bad = NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        bad.write("not valid json {{{")
        bad.close()
        path = Path(bad.name)
        try:
            with pytest.raises((json.JSONDecodeError, KeyError)):
                KnowledgeBaseMapping(config_path=path)
        finally:
            path.unlink(missing_ok=True)

    def test_load_missing_index_path_raises(self):
        path = self._write_config({"kb1": {"no_index_path_here": "oops"}})
        try:
            with pytest.raises(KeyError):
                KnowledgeBaseMapping(config_path=path)
        finally:
            path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# MetadataConditionEvaluator
# ═══════════════════════════════════════════════════════════════


class TestMetadataConditionEvaluator:
    """Tests for metadata_condition filtering."""

    def _cond(self, name: str, op: str, value=None) -> MetadataCondition:
        return MetadataCondition(
            logical_operator="and",
            conditions=[MetadataConditionItem(name=name, comparison_operator=op, value=value)],
        )

    def _match(self, name: str, op: str, value, metadata: dict) -> bool:
        return MetadataConditionEvaluator.evaluate(self._cond(name, op, value), metadata)

    def test_is_match(self):
        assert self._match("category", "is", "policy", {"category": "policy"})

    def test_is_no_match(self):
        assert not self._match("category", "is", "policy", {"category": "guide"})

    def test_is_case_insensitive(self):
        assert self._match("name", "is", "POLICY", {"name": "Policy"})

    def test_contains_match(self):
        assert self._match("title", "contains", "年假", {"title": "年假管理制度"})

    def test_contains_no_match(self):
        assert not self._match("title", "contains", "病假", {"title": "年假管理制度"})

    def test_not_contains_match(self):
        assert self._match("title", "not contains", "病假", {"title": "年假制度"})

    def test_equals_numeric(self):
        assert self._match("count", "=", 5, {"count": 5})
        assert self._match("count", "=", "5", {"count": 5})

    def test_not_equals(self):
        assert self._match("count", "!=", 3, {"count": 5})

    def test_empty_match(self):
        assert MetadataConditionEvaluator.evaluate(
            MetadataCondition(
                logical_operator="and",
                conditions=[MetadataConditionItem(name="missing", comparison_operator="empty")],
            ),
            {},
        )

    def test_empty_no_match(self):
        assert not self._match("name", "empty", None, {"name": "value"})

    def test_not_empty_match(self):
        assert self._match("name", "not empty", None, {"name": "value"})

    def test_not_empty_no_match(self):
        assert not MetadataConditionEvaluator.evaluate(
            MetadataCondition(
                logical_operator="and",
                conditions=[MetadataConditionItem(name="missing", comparison_operator="not empty")],
            ),
            {},
        )

    def test_and_all_match(self):
        assert MetadataConditionEvaluator.evaluate(
            MetadataCondition(
                logical_operator="and",
                conditions=[
                    MetadataConditionItem(name="type", comparison_operator="is", value="policy"),
                    MetadataConditionItem(name="status", comparison_operator="is", value="active"),
                ],
            ),
            {"type": "policy", "status": "active"},
        )

    def test_and_one_fails(self):
        assert not MetadataConditionEvaluator.evaluate(
            MetadataCondition(
                logical_operator="and",
                conditions=[
                    MetadataConditionItem(name="type", comparison_operator="is", value="policy"),
                    MetadataConditionItem(name="status", comparison_operator="is", value="draft"),
                ],
            ),
            {"type": "policy", "status": "active"},
        )

    def test_or_one_matches(self):
        assert MetadataConditionEvaluator.evaluate(
            MetadataCondition(
                logical_operator="or",
                conditions=[
                    MetadataConditionItem(name="type", comparison_operator="is", value="policy"),
                    MetadataConditionItem(name="status", comparison_operator="is", value="draft"),
                ],
            ),
            {"type": "policy", "status": "active"},
        )

    def test_empty_conditions_always_match(self):
        assert MetadataConditionEvaluator.evaluate(
            MetadataCondition(logical_operator="and", conditions=[]), {},
        )

    def test_unsupported_operator_passes_through(self):
        """Unsupported operators should not block results."""
        assert self._match("field", "start with", "abc", {"field": "abcdef"})

    # ── Extended operators ──

    def test_is_not(self):
        assert self._match("status", "is not", "draft", {"status": "active"})
        assert not self._match("status", "is not", "active", {"status": "active"})

    def test_start_with(self):
        assert self._match("path", "start with", "HR/", {"path": "HR/年假制度.md"})
        assert not self._match("path", "start with", "IT/", {"path": "HR/年假制度.md"})

    def test_end_with(self):
        assert self._match("path", "end with", ".md", {"path": "doc.md"})
        assert not self._match("path", "end with", ".txt", {"path": "doc.md"})

    def test_in_list(self):
        assert self._match("category", "in", ["policy", "guide"], {"category": "policy"})
        assert not self._match("category", "in", ["draft", "archived"], {"category": "policy"})

    def test_in_comma_string(self):
        """'in' with a string value should treat it as comma-separated."""
        assert self._match("tag", "in", "hr,policy,guide", {"tag": "policy"})

    def test_not_in(self):
        assert self._match("category", "not in", ["draft"], {"category": "policy"})
        assert not self._match("category", "not in", ["policy"], {"category": "policy"})

    def test_greater_than(self):
        assert self._match("count", ">", 3, {"count": 5})
        assert not self._match("count", ">", 10, {"count": 5})

    def test_less_than(self):
        assert self._match("count", "<", 10, {"count": 5})
        assert not self._match("count", "<", 3, {"count": 5})

    def test_greater_equal(self):
        assert self._match("count", ">=", 5, {"count": 5})
        assert self._match("count", ">=", 4, {"count": 5})

    def test_less_equal(self):
        assert self._match("count", "<=", 5, {"count": 5})
        assert self._match("count", "<=", 6, {"count": 5})

    def test_before_date(self):
        """ISO date string comparison."""
        assert self._match("date", "before", "2024-07", {"date": "2024-01"})

    def test_after_date(self):
        assert self._match("date", "after", "2024-01", {"date": "2024-07"})

    def test_missing_field_with_not_empty(self):
        """Field not in metadata → not_empty should be False."""
        assert not MetadataConditionEvaluator.evaluate(
            MetadataCondition(
                logical_operator="and",
                conditions=[MetadataConditionItem(name="absent", comparison_operator="not empty")],
            ),
            {},
        )


# ═══════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════


class TestDifyModels:
    """Tests for Dify Pydantic request/response models."""

    def test_retrieval_request_minimal(self):
        req = DifyRetrievalRequest(knowledge_id="kb1", query="test")
        assert req.knowledge_id == "kb1"
        assert req.query == "test"
        assert req.retrieval_setting.top_k == 3
        assert req.retrieval_setting.score_threshold == 0.0
        assert req.metadata_condition is None

    def test_retrieval_request_full(self):
        req = DifyRetrievalRequest(
            knowledge_id="kb1",
            query="test query",
            retrieval_setting=DifyRetrievalSetting(top_k=5, score_threshold=0.5),
            metadata_condition=MetadataCondition(
                logical_operator="and",
                conditions=[MetadataConditionItem(name="type", comparison_operator="is", value="doc")],
            ),
        )
        assert req.retrieval_setting.top_k == 5
        assert req.retrieval_setting.score_threshold == 0.5
        assert req.metadata_condition.conditions[0].name == "type"

    def test_top_k_bounds(self):
        """top_k must be 1-100."""
        with pytest.raises(Exception):
            DifyRetrievalSetting(top_k=0)
        with pytest.raises(Exception):
            DifyRetrievalSetting(top_k=101)
        # These should work
        DifyRetrievalSetting(top_k=1)
        DifyRetrievalSetting(top_k=100)

    def test_score_threshold_bounds(self):
        with pytest.raises(Exception):
            DifyRetrievalSetting(score_threshold=-0.1)
        with pytest.raises(Exception):
            DifyRetrievalSetting(score_threshold=1.1)
        DifyRetrievalSetting(score_threshold=0.0)
        DifyRetrievalSetting(score_threshold=1.0)

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            DifyRetrievalRequest(knowledge_id="kb1", query="")

    def test_record_metadata_defaults_to_dict(self):
        record = DifyRecord(content="text", score=0.5, title="title")
        assert record.metadata == {}

    def test_response_defaults_to_empty_records(self):
        resp = DifyRetrievalResponse()
        assert resp.records == []

    def test_response_with_records(self):
        resp = DifyRetrievalResponse(
            records=[
                DifyRecord(content="c1", score=0.9, title="t1", metadata={"k": "v"}),
                DifyRecord(content="c2", score=0.5, title="t2"),
            ]
        )
        assert len(resp.records) == 2
        assert resp.records[0].metadata == {"k": "v"}

    def test_error_response_format(self):
        err = DifyErrorResponse(error_code=1002, error_msg="Auth failed")
        data = err.model_dump()
        assert data == {"error_code": 1002, "error_msg": "Auth failed"}


# ═══════════════════════════════════════════════════════════════
# /retrieval Endpoint — Smoke Tests
# ═══════════════════════════════════════════════════════════════


class TestRetrievalEndpoint:
    """Smoke tests for the /retrieval endpoint registration and error handling."""

    @pytest.fixture(autouse=True)
    def _reset_singletons(self):
        """Reset module-level singletons between tests."""
        import src.web.dify_retrieval as mod
        mod._kb_mapping = None
        mod._retrieval_service = None

    def test_endpoint_registered(self, monkeypatch):
        """Verify /retrieval endpoint exists and responds with structured error."""
        monkeypatch.delenv("DIFY_API_KEY", raising=False)
        # Without config, this will return an app-level 404 (KB not found),
        # NOT a FastAPI route-not-found 404. The distinction is the JSON body.
        r = client.post("/retrieval", json={
            "knowledge_id": "nonexistent",
            "query": "test",
        })
        # App-level 404 = route exists, just KB unknown
        # Route-not-found 404 = endpoint not registered at all
        body = r.json()
        assert "error_code" in body  # structured error → route exists
        assert body["error_code"] == DifyErrorCode.KNOWLEDGE_BASE_NOT_FOUND

    def test_auth_required_when_dify_key_set(self, monkeypatch):
        """When DIFY_API_KEY is set, unauthenticated requests get 401."""
        monkeypatch.setenv("DIFY_API_KEY", "secret-key")
        # Reset module to re-read env
        import src.web.dify_retrieval as mod
        mod._kb_mapping = None
        mod._retrieval_service = None

        r = client.post("/retrieval", json={
            "knowledge_id": "test",
            "query": "test",
        })
        assert r.status_code == 401
        body = r.json()
        assert body["error_code"] == DifyErrorCode.AUTH_FAILED
        assert "Authorization" in body["error_msg"] or "API key" in body["error_msg"].lower()

    def test_auth_bypass_when_dify_key_unset(self, monkeypatch):
        """When DIFY_API_KEY is unset, endpoint is open."""
        monkeypatch.delenv("DIFY_API_KEY", raising=False)
        import src.web.dify_retrieval as mod
        mod._kb_mapping = None
        mod._retrieval_service = None

        r = client.post("/retrieval", json={
            "knowledge_id": "nonexistent",
            "query": "test",
        })
        # Should bypass auth (but still fail on unknown KB)
        assert r.status_code == 404  # knowledge not found

    def test_invalid_auth_format(self, monkeypatch):
        """Wrong Bearer token gets 401."""
        monkeypatch.setenv("DIFY_API_KEY", "correct-key")
        import src.web.dify_retrieval as mod
        mod._kb_mapping = None
        mod._retrieval_service = None

        r = client.post(
            "/retrieval",
            json={"knowledge_id": "test", "query": "test"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401

    def test_valid_auth_passed(self, monkeypatch):
        """Valid Bearer token passes auth (then fails on KB not found)."""
        monkeypatch.setenv("DIFY_API_KEY", "correct-key")
        import src.web.dify_retrieval as mod
        mod._kb_mapping = None
        mod._retrieval_service = None

        r = client.post(
            "/retrieval",
            json={"knowledge_id": "nonexistent", "query": "test"},
            headers={"Authorization": "Bearer correct-key"},
        )
        # Auth passes → 404 for unknown KB
        assert r.status_code == 404

    def test_empty_query_returns_400(self, monkeypatch):
        """Empty query should be rejected by validation."""
        monkeypatch.delenv("DIFY_API_KEY", raising=False)

        r = client.post("/retrieval", json={
            "knowledge_id": "test",
            "query": "",
        })
        assert r.status_code == 400  # validation error

    def test_invalid_body_returns_400(self, monkeypatch):
        """Non-JSON body should be handled gracefully."""
        monkeypatch.delenv("DIFY_API_KEY", raising=False)

        r = client.post(
            "/retrieval",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400
        body = r.json()
        assert "error_code" in body

    def test_unknown_kb_returns_structured_error(self, monkeypatch):
        """Unknown knowledge_id returns 404 with error_code=2001."""
        monkeypatch.delenv("DIFY_API_KEY", raising=False)

        r = client.post("/retrieval", json={
            "knowledge_id": "nonexistent-kb-xyz",
            "query": "test query",
        })
        assert r.status_code == 404
        body = r.json()
        assert body["error_code"] == DifyErrorCode.KNOWLEDGE_BASE_NOT_FOUND
        assert "nonexistent-kb-xyz" in body["error_msg"]


# ═══════════════════════════════════════════════════════════════
# Content Mode (DIFY_CONTENT_MODE env var)
# ═══════════════════════════════════════════════════════════════


class TestContentMode:
    """Tests for DIFY_CONTENT_MODE env var handling."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("DIFY_CONTENT_MODE", raising=False)

    def _resolve(self):
        from src.web.dify_retrieval import DifyRetrievalService
        svc = DifyRetrievalService.__new__(DifyRetrievalService)
        return svc._resolve_content_mode()

    def test_default_mode_is_snippet(self):
        assert self._resolve() == "snippet"

    def test_explicit_snippet(self, monkeypatch):
        monkeypatch.setenv("DIFY_CONTENT_MODE", "snippet")
        assert self._resolve() == "snippet"

    def test_first_500(self, monkeypatch):
        monkeypatch.setenv("DIFY_CONTENT_MODE", "first_500")
        assert self._resolve() == "first_500"

    def test_first_1000(self, monkeypatch):
        monkeypatch.setenv("DIFY_CONTENT_MODE", "first_1000")
        assert self._resolve() == "first_1000"

    def test_full(self, monkeypatch):
        monkeypatch.setenv("DIFY_CONTENT_MODE", "full")
        assert self._resolve() == "full"

    def test_invalid_falls_back_to_snippet(self, monkeypatch):
        monkeypatch.setenv("DIFY_CONTENT_MODE", "invalid_mode_xyz")
        assert self._resolve() == "snippet"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DIFY_CONTENT_MODE", "FULL")
        assert self._resolve() == "full"
