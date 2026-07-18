"""Pytest configuration and shared fixtures for doc-search-lite tests."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.utils.config import Config

# ── Global test environment ────────────────────────────────────────────
# Disable SearchLogger during tests to prevent leakage.
os.environ.setdefault("NO_SEARCH_LOG", "1")


@pytest.fixture
def mock_config():
    """Create a mock Config with sensible defaults for most tests.

    Uses MagicMock(spec=Config) for type-safe attribute access.
    Override specific attributes after retrieval in each test.

    Usage:
        def test_something(mock_config):
            llm = LLMClient(config=mock_config)
    """
    config = MagicMock(spec=Config)
    config.active_api_key = "test-key"
    config.active_base_url = "http://test"
    config.llm_temperature = 0.7
    config.llm_max_tokens = 500
    config.litellm_model = "test-model"
    config.deepseek_api_key = ""
    config.glm_api_key = ""
    config.llm_provider = "glm"
    config.fast_model = "deepseek/deepseek-v4-flash"
    config.power_model = "deepseek/deepseek-v4-pro"
    return config


@pytest.fixture(autouse=True)
def _clear_searcher_pool():
    """Clear SearcherPool cache between tests to prevent cross-test pollution."""
    from src.search.multi_index import SearcherPool

    SearcherPool.clear()
    yield
    SearcherPool.clear()


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory under tmp_path."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def _env_snapshot():
    """Snapshot and restore os.environ around each test.

    Prevents env var mutations from leaking between tests.
    Applied automatically to all tests in the suite.
    """
    snapshot = dict(os.environ)
    yield
    for key in list(os.environ.keys()):
        if key not in snapshot:
            os.environ.pop(key, None)
    for key, val in snapshot.items():
        if os.environ.get(key) != val:
            os.environ[key] = val
