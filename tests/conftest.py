"""
Pytest configuration and shared fixtures for doc-search tests.
"""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
import pytest

# ── Global test environment ────────────────────────────────────────────
# Disable SearchLogger during tests to prevent leakage into production
# D:/docs/search_logs/ directory. Individual tests that need logging
# (e.g. test_search_logger.py) re-enable it via monkeypatch.
os.environ.setdefault("NO_SEARCH_LOG", "1")
# Disable authentication during tests — test endpoints don't send API keys
os.environ["PI_FORCE_AUTH"] = "0"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory for test outputs.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path object for the temporary output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_config():
    """
    Provide mock configuration values for testing.

    Returns:
        SimpleNamespace with test configuration values
    """
    return SimpleNamespace(
        index_dir=Path("test_index"),
        output_dir=Path("test_output"),
        collection_name="test_collection",
        chunk_size=1000,
        chunk_overlap=200,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture(autouse=True)
def _clear_searcher_pool():
    """Clear SearcherPool cache between tests to prevent cross-test pollution."""
    from src.search.multi_index import SearcherPool
    SearcherPool.clear()
    yield
    SearcherPool.clear()


@pytest.fixture
def venv_python() -> str:
    """
    Return the path to the virtual environment Python executable.

    Returns:
        String path to the .venv Python executable
    """
    import sys
    root = Path(__file__).resolve().parent.parent
    if sys.platform == "win32":
        return str(root / ".venv" / "Scripts" / "python.exe")
    else:
        return str(root / ".venv" / "bin" / "python")
