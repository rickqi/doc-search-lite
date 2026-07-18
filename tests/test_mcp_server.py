"""Tests for src/mcp_server.py — MCP Server tool wrappers.

Tests verify:
- Server creation and tool registration
- Result formatting helpers
- Error handling (missing index_path, unknown mode)
- Searcher/agent caching behavior
- Environment variable defaults
- _load_dotenv() cross-platform .env loading
- Cross-platform path resolution (Windows/Linux)

All external services (BM25, LLM) are mocked.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import src.mcp_server as mcp_mod
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

pytestmark = pytest.mark.skipif(not _HAS_MCP, reason="mcp package not installed")


# ── Helper fixtures ────────────────────────────────────────────

@pytest.fixture
def mock_searcher():
    """Mock BM25Searcher with fake results."""
    searcher = MagicMock()

    # Mock PaginatedResults
    preview = MagicMock()
    preview.title = "Annual Leave Policy"
    preview.doc_id = "abc123"
    preview.score = 0.95
    preview.snippet = "Employees are entitled to 15 days of annual leave..."
    preview.source_path = "HR/leave_policy.md"

    result = MagicMock()
    result.results = [preview]
    result.total = 1
    result.execution_time = 35.0

    searcher.search.return_value = result

    # Mock get_full_content
    full = MagicMock()
    full.title = "Annual Leave Policy"
    full.source_path = "HR/leave_policy.md"
    full.full_content = "# Annual Leave Policy\n\nLine 1\nLine 2\nLine 3"
    searcher.get_full_content.return_value = full

    return searcher


@pytest.fixture
def mock_agent_response():
    """Mock AgentResponse."""
    resp = MagicMock()
    resp.success = True
    resp.answer = "Employees get 15 days of annual leave per year."
    resp.search_hits = [
        {"title": "Leave Policy", "score": 0.95, "source_path": "HR/leave.md"}
    ]
    resp.tokens_used = 1500
    resp.processing_time = 5.2
    return resp


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear module-level caches before each test."""
    mcp_mod._bm25_cache.clear()
    mcp_mod._hybrid_cache.clear()
    mcp_mod._agent_cache.clear()
    global _config_cache
    mcp_mod._config_cache = None
    yield
    mcp_mod._bm25_cache.clear()
    mcp_mod._hybrid_cache.clear()
    mcp_mod._agent_cache.clear()
    mcp_mod._config_cache = None


# ── Server creation tests ─────────────────────────────────────

class TestServerCreation:
    def test_create_server_returns_server(self):
        """Server creation should succeed when mcp is available."""
        server = mcp_mod.create_server()
        assert server is not None

    def test_create_server_without_mcp_raises(self):
        """Should raise ImportError if mcp not available."""
        with patch.object(mcp_mod, "_MCP_AVAILABLE", False):
            with pytest.raises(ImportError, match="mcp package not installed"):
                mcp_mod.create_server()


# ── Resolve helpers tests ──────────────────────────────────────

class TestResolveHelpers:
    def test_resolve_index_from_param(self):
        result = mcp_mod._resolve_index("/path/to/index")
        assert result == "/path/to/index"

    def test_resolve_index_from_env(self):
        with patch.object(mcp_mod, "DEFAULT_INDEX", "/env/index"):
            result = mcp_mod._resolve_index("")
            assert result == "/env/index"

    def test_resolve_index_missing_raises(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(mcp_mod, "DEFAULT_INDEX", ""):
            with pytest.raises(ValueError, match="No index_path"):
                mcp_mod._resolve_index("")

    def test_resolve_raw_from_param(self):
        result = mcp_mod._resolve_raw("/path/to/raw")
        assert result == "/path/to/raw"

    def test_resolve_raw_missing_raises(self):
        with patch.object(mcp_mod, "DEFAULT_RAW", ""), pytest.raises(ValueError, match="No raw_dir"):
            mcp_mod._resolve_raw("")


# ── Format helpers tests ──────────────────────────────────────

class TestFormatHelpers:
    def test_format_search_results_with_data(self, mock_searcher):
        """Should format search results with title, score, snippet."""
        results = mock_searcher.search.return_value
        text = mcp_mod._format_search_results(results, limit=10)

        assert "Found 1 results" in text
        assert "Annual Leave Policy" in text
        assert "0.950" in text
        assert "abc123" in text

    def test_format_search_results_empty(self):
        """Should handle empty results gracefully."""
        results = MagicMock()
        results.results = []
        results.total = 0
        results.execution_time = 5.0

        text = mcp_mod._format_search_results(results, limit=10)
        assert "Found 0 results" in text

    def test_format_agent_response_success(self, mock_agent_response):
        """Should format agent response with answer and sources."""
        text = mcp_mod._format_agent_response(mock_agent_response)
        assert "15 days of annual leave" in text
        assert "Leave Policy" in text
        assert "Tokens: 1,500" in text

    def test_format_agent_response_failure(self):
        """Should format error response."""
        resp = MagicMock()
        resp.success = False
        resp.error = "LLM timeout"
        text = mcp_mod._format_agent_response(resp)
        assert "Search failed: LLM timeout" in text


# ── Tool execution tests (mocked) ─────────────────────────────

class TestDocSearchTool:
    def test_bm25_search_calls_searcher(self, mock_searcher):
        """doc_search bm25 mode should call searcher.search()."""
        with patch.object(mcp_mod, "_get_searcher", return_value=mock_searcher):
            result = mock_searcher.search("test query", limit=5)
            text = mcp_mod._format_search_results(result, 5)
            assert "Annual Leave Policy" in text

    def test_unknown_mode_returns_error(self):
        """Unknown search mode should return error message."""
        mode = "invalid"
        assert mode not in ("bm25", "hybrid", "grep")


class TestCaching:
    def test_searcher_cached_per_index(self, mock_searcher):
        """Same index_path should return cached searcher."""
        with patch("src.search.bm25_search.create_searcher", return_value=mock_searcher):
            s1 = mcp_mod._get_searcher("/idx1")
            s2 = mcp_mod._get_searcher("/idx1")
            assert s1 is s2  # Same object

    def test_different_indices_different_searchers(self, mock_searcher):
        """Different index_paths should create different searchers."""
        mock2 = MagicMock()
        call_count = [0]
        def fake_create(index_path=None, **kw):
            call_count[0] += 1
            return mock_searcher if call_count[0] == 1 else mock2

        with patch("src.search.bm25_search.create_searcher", side_effect=fake_create):
            s1 = mcp_mod._get_searcher("/idx1")
            s2 = mcp_mod._get_searcher("/idx2")
            assert s1 is not s2


# ── Environment variable tests ─────────────────────────────────

class TestEnvironmentDefaults:
    def test_default_index_from_env(self):
        """DEFAULT_INDEX should be set from DOC_SEARCH_INDEX env var at import time."""
        # Already loaded — just verify it's a string
        assert isinstance(mcp_mod.DEFAULT_INDEX, str)

    def test_default_raw_from_env(self):
        """DEFAULT_RAW should be set from DOC_SEARCH_RAW env var at import time."""
        assert isinstance(mcp_mod.DEFAULT_RAW, str)


# ── Multi-index handling tests ─────────────────────────────────

class TestMultiIndex:
    def test_comma_separated_uses_first_for_searcher(self, mock_searcher):
        """Comma-separated index_path should use first index for BM25."""
        with patch("src.search.bm25_search.create_searcher", return_value=mock_searcher) as mock_create:
            mcp_mod._get_searcher("/idx1,/idx2,/idx3")
            # Should be called with the first index only
            called_path = mock_create.call_args.kwargs.get("index_path")
            assert str(called_path).replace("\\", "/") == "/idx1"

    def test_get_all_searchers_creates_one_per_index(self, mock_searcher):
        """_get_all_searchers should create one searcher per comma-separated index."""
        with patch("src.search.bm25_search.create_searcher", return_value=mock_searcher):
            searchers = mcp_mod._get_all_searchers("/idx1,/idx2,/idx3")
            assert len(searchers) == 3


# ── Auto-discovery tests ───────────────────────────────────────

class TestAutoDiscovery:
    def test_discover_indexes_finds_convert_db_dirs(self, tmp_path):
        """discover_indexes should find dirs containing convert.db."""
        # Create fake knowledge bases
        kb1 = tmp_path / "kb1"
        kb1.mkdir()
        (kb1 / "convert.db").touch()
        (kb1 / "index").mkdir()

        kb2 = tmp_path / "kb2"
        kb2.mkdir()
        (kb2 / "convert.db").touch()
        (kb2 / "index").mkdir()

        # Non-kb dir (no convert.db)
        (tmp_path / "not_a_kb").mkdir()

        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        assert len(indexes) == 2
        assert len(raw_dirs) == 2
        assert all("index" in i for i in indexes)

    def test_discover_indexes_nested_dirs(self, tmp_path):
        """discover_indexes should find nested subdirs with convert.db."""
        parent = tmp_path / "collection"
        parent.mkdir()
        child = parent / "sub_kb"
        child.mkdir()
        (child / "convert.db").touch()
        (child / "index").mkdir()

        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        assert len(indexes) == 1

    def test_discover_indexes_empty_root(self, tmp_path):
        """discover_indexes should return empty lists for empty root."""
        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        assert indexes == []
        assert raw_dirs == []

    def test_discover_indexes_nonexistent_root(self):
        """discover_indexes should return empty for nonexistent root."""
        indexes, raw_dirs = mcp_mod.discover_indexes("/nonexistent/path/12345")
        assert indexes == []
        assert raw_dirs == []

    def test_discover_indexes_lists_are_aligned(self, tmp_path):
        """indexes and raw_dirs must have equal length (1:1 correspondence)."""
        for name in ("kb1", "kb2", "kb3"):
            kb = tmp_path / name
            kb.mkdir()
            (kb / "convert.db").touch()
            (kb / "index").mkdir()

        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        assert len(indexes) == len(raw_dirs), \
            f"Lists misaligned: {len(indexes)} indexes vs {len(raw_dirs)} raw_dirs"

    def test_discover_indexes_excludes_dirs_without_index(self, tmp_path):
        """Dirs with convert.db but NO index/ must be excluded from both lists."""
        # Good KB: has both convert.db and index/
        good = tmp_path / "good_kb"
        good.mkdir()
        (good / "convert.db").touch()
        (good / "index").mkdir()

        # Bad KB: has convert.db but NO index/
        bad = tmp_path / "bad_kb"
        bad.mkdir()
        (bad / "convert.db").touch()

        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        assert len(indexes) == 1
        assert len(raw_dirs) == 1
        assert "good_kb" in raw_dirs[0]
        assert "good_kb" in indexes[0]
        assert "bad_kb" not in str(raw_dirs)
        assert "bad_kb" not in str(indexes)

    def test_discover_indexes_index_raw_pairing(self, tmp_path):
        """Each index[i] must be the index/ subdir of raw_dirs[i]."""
        for name in ("alpha", "beta"):
            kb = tmp_path / name
            kb.mkdir()
            (kb / "convert.db").touch()
            (kb / "index").mkdir()

        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        for idx_path, raw_path in zip(indexes, raw_dirs):
            # The index path should be raw_path + "/index"
            assert Path(idx_path).parent.resolve() == Path(raw_path).resolve()

    def test_discover_indexes_deduplicates(self, tmp_path):
        """A directory appearing at multiple levels should not be added twice."""
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "convert.db").touch()
        (kb / "index").mkdir()

        # Create a nested structure where kb might be scanned twice
        # (parent dir scan + nested scan)
        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        assert len(indexes) == 1
        assert len(raw_dirs) == 1

    def test_discover_indexes_mixed_validity(self, tmp_path):
        """Mix of valid (has index), invalid (no index), and non-kb dirs."""
        # Valid KB
        kb1 = tmp_path / "kb1"
        kb1.mkdir()
        (kb1 / "convert.db").touch()
        (kb1 / "index").mkdir()

        # Invalid: convert.db but no index
        kb2 = tmp_path / "kb2"
        kb2.mkdir()
        (kb2 / "convert.db").touch()

        # Non-kb: no convert.db at all
        (tmp_path / "random").mkdir()

        # Nested valid KB
        parent = tmp_path / "collection"
        parent.mkdir()
        kb3 = parent / "kb3"
        kb3.mkdir()
        (kb3 / "convert.db").touch()
        (kb3 / "index").mkdir()

        indexes, raw_dirs = mcp_mod.discover_indexes(str(tmp_path))
        assert len(indexes) == 2
        assert len(raw_dirs) == 2
        # Both lists same length — the core invariant
        assert len(indexes) == len(raw_dirs)


# ── _load_dotenv() tests ────────────────────────────────────────

class TestLoadDotenv:
    """测试 _load_dotenv() 跨平台 .env 加载行为。"""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """每个测试前清理 GLM_API_KEY 等环境变量残留。"""
        saved = {}
        for key in ("GLM_API_KEY", "DEEPSEEK_API_KEY"):
            saved[key] = os.environ.pop(key, None)
        yield
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    def test_loads_env_file_when_present(self, tmp_path):
        """存在 .env 时正确加载环境变量。"""
        env_file = tmp_path / ".env"
        env_file.write_text("GLM_API_KEY=test_key_123\nDEEPSEEK_API_KEY=dk_test_456\n")

        with patch.object(mcp_mod.Path, "resolve", return_value=tmp_path / "src" / "mcp_server.py"):
            mcp_mod._load_dotenv()

        assert os.environ.get("GLM_API_KEY") == "test_key_123"
        assert os.environ.get("DEEPSEEK_API_KEY") == "dk_test_456"

    def test_silent_when_env_file_missing(self, tmp_path):
        """.env 文件不存在时静默跳过不报错。"""
        empty_dir = tmp_path / "no_env"
        empty_dir.mkdir()

        with patch.object(mcp_mod.Path, "resolve", return_value=empty_dir / "src" / "mcp_server.py"):
            mcp_mod._load_dotenv()

        # 不应抛出异常

    def test_silent_when_dotenv_not_installed(self, tmp_path):
        """python-dotenv 未安装时 ImportError 被静默捕获。"""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val")

        with patch.object(mcp_mod.Path, "resolve", return_value=tmp_path / "src" / "mcp_server.py"):
            with patch.dict("sys.modules", {"dotenv": None}):
                with patch("builtins.__import__", side_effect=ImportError("No module named 'dotenv'")):
                    mcp_mod._load_dotenv()

        # 不应抛出异常

    def test_override_false_respects_existing_env(self, tmp_path):
        """override=False 时不覆盖已存在的环境变量。"""
        os.environ["GLM_API_KEY"] = "preexisting_key"
        env_file = tmp_path / ".env"
        env_file.write_text("GLM_API_KEY=new_key_from_file\n")

        with patch.object(mcp_mod.Path, "resolve", return_value=tmp_path / "src" / "mcp_server.py"):
            mcp_mod._load_dotenv()

        assert os.environ["GLM_API_KEY"] == "preexisting_key"

    def test_path_resolution_uses_file_location(self, tmp_path):
        """验证使用 __file__ 位置（而非 cwd）解析项目根目录。

        无论在何目录运行，都应找到 mcp_server.py 所在项目的 .env。
        """
        project = tmp_path / "myproject"
        src_dir = project / "src"
        src_dir.mkdir(parents=True)
        env_file = project / ".env"
        env_file.write_text("GLM_API_KEY=found_by_resolve\n")

        mcp_file = src_dir / "mcp_server.py"
        with patch.object(mcp_mod.Path, "resolve", return_value=mcp_file):
            mcp_mod._load_dotenv()

        assert os.environ.get("GLM_API_KEY") == "found_by_resolve"

    def test_path_handles_windows_backslash(self, tmp_path):
        """Path 自动规范化 Windows 反斜杠路径。"""
        project = tmp_path / "project"
        project.mkdir()
        env_file = project / ".env"
        env_file.write_text("GLM_API_KEY=win_path_test\n")

        fake_file = project / "src" / "mcp_server.py"
        with patch.object(mcp_mod.Path, "resolve", return_value=fake_file):
            mcp_mod._load_dotenv()

        assert os.environ.get("GLM_API_KEY") == "win_path_test"

    def test_env_vars_loaded_before_server_creation(self):
        """验证 main() 在创建 server 之前调用 _load_dotenv。"""
        call_order = []

        with patch.object(mcp_mod, "_load_dotenv") as mock_load:
            mock_load.side_effect = lambda: call_order.append("load_dotenv")

            def tracking_create(**kwargs):
                call_order.append("create_server")
                return MagicMock()

            with patch.object(mcp_mod, "create_server", side_effect=tracking_create):
                # Prevent argparse from reading pytest's sys.argv
                with patch.object(mcp_mod.sys, "argv", ["mcp_server"]):
                    mcp_mod.main()

        assert "load_dotenv" in call_order
        load_idx = call_order.index("load_dotenv")
        create_idx = call_order.index("create_server")
        assert load_idx < create_idx, (
            f"load_dotenv ({load_idx}) must be called before create_server ({create_idx})"
        )


# ── Cross-platform _find_venv_python tests ──────────────────────

class TestFindVenvPython:
    """测试 run_mcp.py 的 _find_venv_python() 跨平台路径查找逻辑。

    验证：
    - Linux 查找 .venv/bin/python
    - Windows 查找 .venv/Scripts/python.exe
    - 不存在 venv 时回退到 sys.executable
    - Path 跨平台路径解析
    """

    @staticmethod
    def _find_venv_python(root: Path, platform: str = "linux") -> Path:
        """Replicate run_mcp._find_venv_python logic (avoiding execv side effects)."""
        if platform == "win32":
            candidates = [
                root / ".venv" / "Scripts" / "python.exe",
                root / ".venv" / "Scripts" / "python",
            ]
        else:
            candidates = [
                root / ".venv" / "bin" / "python",
                root / ".venv" / "bin" / "python3",
            ]
        for p in candidates:
            if p.exists():
                return p
        return Path("/usr/bin/python3")  # sys.executable fallback

    def test_linux_finds_bin_python(self, tmp_path):
        """Linux 下找到 .venv/bin/python。"""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        python_bin = venv_bin / "python"
        python_bin.touch()

        result = self._find_venv_python(tmp_path, platform="linux")
        assert result == python_bin

    def test_linux_finds_bin_python3_fallback(self, tmp_path):
        """Linux 下 python 不存在时找到 python3。"""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        python3_bin = venv_bin / "python3"
        python3_bin.touch()

        result = self._find_venv_python(tmp_path, platform="linux")
        assert result == python3_bin

    def test_windows_finds_python_exe(self, tmp_path):
        """Windows 下找到 .venv/Scripts/python.exe。"""
        venv_scripts = tmp_path / ".venv" / "Scripts"
        venv_scripts.mkdir(parents=True)
        python_exe = venv_scripts / "python.exe"
        python_exe.touch()

        result = self._find_venv_python(tmp_path, platform="win32")
        assert result == python_exe

    def test_windows_falls_back_to_python_bare(self, tmp_path):
        """Windows 下 python.exe 不存在时找到 python（无后缀）。"""
        venv_scripts = tmp_path / ".venv" / "Scripts"
        venv_scripts.mkdir(parents=True)
        python_bare = venv_scripts / "python"
        python_bare.touch()

        result = self._find_venv_python(tmp_path, platform="win32")
        assert result == python_bare

    def test_fallback_to_sys_executable(self, tmp_path):
        """不存在 .venv 时回退到 sys.executable。"""
        empty_dir = tmp_path / "no_venv"
        empty_dir.mkdir()

        result = self._find_venv_python(empty_dir, platform="linux")
        assert str(result).replace("\\", "/") == "/usr/bin/python3"

    def test_path_cross_platform_normalization(self, tmp_path):
        """Path 自动处理跨平台路径分隔符（无须额外处理）。"""
        sub = tmp_path / "deep" / "project"
        sub.mkdir(parents=True)
        venv_bin = sub / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").touch()

        result = self._find_venv_python(sub, platform="linux")
        assert result == venv_bin / "python"
        # Path 在不同平台自动使用正确的分隔符
        assert "/.venv/bin/python" in str(result).replace("\\", "/")
