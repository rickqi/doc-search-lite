"""CLI integration tests using Click CliRunner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import TaskManager, cli
from src.search.bm25_search import PaginatedResults, SearchPreview

# ── Fixtures ──────────────────────────────────────────


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_index_dir(tmp_path):
    """Create a minimal temp directory with .md files and an index/ subdirectory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc1.md").write_text("年假规定 员工入职满一年可享受年假", encoding="utf-8")
    (raw_dir / "doc2.md").write_text("采购流程 需要经理审批", encoding="utf-8")
    index_dir = raw_dir / "index"
    index_dir.mkdir()
    # Simulate Tantivy index by creating a .store segment file
    (index_dir / "fake.store").write_bytes(b"\x00")
    return raw_dir


def _make_paginated_results(query_text, results=None, total=None):
    """Helper to build a PaginatedResults for mocking."""
    if results is None:
        results = []
    if total is None:
        total = len(results)
    return PaginatedResults(
        results=results,
        total=total,
        offset=0,
        limit=10,
        has_more=False,
        query=query_text,
        execution_time=0.05,
    )


def _make_preview(doc_id="d1", title="Doc1", score=5.0, snippet="test snippet", source_path=None):
    """Helper to build a SearchPreview."""
    return SearchPreview(
        doc_id=doc_id,
        title=title,
        score=score,
        snippet=snippet,
        source_path=source_path,
        highlights=[],
    )


# ── Top-level CLI ─────────────────────────────────────


class TestCLIGroup:
    """Test the top-level CLI group."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "本地文档搜索系统" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        # Version output contains the version string
        assert "version" in result.output.lower() or "." in result.output


# ── query command ─────────────────────────────────────


class TestQueryCommand:
    """Test the query CLI command."""

    def test_query_help(self, runner):
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "搜索文档" in result.output

    def test_query_help_shows_options(self, runner):
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "--agent" in result.output
        assert "--rerank" in result.output
        assert "--index" in result.output or "-i" in result.output
        assert "--output-format" in result.output or "-f" in result.output
        assert "--limit" in result.output or "-l" in result.output
        assert "--interactive" in result.output or "-I" in result.output

    @patch("src.cli.create_searcher")
    def test_query_basic_search(self, mock_create, runner, temp_index_dir):
        """Test basic BM25 search via CLI."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "年假",
            results=[_make_preview(doc_id="d1", title="年假规定", score=5.0, snippet="年假规定", source_path=Path("doc1.md"))],
        )
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "年假", "-i", str(temp_index_dir / "index")])
        assert result.exit_code == 0
        assert "年假" in result.output

    @patch("src.cli.create_searcher")
    def test_query_no_results(self, mock_create, runner, temp_index_dir):
        """Test query that returns no results."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results("xyz")
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "xyz", "-i", str(temp_index_dir / "index")])
        assert result.exit_code == 0
        assert "未找到" in result.output

    def test_query_missing_index(self, runner, tmp_path):
        """Query with non-existent index should fail gracefully."""
        result = runner.invoke(cli, ["query", "test", "-i", str(tmp_path / "nonexistent")])
        # Should not crash with traceback
        assert "索引目录不存在" in result.output or result.exit_code != 0

    @patch("src.cli.create_searcher")
    def test_query_json_format(self, mock_create, runner, temp_index_dir):
        """Test --output-format json produces valid JSON."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "test",
            results=[_make_preview(doc_id="d1", title="Doc1", score=5.0, snippet="test content", source_path=Path("d.md"))],
        )
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index"), "-f", "json"])
        assert result.exit_code == 0
        # CLI appends timing info after JSON; extract the JSON portion
        json_text = result.output
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            # Find the end of the JSON object
            brace_depth = 0
            for i, ch in enumerate(json_text):
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        data = json.loads(json_text[: i + 1])
                        break
            else:
                raise
        assert "results" in data

    @patch("src.cli.create_searcher")
    def test_query_markdown_format(self, mock_create, runner, temp_index_dir):
        """Test --output-format markdown."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "test",
            results=[_make_preview(doc_id="d1", title="Doc1", score=5.0, snippet="test", source_path=Path("d.md"))],
        )
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index"), "-f", "markdown"])
        assert result.exit_code == 0

    @patch("src.cli.create_searcher")
    def test_query_text_format_default(self, mock_create, runner, temp_index_dir):
        """Test default text format shows score and snippet."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "年假",
            results=[_make_preview(doc_id="d1", title="年假规定", score=3.14, snippet="年假规定", source_path=Path("doc1.md"))],
        )
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "年假", "-i", str(temp_index_dir / "index")])
        assert result.exit_code == 0
        assert "3.14" in result.output
        assert "年假规定" in result.output

    @patch("src.cli.create_searcher")
    def test_query_with_limit(self, mock_create, runner, temp_index_dir):
        """Test --limit option is passed through."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results("test")
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index"), "-l", "5"])
        assert result.exit_code == 0
        mock_searcher.search.assert_called_once_with("test", limit=5)

    @patch("src.cli.create_searcher")
    def test_query_multiple_results(self, mock_create, runner, temp_index_dir):
        """Test query with multiple results."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "制度",
            results=[
                _make_preview(doc_id="d1", title="年假制度", score=8.0, snippet="年假制度内容", source_path=Path("doc1.md")),
                _make_preview(doc_id="d2", title="采购制度", score=5.0, snippet="采购制度内容", source_path=Path("doc2.md")),
            ],
            total=2,
        )
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "制度", "-i", str(temp_index_dir / "index")])
        assert result.exit_code == 0
        assert "年假制度" in result.output
        assert "采购制度" in result.output
        assert "2 条结果" in result.output

    @patch("src.cli.create_searcher")
    def test_query_search_exception(self, mock_create, runner, temp_index_dir):
        """Test that search exceptions are handled."""
        mock_searcher = MagicMock()
        mock_searcher.search.side_effect = RuntimeError("index corrupt")
        mock_create.return_value = mock_searcher

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index")])
        assert result.exit_code != 0
        assert "搜索失败" in result.output


# ── query --agent ─────────────────────────────────────


class TestQueryAgentMode:
    """Test the query --agent flag."""

    @patch("src.agent.search_agent.create_search_agent")
    @patch("src.utils.config.Config.from_env")
    def test_agent_basic(self, mock_config, mock_create_agent, runner, temp_index_dir):
        """Test --agent flag invokes SearchAgent."""
        mock_config.return_value = MagicMock(
            glm_api_key="test-key",
            glm_base_url="http://test",
            llm_model="glm-4",
        )
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.answer = "员工可申请年假。"
        mock_response.sources = ["doc1.md"]
        mock_response.tokens_used = 100
        mock_response.processing_time = 1.5
        mock_agent.run.return_value = mock_response
        mock_create_agent.return_value = mock_agent

        result = runner.invoke(cli, ["query", "年假如何申请", "-i", str(temp_index_dir / "index"), "--agent"])
        assert result.exit_code == 0
        assert "员工可申请年假" in result.output

    @patch("src.agent.search_agent.create_search_agent")
    @patch("src.utils.config.Config.from_env")
    def test_agent_json_output(self, mock_config, mock_create_agent, runner, temp_index_dir):
        """Test --agent with -f json."""
        mock_config.return_value = MagicMock(
            glm_api_key="test-key",
            glm_base_url="http://test",
            llm_model="glm-4",
        )
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.answer = "answer text"
        mock_response.sources = []
        mock_response.tokens_used = 50
        mock_response.processing_time = 0.5
        mock_agent.run.return_value = mock_response
        mock_create_agent.return_value = mock_agent

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index"), "--agent", "-f", "json"])
        assert result.exit_code == 0
        # Agent mode echoes "🤔 正在分析..." before JSON; find the JSON object
        json_start = result.output.index("{")
        data = json.loads(result.output[json_start:])
        assert "answer" in data

    @patch("src.utils.config.Config.from_env")
    def test_agent_missing_api_key(self, mock_config, runner, temp_index_dir):
        """Test --agent without API key shows error."""
        mock_config.side_effect = ValueError("GLM_API_KEY not set")

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index"), "--agent"])
        assert result.exit_code != 0
        assert "API密钥" in result.output or "GLM_API_KEY" in result.output

    @patch("src.agent.search_agent.create_search_agent")
    @patch("src.utils.config.Config.from_env")
    def test_agent_failure_response(self, mock_config, mock_create_agent, runner, temp_index_dir):
        """Test --agent when agent returns failure."""
        mock_config.return_value = MagicMock(
            glm_api_key="k", glm_base_url="http://t", llm_model="glm-4",
        )
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error = "Model unavailable"
        mock_agent.run.return_value = mock_response
        mock_create_agent.return_value = mock_agent

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index"), "--agent"])
        assert result.exit_code != 0
        assert "查询失败" in result.output

    @patch("src.agent.search_agent.create_search_agent")
    @patch("src.utils.config.Config.from_env")
    def test_agent_with_rerank(self, mock_config, mock_create_agent, runner, temp_index_dir):
        """Test --agent --rerank passes use_rerank=True."""
        mock_config.return_value = MagicMock(
            glm_api_key="k", glm_base_url="http://t", llm_model="glm-4",
        )
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.answer = "answer"
        mock_response.sources = []
        mock_response.tokens_used = 0
        mock_response.processing_time = 0.1
        mock_agent.run.return_value = mock_response
        mock_create_agent.return_value = mock_agent

        result = runner.invoke(cli, ["query", "test", "-i", str(temp_index_dir / "index"), "--agent", "--rerank"])
        assert result.exit_code == 0
        # Verify create_search_agent was called with use_rerank=True
        call_kwargs = mock_create_agent.call_args
        assert call_kwargs.kwargs.get("use_rerank") is True or call_kwargs[1].get("use_rerank") is True


# ── build-index command ──────────────────────────────


class TestBuildIndexCommand:
    """Test the build-index CLI command."""

    def test_build_index_help(self, runner):
        result = runner.invoke(cli, ["build-index", "--help"])
        assert result.exit_code == 0
        assert "RAW_DIR" in result.output or "raw_dir" in result.output.lower()

    def test_build_index_creates_index(self, runner, tmp_path):
        """Test that build-index creates an index from .md files."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc1.md").write_text("测试文档内容", encoding="utf-8")
        (raw / "doc2.md").write_text("另一个文档", encoding="utf-8")

        result = runner.invoke(cli, ["build-index", str(raw)])
        assert result.exit_code == 0
        assert (raw / "index").exists()
        assert "Indexed" in result.output or "2" in result.output

    def test_build_index_empty_dir(self, runner, tmp_path):
        """Empty directory should succeed (no files found)."""
        raw = tmp_path / "empty"
        raw.mkdir()
        result = runner.invoke(cli, ["build-index", str(raw)])
        assert result.exit_code == 0
        assert "No markdown files" in result.output

    def test_build_index_nonexistent_dir(self, runner):
        """Non-existent directory should fail."""
        result = runner.invoke(cli, ["build-index", "C:\\nonexistent_path_xyz"])
        assert result.exit_code != 0

    def test_build_index_with_md_files(self, runner, tmp_path):
        """Test build-index with multiple .md files creates correct doc count."""
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(5):
            (raw / f"doc{i}.md").write_text(f"文档内容 {i}", encoding="utf-8")

        result = runner.invoke(cli, ["build-index", str(raw)])
        assert result.exit_code == 0
        assert "5" in result.output

    def test_build_index_skips_underscore_prefix(self, runner, tmp_path):
        """Files starting with _ should be skipped."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc1.md").write_text("内容", encoding="utf-8")
        (raw / "_index.md").write_text("索引文件", encoding="utf-8")

        result = runner.invoke(cli, ["build-index", str(raw)])
        assert result.exit_code == 0
        assert "1" in result.output  # Only doc1.md indexed

    def test_build_index_replaces_existing(self, runner, tmp_path):
        """Running build-index twice should replace the old index."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc1.md").write_text("内容", encoding="utf-8")

        result1 = runner.invoke(cli, ["build-index", str(raw)])
        assert result1.exit_code == 0

        result2 = runner.invoke(cli, ["build-index", str(raw)])
        assert result2.exit_code == 0
        assert "Removing old index" in result2.output


# ── status command ────────────────────────────────────


class TestStatusCommand:
    """Test the status CLI command."""

    def test_status_help(self, runner):
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0

    def test_status_no_args(self, runner):
        """status without args uses ./output default."""
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0

    def test_status_with_path(self, runner, tmp_path):
        """status with an existing path."""
        (tmp_path / "test.md").write_text("test content", encoding="utf-8")
        result = runner.invoke(cli, ["status", str(tmp_path)])
        assert result.exit_code == 0
        assert "文档数量" in result.output
        assert "1" in result.output

    def test_status_with_detailed_flag(self, runner, tmp_path):
        """--detailed flag should be accepted."""
        (tmp_path / "doc.md").write_text("content", encoding="utf-8")
        result = runner.invoke(cli, ["status", str(tmp_path), "--detailed"])
        assert result.exit_code == 0

    def test_status_nonexistent_path(self, runner, tmp_path):
        """Non-existent path should fail (Click validates exists=True on the argument)."""
        bad = tmp_path / "nope"
        result = runner.invoke(cli, ["status", str(bad)])
        # Click validates the path exists before the command runs
        assert result.exit_code != 0

    def test_status_with_index(self, runner, tmp_path):
        """status when index/ exists shows index info."""
        (tmp_path / "doc.md").write_text("content", encoding="utf-8")
        (tmp_path / "index").mkdir()
        # Write a minimal tantivy meta.json so it doesn't crash
        import json as _json
        meta = {"segments": [], "schema": [], "opstamp": 0}
        (tmp_path / "index" / "meta.json").write_text(_json.dumps(meta), encoding="utf-8")

        result = runner.invoke(cli, ["status", str(tmp_path)])
        assert result.exit_code == 0
        assert "索引状态" in result.output


# ── convert command ───────────────────────────────────


class TestConvertCommand:
    """Test the convert CLI command."""

    def test_convert_help(self, runner):
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "源文档" in result.output or "转换" in result.output

    def test_convert_dry_run(self, runner, tmp_path):
        """convert --dry-run should list files without converting."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "doc.txt").write_text("hello", encoding="utf-8")
        output = tmp_path / "output"

        result = runner.invoke(cli, ["convert", str(source), "-o", str(output), "--dry-run"])
        assert result.exit_code == 0
        assert "模拟运行" in result.output
        assert "doc.txt" in result.output

    def test_convert_no_matching_files(self, runner, tmp_path):
        """convert with non-supported extension only shows not found."""
        source = tmp_path / "source"
        source.mkdir()
        # .crdownload is in SKIP_EXTENSIONS, not SUPPORTED_EXTENSIONS, so it gets skipped
        # and won't be collected by _collect_files when no formats specified
        # Actually _collect_files with formats=() matches ALL files.
        # Use --formats to filter to something that won't match
        (source / "data.xyz").write_text("xyz", encoding="utf-8")

        result = runner.invoke(cli, ["convert", str(source), "-o", str(tmp_path / "out"), "--formats", "pdf"])
        assert result.exit_code == 0
        assert "未找到匹配" in result.output

    @patch("src.cli.ConverterCoordinator")
    @patch("src.cli.MarkdownStore")
    @patch("src.cli.MetadataManager")
    def test_convert_single_file(self, MockMetaMgr, MockStore, MockCoord, runner, tmp_path):
        """Test convert with a single file and mocked dependencies."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
        output = tmp_path / "output"

        # Mock converter
        mock_coordinator = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "converted text"
        mock_result.converter_name = "PDFConverter"
        mock_result.convert_time = 0.5
        mock_result.ocr_used = False
        mock_result.metadata = {}
        mock_result.images = None
        mock_coordinator.convert.return_value = mock_result
        MockCoord.return_value = mock_coordinator

        # Mock store
        mock_store = MagicMock()
        mock_store.exists_by_source.return_value = False
        mock_store._generate_doc_id.return_value = "abc123"
        mock_store.get_output_path.return_value = output / "doc.pdf.md"
        MockStore.return_value = mock_store

        # Mock metadata
        MockMetaMgr.return_value = MagicMock()

        result = runner.invoke(cli, ["convert", str(source), "-o", str(output), "--no-index"])
        assert result.exit_code == 0
        assert "成功" in result.output


# ── batch-convert command ─────────────────────────────


class TestBatchConvertCommand:
    """Test the batch-convert CLI command."""

    def test_batch_convert_help(self, runner):
        result = runner.invoke(cli, ["batch-convert", "--help"])
        assert result.exit_code == 0
        assert "批量转换" in result.output

    def test_batch_convert_dry_run(self, runner, tmp_path):
        """batch-convert --dry-run with no files shows clean exit."""
        source = tmp_path / "source"
        source.mkdir()
        raw_root = tmp_path / "raw"

        result = runner.invoke(cli, [
            "batch-convert", str(source),
            "--raw-root", str(raw_root),
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "模拟运行" in result.output or "没有待处理" in result.output

    def test_batch_convert_with_files(self, runner, tmp_path):
        """batch-convert with a text file (end-to-end with real ConvertDB)."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "readme.txt").write_text("hello world", encoding="utf-8")
        raw_root = tmp_path / "raw"

        result = runner.invoke(cli, [
            "batch-convert", str(source),
            "--raw-root", str(raw_root),
            "--no-ocr",
        ])
        assert result.exit_code == 0
        assert "转换完成" in result.output
        # Verify DB was created
        assert (raw_root / source.name / "convert.db").exists()


# ── catalog command group ─────────────────────────────


class TestCatalogCommand:
    """Test the catalog CLI commands."""

    def test_catalog_help(self, runner):
        result = runner.invoke(cli, ["catalog", "--help"])
        assert result.exit_code == 0
        assert "转换目录管理" in result.output

    def test_catalog_status(self, runner, tmp_path):
        """catalog status with a real ConvertDB."""
        from src.storage.convert_db import ConvertDB

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        db_path = raw_dir / "convert.db"
        db = ConvertDB(db_path)
        db.open()
        db.close()

        result = runner.invoke(cli, ["catalog", "status", str(raw_dir)])
        assert result.exit_code == 0
        assert "转换状态" in result.output

    def test_catalog_status_no_db(self, runner, tmp_path):
        """catalog status without convert.db should show error."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        result = runner.invoke(cli, ["catalog", "status", str(raw_dir)])
        assert result.exit_code == 0
        assert "未找到转换数据库" in result.output

    def test_catalog_failed(self, runner, tmp_path):
        """catalog failed with a real ConvertDB."""
        from src.storage.convert_db import ConvertDB

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        db_path = raw_dir / "convert.db"
        db = ConvertDB(db_path)
        db.open()
        db.close()

        result = runner.invoke(cli, ["catalog", "failed", str(raw_dir)])
        assert result.exit_code == 0
        assert "没有失败文件" in result.output

    def test_catalog_failed_no_db(self, runner, tmp_path):
        """catalog failed without convert.db shows error."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        result = runner.invoke(cli, ["catalog", "failed", str(raw_dir)])
        assert result.exit_code == 0
        assert "未找到转换数据库" in result.output

    def test_catalog_token(self, runner, tmp_path):
        """catalog token with a real ConvertDB."""
        from src.storage.convert_db import ConvertDB

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        db_path = raw_dir / "convert.db"
        db = ConvertDB(db_path)
        db.open()
        db.close()

        result = runner.invoke(cli, ["catalog", "token", str(raw_dir)])
        assert result.exit_code == 0
        assert "Token" in result.output or "token" in result.output.lower()

    def test_catalog_token_no_db(self, runner, tmp_path):
        """catalog token without convert.db shows error."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        result = runner.invoke(cli, ["catalog", "token", str(raw_dir)])
        assert result.exit_code == 0
        assert "未找到转换数据库" in result.output


# ── task command group ────────────────────────────────


class TestTaskCommand:
    """Test the task CLI commands."""

    def setup_method(self):
        """Reset TaskManager singleton before each test."""
        TaskManager._instance = None

    def test_task_help(self, runner):
        result = runner.invoke(cli, ["task", "--help"])
        assert result.exit_code == 0
        assert "任务管理" in result.output

    def test_task_list(self, runner, tmp_path):
        """task list should run without error."""
        result = runner.invoke(cli, ["task", "list", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "暂无任务记录" in result.output

    def test_task_list_with_status(self, runner, tmp_path):
        """task list --status filter."""
        result = runner.invoke(cli, ["task", "list", "--status", "failed", "--path", str(tmp_path)])
        assert result.exit_code == 0

    def test_task_show_nonexistent(self, runner, tmp_path):
        """task show with invalid ID."""
        result = runner.invoke(cli, ["task", "show", "nonexistent_id", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "不存在" in result.output

    def test_task_resume_nonexistent(self, runner, tmp_path):
        """task resume with invalid ID."""
        result = runner.invoke(cli, ["task", "resume", "nonexistent_id", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "不存在" in result.output

    def test_task_cancel_nonexistent(self, runner, tmp_path):
        """task cancel with invalid ID."""
        result = runner.invoke(cli, ["task", "cancel", "nonexistent_id", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "不存在" in result.output

    def test_task_retry_nonexistent(self, runner, tmp_path):
        """task retry with invalid ID."""
        result = runner.invoke(cli, ["task", "retry", "nonexistent_id", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "不存在" in result.output


# ── update command ────────────────────────────────────


class TestUpdateCommand:
    """Test the update CLI command."""

    def test_update_help(self, runner):
        result = runner.invoke(cli, ["update", "--help"])
        assert result.exit_code == 0
        assert "更新" in result.output or "索引" in result.output

    def test_update_no_output_dir(self, runner, tmp_path):
        """update without output dir should fail."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "nonexistent_output"

        result = runner.invoke(cli, ["update", str(source), "-o", str(output)])
        assert result.exit_code != 0
        assert "输出目录不存在" in result.output

    def test_update_no_metadata(self, runner, tmp_path):
        """update with output dir but no metadata should fail."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        result = runner.invoke(cli, ["update", str(source), "-o", str(output)])
        assert result.exit_code != 0
        assert "元数据" in result.output

    def test_update_dry_run(self, runner, tmp_path):
        """update --dry-run with metadata.json shows dry-run mode."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        (output / "metadata.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(cli, ["update", str(source), "-o", str(output), "--dry-run"])
        assert result.exit_code == 0
        assert "模拟运行" in result.output

    def test_update_strategy_full(self, runner, tmp_path):
        """update --strategy full shows rebuild message."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        (output / "metadata.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(cli, ["update", str(source), "-o", str(output), "--strategy", "full"])
        assert result.exit_code == 0
        assert "完全重建" in result.output


# ── stats command group ──────────────────────────────


class TestStatsCLI:
    """F3-6: CLI stats command tests."""

    def test_stats_group_registered(self, runner):
        """stats group should be registered in CLI."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "stats" in result.output

    def test_stats_help(self, runner):
        """stats --help should show usage info."""
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0
        assert "用量统计" in result.output or "API" in result.output

    def test_stats_summary_no_data(self, runner, tmp_path, monkeypatch):
        """stats summary with no data dir should show graceful message."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path / "nonexistent_raw"))
        result = runner.invoke(cli, ["stats", "summary"])
        assert result.exit_code == 0
        assert "未找到" in result.output

    def test_stats_summary_with_data(self, runner, tmp_path, monkeypatch):
        """stats summary with a convert.db containing token usage."""
        from src.storage.convert_db import ConvertDB

        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        source_dir = raw_root / "test-docs"
        source_dir.mkdir()
        db_path = source_dir / "convert.db"

        db = ConvertDB(db_path)
        db.open()
        db.add_token_usage_extended(
            call_type="ocr",
            model="glm-ocr",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_millicents=75,
        )
        db.add_token_usage_extended(
            call_type="llm_chat",
            model="zai/glm-4",
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            cost_millicents=15000,
        )
        db.close()

        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "summary"])
        assert result.exit_code == 0
        # Should show some stats output
        assert "100" in result.output or "OCR" in result.output or "用量" in result.output

    def test_stats_daily_default_days(self, runner):
        """stats daily --help should show default 30 days."""
        result = runner.invoke(cli, ["stats", "daily", "--help"])
        assert result.exit_code == 0
        assert "30" in result.output

    def test_stats_daily_with_data(self, runner, tmp_path, monkeypatch):
        """stats daily with a convert.db should show daily trend."""
        from src.storage.convert_db import ConvertDB

        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        source_dir = raw_root / "test-docs"
        source_dir.mkdir()
        db_path = source_dir / "convert.db"

        db = ConvertDB(db_path)
        db.open()
        db.add_token_usage_extended(
            call_type="ocr",
            model="glm-ocr",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_millicents=75,
        )
        db.close()

        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "daily"])
        assert result.exit_code == 0

    def test_stats_models_help(self, runner):
        """stats models --help should show help text."""
        result = runner.invoke(cli, ["stats", "models", "--help"])
        assert result.exit_code == 0
        assert "模型" in result.output

    def test_stats_models_with_data(self, runner, tmp_path, monkeypatch):
        """stats models with data should show model breakdown."""
        from src.storage.convert_db import ConvertDB

        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        source_dir = raw_root / "test-docs"
        source_dir.mkdir()
        db_path = source_dir / "convert.db"

        db = ConvertDB(db_path)
        db.open()
        db.add_token_usage_extended(
            call_type="ocr",
            model="glm-ocr",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_millicents=75,
        )
        db.close()

        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "models"])
        assert result.exit_code == 0

    def test_stats_export_json(self, runner, tmp_path, monkeypatch):
        """stats export with JSON format should produce valid JSON."""
        from src.storage.convert_db import ConvertDB

        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        source_dir = raw_root / "test-docs"
        source_dir.mkdir()
        db_path = source_dir / "convert.db"

        db = ConvertDB(db_path)
        db.open()
        db.add_token_usage_extended(
            call_type="ocr",
            model="glm-ocr",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            cost_millicents=350,
        )
        db.close()

        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "export", "-f", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "summary" in data
        assert data["summary"]["ocr"]["calls"] == 1
        assert data["summary"]["ocr"]["input"] == 500

    def test_stats_export_csv(self, runner, tmp_path, monkeypatch):
        """stats export with CSV format should produce CSV output."""
        from src.storage.convert_db import ConvertDB

        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        source_dir = raw_root / "test-docs"
        source_dir.mkdir()
        db_path = source_dir / "convert.db"

        db = ConvertDB(db_path)
        db.open()
        db.add_token_usage_extended(
            call_type="llm_chat",
            model="zai/glm-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_millicents=7500,
        )
        db.close()

        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "export", "-f", "csv"])
        assert result.exit_code == 0
        assert "section,key,calls" in result.output
        assert "summary,llm_chat,1" in result.output

    def test_stats_export_markdown(self, runner, tmp_path, monkeypatch):
        """stats export with markdown format should produce markdown."""
        from src.storage.convert_db import ConvertDB

        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        source_dir = raw_root / "test-docs"
        source_dir.mkdir()
        db_path = source_dir / "convert.db"

        db = ConvertDB(db_path)
        db.open()
        db.add_token_usage_extended(
            call_type="rerank",
            model="rerank",
            input_tokens=50,
            total_tokens=50,
            cost_millicents=0,
        )
        db.close()

        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "export", "-f", "markdown"])
        assert result.exit_code == 0
        assert "# API 用量统计报告" in result.output
        assert "rerank" in result.output

    def test_stats_export_to_file(self, runner, tmp_path, monkeypatch):
        """stats export -o should write to file."""
        from src.storage.convert_db import ConvertDB

        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        source_dir = raw_root / "test-docs"
        source_dir.mkdir()
        db_path = source_dir / "convert.db"

        db = ConvertDB(db_path)
        db.open()
        db.add_token_usage_extended(
            call_type="ocr",
            model="glm-ocr",
            input_tokens=100,
            total_tokens=100,
            cost_millicents=50,
        )
        db.close()

        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        out_file = tmp_path / "report.json"
        result = runner.invoke(cli, ["stats", "export", "-f", "json", "-o", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "summary" in data

    def test_stats_subcommands_registered(self, runner):
        """All 4 stats subcommands should be listed in help."""
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0
        assert "summary" in result.output
        assert "daily" in result.output
        assert "models" in result.output
        assert "export" in result.output


# ── query --search-mode / --export ─────────────────────


class TestQuerySearchModes:
    """Test the query CLI --search-mode (grep/hybrid/tag) and --export options."""

    def test_query_grep_mode(self, runner, tmp_path):
        """Grep search finds content in .md files without needing an index."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc.md").write_text("# Test\n\n颈椎病治疗方案。\n", encoding="utf-8")
        result = runner.invoke(cli, ["query", "颈椎病", "-i", str(raw), "--search-mode", "grep"])
        assert result.exit_code == 0

    def test_query_grep_no_match(self, runner, tmp_path):
        """Grep mode with a query that doesn't match shows no results."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc.md").write_text("# Test\n\n颈椎病治疗方案。\n", encoding="utf-8")
        result = runner.invoke(
            cli, ["query", "完全不存在的术语XYZ", "-i", str(raw), "--search-mode", "grep"]
        )
        assert result.exit_code == 0
        assert "未找到" in result.output

    def test_query_grep_multiple_files(self, runner, tmp_path):
        """Grep search finds matches across multiple .md files."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "doc1.md").write_text("# Doc1\n\n报销流程说明。\n", encoding="utf-8")
        (raw / "doc2.md").write_text("# Doc2\n\n差旅报销标准。\n", encoding="utf-8")
        result = runner.invoke(cli, ["query", "报销", "-i", str(raw), "--search-mode", "grep"])
        assert result.exit_code == 0
        assert "doc1" in result.output or "doc2" in result.output

    @patch("src.cli.create_searcher")
    def test_query_export_json(self, mock_create, runner, temp_index_dir, tmp_path):
        """--export to a .json file writes a JSON file with search results."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "test",
            results=[
                _make_preview(
                    doc_id="d1", title="Doc1", score=5.0, snippet="content", source_path=Path("doc1.md")
                )
            ],
        )
        mock_create.return_value = mock_searcher

        out = tmp_path / "results.json"
        result = runner.invoke(
            cli, ["query", "test", "-i", str(temp_index_dir / "index"), "--export", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "results" in data

    @patch("src.cli.create_searcher")
    def test_query_export_csv(self, mock_create, runner, temp_index_dir, tmp_path):
        """--export to a .csv file writes a CSV file with search results."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "test",
            results=[
                _make_preview(
                    doc_id="d1", title="Doc1", score=5.0, snippet="content", source_path=Path("doc1.md")
                )
            ],
        )
        mock_create.return_value = mock_searcher

        out = tmp_path / "results.csv"
        result = runner.invoke(
            cli, ["query", "test", "-i", str(temp_index_dir / "index"), "--export", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8-sig")
        assert "," in content

    @patch("src.cli.create_searcher")
    def test_query_export_markdown(self, mock_create, runner, temp_index_dir, tmp_path):
        """--export to a .md file writes a Markdown file with search results."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "test",
            results=[
                _make_preview(
                    doc_id="d1", title="Doc1", score=5.0, snippet="content", source_path=Path("doc1.md")
                )
            ],
        )
        mock_create.return_value = mock_searcher

        out = tmp_path / "results.md"
        result = runner.invoke(
            cli, ["query", "test", "-i", str(temp_index_dir / "index"), "--export", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "搜索结果" in content

    @patch("src.cli.create_searcher")
    def test_query_export_to_file(self, mock_create, runner, temp_index_dir, tmp_path):
        """Export output file is created at the specified path, including nested dirs."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "export-test",
            results=[
                _make_preview(
                    doc_id="d1",
                    title="ExportDoc",
                    score=3.0,
                    snippet="snippet text",
                    source_path=Path("doc.md"),
                )
            ],
        )
        mock_create.return_value = mock_searcher

        out = tmp_path / "output" / "exported.json"
        result = runner.invoke(
            cli,
            ["query", "export-test", "-i", str(temp_index_dir / "index"), "--export", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        assert out.stat().st_size > 0

    @patch("src.cli.create_searcher")
    def test_query_export_short_alias(self, mock_create, runner, temp_index_dir, tmp_path):
        """The -e short alias works identically to --export."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results(
            "alias-test",
            results=[
                _make_preview(
                    doc_id="d1", title="AliasDoc", score=2.0, snippet="data", source_path=Path("a.md")
                )
            ],
        )
        mock_create.return_value = mock_searcher

        out = tmp_path / "alias_out.json"
        result = runner.invoke(
            cli, ["query", "alias-test", "-i", str(temp_index_dir / "index"), "-e", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()

    @patch("src.cli.create_searcher")
    def test_query_export_no_results(self, mock_create, runner, temp_index_dir, tmp_path):
        """When search returns no results, export file is not created."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = _make_paginated_results("nomatch")
        mock_create.return_value = mock_searcher

        out = tmp_path / "empty.json"
        result = runner.invoke(
            cli, ["query", "nomatch", "-i", str(temp_index_dir / "index"), "--export", str(out)]
        )
        assert result.exit_code == 0
        assert not out.exists()

    def test_query_search_mode_help(self, runner):
        """--search-mode option should be visible in query help."""
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "--search-mode" in result.output


# ── stats diagnostics 子命令 ────────────────────────────


def _make_diagnostics_db(tmp_path):
    """Create a real convert.db with diagnostic data for stats tests."""
    from src.storage.convert_db import ConvertDB

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    source_dir = raw_root / "test-docs"
    source_dir.mkdir()
    db_path = source_dir / "convert.db"

    db = ConvertDB(db_path)
    db.open()
    diag_id = db.add_query_diagnostic(
        session_id="test-session-1",
        query_hash="testhash001",
        query_preview="test query for diagnostics",
        complexity="complex",
        total_ms=50000,
        success=1,
        llm_call_count=5,
        tool_call_count=10,
        tool_cache_hits=3,
        step_timings=json.dumps({"tool_loop": 30000.0, "expand": 1000.0}),
        model="glm-4",
        source_dir=str(source_dir),
    )
    db.add_llm_call_log(
        diagnostic_id=diag_id,
        call_type="tool_loop",
        call_sequence=1,
        latency_ms=2000,
        input_tokens=100,
        output_tokens=50,
    )
    db.close()
    return raw_root


class TestStatsDiagnostics:
    """P1: CLI stats diagnostics subcommands."""

    def test_diagnostics_no_data(self, runner, tmp_path, monkeypatch):
        """diagnostics with no convert.db exits cleanly."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path / "nonexistent"))
        result = runner.invoke(cli, ["stats", "diagnostics"])
        assert result.exit_code == 0
        assert "未找到" in result.output

    def test_diagnostics_with_data(self, runner, tmp_path, monkeypatch):
        """diagnostics with data shows summary stats."""
        raw_root = _make_diagnostics_db(tmp_path)
        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "diagnostics"])
        assert result.exit_code == 0
        assert "诊断" in result.output or "查询" in result.output

    def test_slow_queries_no_data(self, runner, tmp_path, monkeypatch):
        """slow-queries with no db exits cleanly."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path / "nonexistent"))
        result = runner.invoke(cli, ["stats", "slow-queries"])
        assert result.exit_code == 0
        assert "未找到" in result.output

    def test_slow_queries_with_data(self, runner, tmp_path, monkeypatch):
        """slow-queries with data lists queries above threshold."""
        raw_root = _make_diagnostics_db(tmp_path)
        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "slow-queries", "--threshold", "30000"])
        assert result.exit_code == 0
        assert "test query" in result.output

    def test_step_breakdown_no_data(self, runner, tmp_path, monkeypatch):
        """step-breakdown with no db exits cleanly."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path / "nonexistent"))
        result = runner.invoke(cli, ["stats", "step-breakdown"])
        assert result.exit_code == 0
        assert "未找到" in result.output

    def test_step_breakdown_with_data(self, runner, tmp_path, monkeypatch):
        """step-breakdown with data shows step timings."""
        raw_root = _make_diagnostics_db(tmp_path)
        monkeypatch.setenv("RAW_ROOT", str(raw_root))
        result = runner.invoke(cli, ["stats", "step-breakdown"])
        assert result.exit_code == 0
        assert "分步" in result.output or "tool_loop" in result.output

    def test_llm_calls_no_data(self, runner, tmp_path, monkeypatch):
        """llm-calls with no db exits cleanly."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path / "nonexistent"))
        result = runner.invoke(cli, ["stats", "llm-calls"])
        assert result.exit_code == 0
        assert "未找到" in result.output


# ── stats budget 子命令 ─────────────────────────────


class TestStatsBudgetCLI:
    """P1: CLI stats budget subcommands."""

    def test_budget_list_no_data(self, runner, tmp_path, monkeypatch):
        """budget list with no db exits cleanly."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path / "nonexistent"))
        result = runner.invoke(cli, ["stats", "budget", "list"])
        assert result.exit_code == 0

    def test_budget_set(self, runner, tmp_path, monkeypatch):
        """budget set creates a budget in convert.db."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path))
        result = runner.invoke(cli, [
            "stats", "budget", "set",
            "--name", "test-budget",
            "--limit", "10000",
            "--period", "monthly",
        ])
        assert result.exit_code == 0
        assert "预算已设置" in result.output or "test-budget" in result.output

    def test_budget_check(self, runner, tmp_path, monkeypatch):
        """budget check after setting a budget shows status."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path))
        runner.invoke(cli, [
            "stats", "budget", "set",
            "--name", "check-budget",
            "--limit", "5000",
        ])
        result = runner.invoke(cli, ["stats", "budget", "check"])
        assert result.exit_code == 0

    def test_budget_remove(self, runner, tmp_path, monkeypatch):
        """budget remove deletes a budget by ID."""
        monkeypatch.setenv("RAW_ROOT", str(tmp_path))
        runner.invoke(cli, [
            "stats", "budget", "set",
            "--name", "remove-me",
            "--limit", "1000",
        ])
        result = runner.invoke(cli, ["stats", "budget", "remove", "1"])
        assert result.exit_code == 0
        assert "删除" in result.output or "未找到" in result.output


# ── catalog backfill-headings ────────────────────────────


class TestCatalogBackfill:
    """P2: catalog backfill-headings command."""

    def test_backfill_headings_dry_run(self, runner, tmp_path):
        """backfill-headings --dry-run shows what would change without writing."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.md").write_text(
            "# 标题一\n\n内容\n\n## 子标题\n\n更多内容\n", encoding="utf-8"
        )
        (raw_dir / "doc.md.json").write_text(
            json.dumps({"source": "doc.md"}, ensure_ascii=False), encoding="utf-8"
        )

        result = runner.invoke(cli, ["catalog", "backfill-headings", str(raw_dir), "--dry-run"])
        assert result.exit_code == 0
        assert "doc.md" in result.output
        assert "dry-run" in result.output.lower()
        # .md.json should NOT have headings after dry-run
        data = json.loads((raw_dir / "doc.md.json").read_text(encoding="utf-8"))
        assert "headings" not in data

    def test_backfill_headings_apply(self, runner, tmp_path):
        """backfill-headings writes headings to .md.json."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.md").write_text(
            "# 标题一\n\n内容\n\n## 子标题\n\n更多内容\n", encoding="utf-8"
        )
        (raw_dir / "doc.md.json").write_text(
            json.dumps({"source": "doc.md"}, ensure_ascii=False), encoding="utf-8"
        )

        result = runner.invoke(cli, ["catalog", "backfill-headings", str(raw_dir)])
        assert result.exit_code == 0
        assert "已更新" in result.output or "回填" in result.output
        data = json.loads((raw_dir / "doc.md.json").read_text(encoding="utf-8"))
        assert "headings" in data
        assert len(data["headings"]) >= 2

    def test_backfill_headings_already_has(self, runner, tmp_path):
        """backfill-headings skips files that already have headings."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.md").write_text("# 标题\n\n内容\n", encoding="utf-8")
        (raw_dir / "doc.md.json").write_text(
            json.dumps(
                {"source": "doc.md", "headings": [{"level": 1, "text": "标题", "line": 1}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["catalog", "backfill-headings", str(raw_dir)])
        assert result.exit_code == 0
        assert "跳过" in result.output or "已有" in result.output


# ── catalog inject-frontmatter ────────────────────────────


class TestCatalogInjectFrontmatter:
    """P2: catalog inject-frontmatter command."""

    def test_inject_frontmatter_dry_run(self, runner, tmp_path):
        """inject-frontmatter --dry-run shows preview without writing."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.md").write_text("# 标题\n\n内容\n", encoding="utf-8")
        (raw_dir / "doc.md.json").write_text(
            json.dumps(
                {"title": "doc", "doc_type": "document", "tags": [], "source": "doc.md"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["catalog", "inject-frontmatter", str(raw_dir), "--dry-run"])
        assert result.exit_code == 0
        assert "doc.md" in result.output
        assert "dry-run" in result.output.lower()
        # .md should NOT have frontmatter after dry-run
        content = (raw_dir / "doc.md").read_text(encoding="utf-8")
        assert not content.startswith("---")

    def test_inject_frontmatter_apply(self, runner, tmp_path):
        """inject-frontmatter writes YAML frontmatter to .md."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.md").write_text("# 标题\n\n内容\n", encoding="utf-8")
        (raw_dir / "doc.md.json").write_text(
            json.dumps(
                {"title": "doc", "doc_type": "document", "tags": [], "source": "doc.md"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["catalog", "inject-frontmatter", str(raw_dir)])
        assert result.exit_code == 0
        content = (raw_dir / "doc.md").read_text(encoding="utf-8")
        assert content.startswith("---")

    def test_inject_frontmatter_already_has(self, runner, tmp_path):
        """inject-frontmatter skips files that already have frontmatter."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.md").write_text(
            "---\ntitle: doc\n---\n\n# 标题\n\n内容\n", encoding="utf-8"
        )
        (raw_dir / "doc.md.json").write_text(
            json.dumps({"title": "doc"}, ensure_ascii=False), encoding="utf-8"
        )

        result = runner.invoke(cli, ["catalog", "inject-frontmatter", str(raw_dir)])
        assert result.exit_code == 0
        assert "跳过" in result.output or "已有" in result.output


# ── diff-migrate ────────────────────────────


class TestDiffMigrate:
    """P2: diff-migrate command."""

    def test_diff_migrate_basic(self, runner, tmp_path):
        """diff-migrate shows new files between two directories."""
        base = tmp_path / "base"
        base.mkdir()
        compare = tmp_path / "compare"
        compare.mkdir()
        (base / "file1.txt").write_text("content1", encoding="utf-8")
        (compare / "file1.txt").write_text("content1", encoding="utf-8")
        (compare / "file2.txt").write_text("content2", encoding="utf-8")

        result = runner.invoke(cli, ["diff-migrate", str(base), str(compare)])
        assert result.exit_code == 0
        assert "file2.txt" in result.output
        assert "新增" in result.output

    def test_diff_migrate_export_new(self, runner, tmp_path):
        """diff-migrate --export-new copies new files to output directory."""
        base = tmp_path / "base"
        base.mkdir()
        compare = tmp_path / "compare"
        compare.mkdir()
        export = tmp_path / "export"
        (base / "file1.txt").write_text("content1", encoding="utf-8")
        (compare / "file1.txt").write_text("content1", encoding="utf-8")
        (compare / "file2.txt").write_text("content2", encoding="utf-8")

        result = runner.invoke(cli, [
            "diff-migrate", str(base), str(compare), "--export-new", str(export),
        ])
        assert result.exit_code == 0
        assert (export / "file2.txt").exists()
        assert (export / "file2.txt").read_text(encoding="utf-8") == "content2"

    def test_diff_migrate_identical_dirs(self, runner, tmp_path):
        """diff-migrate with identical dirs reports zero additions."""
        base = tmp_path / "base"
        base.mkdir()
        compare = tmp_path / "compare"
        compare.mkdir()
        (base / "file1.txt").write_text("same content", encoding="utf-8")
        (compare / "file1.txt").write_text("same content", encoding="utf-8")

        result = runner.invoke(cli, ["diff-migrate", str(base), str(compare)])
        assert result.exit_code == 0
        assert "新增: 0" in result.output

    def test_diff_migrate_report_json(self, runner, tmp_path):
        """diff-migrate -o exports JSON diff report."""
        base = tmp_path / "base"
        base.mkdir()
        compare = tmp_path / "compare"
        compare.mkdir()
        (base / "file1.txt").write_text("c1", encoding="utf-8")
        (compare / "file1.txt").write_text("c1", encoding="utf-8")
        (compare / "file2.txt").write_text("c2", encoding="utf-8")
        report = tmp_path / "report.json"

        result = runner.invoke(cli, ["diff-migrate", str(base), str(compare), "-o", str(report)])
        assert result.exit_code == 0
        assert report.exists()
        data = json.loads(report.read_text(encoding="utf-8"))
        assert len(data) >= 2

    def test_diff_migrate_all_files(self, runner, tmp_path):
        """diff-migrate --all-files compares all file types including unsupported."""
        base = tmp_path / "base"
        base.mkdir()
        compare = tmp_path / "compare"
        compare.mkdir()
        (base / "data.xyz").write_text("c1", encoding="utf-8")
        (compare / "data.xyz").write_text("c1", encoding="utf-8")
        (compare / "new.xyz").write_text("c2", encoding="utf-8")

        result = runner.invoke(cli, [
            "diff-migrate", str(base), str(compare), "--all-files",
        ])
        assert result.exit_code == 0
        assert "new.xyz" in result.output


# ── analyze ────────────────────────────


class TestAnalyzeCommand:
    """P2: analyze command."""

    def test_analyze_help(self, runner):
        """analyze --help exits 0 and shows usage."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "分析" in result.output

    def test_analyze_modes_in_help(self, runner):
        """analyze --help lists all analysis modes."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.output
        assert "extract" in result.output
        assert "summarize" in result.output
        assert "table" in result.output

    def test_analyze_missing_required_index(self, runner):
        """analyze without -i triggers Click missing-required-option error."""
        result = runner.invoke(cli, ["analyze", "test"])
        assert result.exit_code != 0

    def test_analyze_mock(self, runner, tmp_path):
        """analyze with mocked analysis agent returns answer."""
        with patch("src.utils.config.Config.from_env") as mock_config, \
             patch("src.agent.analysis_agent.search_and_analyze") as mock_search:
            mock_config.return_value = MagicMock(
                glm_api_key="test-key",
                glm_base_url="http://test",
                llm_model="glm-4",
            )
            mock_search.return_value = MagicMock(
                success=True,
                answer="分析结果文本",
                sources=["doc1.md"],
                tokens_used=10,
                processing_time=1.0,
            )

            result = runner.invoke(cli, ["analyze", "测试", "-i", str(tmp_path / "idx")])

        assert result.exit_code == 0
        assert "分析结果文本" in result.output
