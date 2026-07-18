"""Tests for SummarizeTool — document summarization via LLM."""

from unittest.mock import MagicMock

from src.agent.tools.summarize import SummarizeTool


class TestSummarizeToolBasics:
    """Basic SummarizeTool tests."""

    def test_name(self):
        tool = SummarizeTool(markdown_store=MagicMock())
        assert tool.name == "summarize"

    def test_description(self):
        tool = SummarizeTool(markdown_store=MagicMock())
        assert "总结" in tool.description or "要点" in tool.description

    def test_to_openai_tool(self):
        tool = SummarizeTool(markdown_store=MagicMock())
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "summarize"
        props = schema["function"]["parameters"]["properties"]
        assert "doc_id" in props
        assert "source_path" in props
        assert "focus" in props
        assert "max_lines" in props


class TestSummarizeToolExecute:
    """SummarizeTool.execute() tests."""

    def test_missing_both_doc_id_and_source_path(self):
        """Should fail when neither doc_id nor source_path provided."""
        tool = SummarizeTool(markdown_store=MagicMock(), llm_client=MagicMock())
        result = tool.execute()
        assert not result.success
        assert "doc_id" in result.error or "source_path" in result.error

    def test_missing_llm_client(self):
        """Should fail when LLM client is None."""
        mock_store = MagicMock()
        mock_store.load.return_value = (MagicMock(), "文档内容")
        tool = SummarizeTool(markdown_store=mock_store, llm_client=None)
        result = tool.execute(doc_id="abc123")
        assert not result.success
        assert "LLM" in result.error

    def test_doc_not_found(self):
        """Should fail when document not found."""
        mock_store = MagicMock()
        mock_store.load.return_value = None
        mock_searcher = MagicMock()
        mock_searcher.get_full_content.return_value = None
        tool = SummarizeTool(
            markdown_store=mock_store,
            searcher=mock_searcher,
            llm_client=MagicMock(),
        )
        result = tool.execute(doc_id="nonexistent")
        assert not result.success
        assert "not found" in result.error

    def test_successful_summarize_by_doc_id(self):
        """Should return LLM summary when document found."""
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.file_size = 100
        mock_store.load.return_value = (mock_record, "这是一份年假制度文档。")
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "## 文档主旨\n年假管理制度"
        mock_response.usage = {"total_tokens": 50}
        mock_llm.chat.return_value = mock_response

        tool = SummarizeTool(markdown_store=mock_store, llm_client=mock_llm)
        result = tool.execute(doc_id="abc123")

        assert result.success
        assert "年假" in result.data
        assert result.metadata["tokens_used"] == 50
        assert result.metadata["doc_id"] == "abc123"
        assert "execution_time" in result.metadata

    def test_successful_summarize_with_focus(self):
        """Should include focus area in the prompt."""
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.file_size = 100
        mock_store.load.return_value = (mock_record, "文档内容")
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "总结结果"
        mock_response.usage = {"total_tokens": 30}
        mock_llm.chat.return_value = mock_response

        tool = SummarizeTool(markdown_store=mock_store, llm_client=mock_llm)
        result = tool.execute(doc_id="abc123", focus="申请流程")

        assert result.success
        assert result.metadata["focus"] == "申请流程"
        # Verify focus was included in the prompt
        call_args = mock_llm.chat.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "申请流程" in prompt_content

    def test_truncation_notice_for_long_docs(self):
        """Should append truncation notice for long documents."""
        long_content = "\n".join([f"第{i}行" for i in range(600)])
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.file_size = 10000
        mock_store.load.return_value = (mock_record, long_content)
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "总结"
        mock_response.usage = {"total_tokens": 30}
        mock_llm.chat.return_value = mock_response

        tool = SummarizeTool(markdown_store=mock_store, llm_client=mock_llm)
        result = tool.execute(doc_id="abc123", max_lines=500)

        assert result.success
        assert result.metadata["truncated"] is True
        assert result.metadata["total_lines"] == 600
        assert "⚠️" in result.data or "仅总结" in result.data

    def test_llm_failure_returns_error(self):
        """Should return error when LLM raises exception."""
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.file_size = 100
        mock_store.load.return_value = (mock_record, "文档内容")
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API timeout")

        tool = SummarizeTool(markdown_store=mock_store, llm_client=mock_llm)
        result = tool.execute(doc_id="abc123")

        assert not result.success
        assert "Summarization failed" in result.error

    def test_empty_llm_response_returns_error(self):
        """Should return error when LLM returns empty content."""
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.file_size = 100
        mock_store.load.return_value = (mock_record, "文档内容")
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.usage = {"total_tokens": 5}
        mock_llm.chat.return_value = mock_response

        tool = SummarizeTool(markdown_store=mock_store, llm_client=mock_llm)
        result = tool.execute(doc_id="abc123")

        assert not result.success
        assert "empty" in result.error.lower()

    def test_source_path_loading(self):
        """Should load document by source_path."""
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.file_size = 100
        mock_store.load_by_source.return_value = (mock_record, "文档内容")
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "总结"
        mock_response.usage = {"total_tokens": 30}
        mock_llm.chat.return_value = mock_response

        tool = SummarizeTool(markdown_store=mock_store, llm_client=mock_llm)
        result = tool.execute(source_path="docs/policy.md")

        assert result.success
        assert result.metadata["source_path"] == "docs/policy.md"
