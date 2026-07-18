"""Tests for document structure awareness (headings extraction + ReadTool TOC injection)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.tools.read import ReadTool
from src.converter.headings import extract_headings

# ── extract_headings tests ──────────────────────────────────────────


class TestExtractHeadings:
    """Tests for the extract_headings utility function."""

    def test_empty_input(self):
        assert extract_headings("") == []
        assert extract_headings(None) == []

    def test_no_headings(self):
        md = "This is plain text.\nNo headings here.\nJust paragraphs."
        assert extract_headings(md) == []

    def test_single_heading(self):
        md = "# Title"
        result = extract_headings(md)
        assert len(result) == 1
        assert result[0]["level"] == 1
        assert result[0]["text"] == "Title"
        assert result[0]["line"] == 1

    def test_multi_level_headings(self):
        md = """# Main Title

Some intro text.

## Section A

Content A.

### Subsection A.1

Detail A.1.

## Section B

Content B.
"""
        result = extract_headings(md)
        assert len(result) == 4
        assert result[0] == {"level": 1, "text": "Main Title", "line": 1}
        assert result[1] == {"level": 2, "text": "Section A", "line": 5}
        assert result[2] == {"level": 3, "text": "Subsection A.1", "line": 9}
        assert result[3] == {"level": 2, "text": "Section B", "line": 13}

    def test_all_heading_levels(self):
        md = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        result = extract_headings(md)
        assert len(result) == 6
        for i, h in enumerate(result):
            assert h["level"] == i + 1

    def test_heading_with_inline_formatting(self):
        md = "# **Bold Title** and `code`"
        result = extract_headings(md)
        assert len(result) == 1
        assert result[0]["text"] == "**Bold Title** and `code`"

    def test_chinese_headings(self):
        md = "# 年假管理制度\n## 一、年假天数标准\n## 二、申请流程\n### 2.1 线上申请"
        result = extract_headings(md)
        assert len(result) == 4
        assert result[0]["text"] == "年假管理制度"
        assert result[1]["text"] == "一、年假天数标准"
        assert result[2]["text"] == "二、申请流程"
        assert result[3]["text"] == "2.1 线上申请"

    def test_skip_code_block_hashes(self):
        md = """# Real Title

```python
# This is a comment, not a heading
x = 1
```

## Another Section
"""
        result = extract_headings(md)
        assert len(result) == 2
        assert result[0]["text"] == "Real Title"
        assert result[1]["text"] == "Another Section"

    def test_max_headings_limit(self):
        md = "\n".join(f"# Heading {i}" for i in range(50))
        result = extract_headings(md)
        assert len(result) == 30  # MAX_HEADINGS = 30

    def test_no_hash_no_heading(self):
        md = "Plain text\nNo heading markers\nJust words"
        assert extract_headings(md) == []

    def test_hash_without_space_not_heading(self):
        """# followed by non-space is NOT a heading in standard Markdown."""
        md = "#NotAHeading\n# IsAHeading"
        result = extract_headings(md)
        assert len(result) == 1
        assert result[0]["text"] == "IsAHeading"

    def test_line_numbers_correct(self):
        md = "Line 1\nLine 2\n# Heading at line 3\nLine 4\n## Heading at line 5"
        result = extract_headings(md)
        assert result[0]["line"] == 3
        assert result[1]["line"] == 5

    def test_trailing_whitespace_stripped(self):
        md = "# Title   \n## Section  "
        result = extract_headings(md)
        assert result[0]["text"] == "Title"
        assert result[1]["text"] == "Section"


# ── ReadTool TOC injection tests ────────────────────────────────────


class TestReadToolTocInjection:
    """Tests for ReadTool TOC formatting and injection."""

    def _make_read_tool(self, store=None, raw_dirs=None):
        store = store or MagicMock()
        return ReadTool(markdown_store=store, raw_dirs=raw_dirs or [])

    def test_format_toc_empty(self):
        tool = self._make_read_tool()
        assert tool._format_toc([]) == ""
        assert tool._format_toc(None) == ""

    def test_format_toc_single_heading(self):
        tool = self._make_read_tool()
        headings = [{"level": 1, "text": "Title", "line": 1}]
        result = tool._format_toc(headings)
        assert "## 文档目录" in result
        assert "- Title (行 1)" in result
        assert "---" in result

    def test_format_toc_multi_level(self):
        tool = self._make_read_tool()
        headings = [
            {"level": 1, "text": "Main", "line": 1},
            {"level": 2, "text": "Sub A", "line": 10},
            {"level": 3, "text": "Sub A.1", "line": 15},
            {"level": 2, "text": "Sub B", "line": 30},
        ]
        result = tool._format_toc(headings)
        assert "- Main (行 1)" in result
        assert "  - Sub A (行 10)" in result
        assert "    - Sub A.1 (行 15)" in result
        assert "  - Sub B (行 30)" in result

    def test_format_toc_prepended_on_first_read(self):
        """Verify TOC is prepended when start_line == 0."""
        store = MagicMock()
        record = MagicMock()
        record.file_size = 100
        record.id = "test123"
        record.source_path = Path("test.md")
        record.title = "Test"
        record.metadata = {
            "headings": [
                {"level": 1, "text": "Title", "line": 1},
                {"level": 2, "text": "Section", "line": 5},
            ]
        }
        content = "# Title\n\nSome content\n\n## Section\n\nMore content\n"
        store.load.return_value = (record, content)

        tool = self._make_read_tool(store=store)
        result = tool.execute(doc_id="test123", start_line=0)

        assert result.success
        assert "## 文档目录" in result.data
        assert "- Title (行 1)" in result.data
        assert "# Title" in result.data  # Original content still present

    def test_no_toc_on_subsequent_read(self):
        """Verify TOC is NOT prepended when start_line > 0."""
        store = MagicMock()
        record = MagicMock()
        record.file_size = 100
        record.id = "test123"
        record.source_path = Path("test.md")
        record.title = "Test"
        record.metadata = {
            "headings": [{"level": 1, "text": "Title", "line": 1}]
        }
        content_lines = [f"Line {i}" for i in range(100)]
        content = "\n".join(content_lines)
        store.load.return_value = (record, content)

        tool = self._make_read_tool(store=store)
        result = tool.execute(doc_id="test123", start_line=10, max_lines=20)

        assert result.success
        assert "## 文档目录" not in result.data
        assert "Line 10" in result.data

    def test_no_headings_graceful_fallback(self):
        """Verify no TOC when .md.json has no headings."""
        store = MagicMock()
        record = MagicMock()
        record.file_size = 50
        record.id = "test123"
        record.source_path = Path("test.md")
        record.title = "Test"
        record.metadata = {}  # No headings
        content = "Plain text without headings."
        store.load.return_value = (record, content)

        tool = self._make_read_tool(store=store)
        result = tool.execute(doc_id="test123", start_line=0)

        assert result.success
        assert "## 文档目录" not in result.data
        assert "Plain text without headings." in result.data

    def test_load_headings_from_raw_dirs(self):
        """Verify _load_headings can read from raw_dirs as fallback."""
        import tempfile
        from datetime import datetime

        from src.storage.base import DocumentRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .md file with a realistic name
            md_file = Path(tmpdir) / "test.docx.md"
            md_content = "# From Raw Dir\n\nContent here\n"
            md_file.write_text(md_content, encoding="utf-8")

            # Create .md.json with headings
            md_json = Path(tmpdir) / "test.docx.md.json"
            headings_data = [
                {"level": 1, "text": "From Raw Dir", "line": 1}
            ]
            md_json.write_text(
                json.dumps({"headings": headings_data}, ensure_ascii=False),
                encoding="utf-8",
            )

            # Mock store: load_by_source returns the record + content for our file
            record = DocumentRecord(
                id="raw_read",
                title="test.docx",
                source_path=md_file,
                output_path=md_file,
                content_hash="",
                file_size=len(md_content),
                file_mtime=datetime.now(),
                metadata={"headings": headings_data},
            )
            store = MagicMock()
            store.load.return_value = None
            store.load_by_source.return_value = (record, md_content)

            tool = self._make_read_tool(store=store, raw_dirs=[tmpdir])
            result = tool.execute(source_path="test.docx.md", start_line=0)

            assert result.success
            assert "## 文档目录" in result.data
            assert "From Raw Dir" in result.data


# ── Coordinator integration test ────────────────────────────────────


class TestCoordinatorHeadingsIntegration:
    """Test that coordinator includes headings in metadata."""

    def test_headings_in_metadata(self):
        """Verify extract_headings is called and result stored in metadata."""
        from src.converter.headings import extract_headings

        md = "# Title\n## Section\nContent"
        headings = extract_headings(md)

        # Simulate what coordinator does
        metadata = {}
        metadata["headings"] = headings

        assert "headings" in metadata
        assert len(metadata["headings"]) == 2
        assert metadata["headings"][0]["text"] == "Title"

    def test_headings_empty_for_plain_text(self):
        from src.converter.headings import extract_headings

        md = "Just plain text.\nNo headings at all."
        headings = extract_headings(md)
        assert headings == []

    def test_headings_survives_json_roundtrip(self):
        """Verify headings data survives .md.json write/read cycle."""
        from src.converter.headings import extract_headings

        md = "# 主标题\n## 第一节\n### 1.1 子节\n## 第二节"
        headings = extract_headings(md)

        metadata = {
            "tags": ["test"],
            "doc_type": "test",
            "headings": headings,
        }

        # Write to JSON string and back
        json_str = json.dumps(metadata, ensure_ascii=False)
        loaded = json.loads(json_str)

        assert loaded["headings"] == headings
        assert len(loaded["headings"]) == 4
        assert loaded["headings"][2]["text"] == "1.1 子节"
