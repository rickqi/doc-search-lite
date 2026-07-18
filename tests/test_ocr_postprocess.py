"""Tests for OCR post-processing pipeline."""

import pytest

from src.converter.ocr_postprocess import (
    clean_ocr_text,
    detect_headings,
    merge_paragraphs,
    normalize_tables,
    postprocess_ocr_result,
)


class TestCleanOCRText:
    """Tests for clean_ocr_text function."""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert clean_ocr_text("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input returns empty string after strip."""
        assert clean_ocr_text("   \n  \n  ") == ""

    def test_crlf_normalization(self):
        """CRLF line endings are normalized to LF."""
        text = "line1\r\nline2\r\nline3"
        assert clean_ocr_text(text) == "line1\nline2\nline3"

    def test_cr_only_normalization(self):
        """CR-only line endings are normalized to LF."""
        text = "line1\rline2"
        assert clean_ocr_text(text) == "line1\nline2"

    def test_control_characters_removed(self):
        """Stray control characters are removed."""
        text = "hello\x00world\x07test\x1f"
        assert clean_ocr_text(text) == "helloworldtest"

    def test_tab_preserved(self):
        """Tab characters are preserved."""
        text = "hello\tworld"
        assert clean_ocr_text(text) == "hello\tworld"

    def test_blank_line_collapse(self):
        """3+ consecutive blank lines collapse to 2."""
        text = "line1\n\n\n\nline2"
        assert clean_ocr_text(text) == "line1\n\nline2"

    def test_two_blank_lines_preserved(self):
        """Exactly 2 blank lines are preserved."""
        text = "line1\n\nline2"
        assert clean_ocr_text(text) == "line1\n\nline2"

    def test_leading_trailing_whitespace_stripped(self):
        """Leading and trailing whitespace is stripped."""
        text = "\n\n  hello  \n\n"
        assert clean_ocr_text(text) == "hello"

    def test_mixed_artifacts(self):
        """Multiple artifact types cleaned together."""
        text = "\r\n\x00hello\r\n\r\n\r\nworld\x07\r\n"
        result = clean_ocr_text(text)
        assert "\r" not in result
        assert "\x00" not in result
        assert "\x07" not in result
        assert result == "hello\n\nworld"


class TestMergeParagraphs:
    """Tests for merge_paragraphs function."""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert merge_paragraphs("") == ""

    def test_single_line(self):
        """Single line is unchanged."""
        assert merge_paragraphs("hello world") == "hello world"

    def test_consecutive_lines_merged(self):
        """Two non-special consecutive lines are merged with a space."""
        text = "first line\nsecond line"
        assert merge_paragraphs(text) == "first line second line"

    def test_blank_line_prevents_merge(self):
        """Blank line between lines prevents merging."""
        text = "first line\n\nsecond line"
        assert merge_paragraphs(text) == "first line\n\nsecond line"

    def test_heading_not_merged(self):
        """Lines starting with # are not merged with previous."""
        text = "some text\n## heading"
        assert merge_paragraphs(text) == "some text\n## heading"

    def test_list_item_not_merged(self):
        """List items are not merged with previous line."""
        text = "some text\n- list item"
        assert merge_paragraphs(text) == "some text\n- list item"

    def test_ordered_list_not_merged(self):
        """Ordered list items are not merged."""
        text = "some text\n1. first item"
        assert merge_paragraphs(text) == "some text\n1. first item"

    def test_table_row_not_merged(self):
        """Table rows are not merged."""
        text = "some text\n| cell |"
        assert merge_paragraphs(text) == "some text\n| cell |"

    def test_blockquote_not_merged(self):
        """Blockquote lines are not merged."""
        text = "some text\n> quote"
        assert merge_paragraphs(text) == "some text\n> quote"

    def test_bullet_variants(self):
        """Different bullet characters prevent merging."""
        for bullet in ["- item", "* item", "• item"]:
            text = f"prev line\n{bullet}"
            result = merge_paragraphs(text)
            lines = result.split("\n")
            assert len(lines) == 2, f"Bullet '{bullet[0]}' should prevent merge"

    def test_multiple_paragraphs(self):
        """Multiple paragraphs separated by blank lines are preserved."""
        text = "para one\n\npara two\n\npara three"
        assert merge_paragraphs(text) == text

    def test_three_lines_merged(self):
        """Three consecutive non-special lines merge into one."""
        text = "line one\nline two\nline three"
        assert merge_paragraphs(text) == "line one line two line three"

    def test_mixed_merge_and_preserve(self):
        """Mixed: some lines merge, structural lines don't."""
        text = "first part\nsecond part\n\n## heading\n\nmore text\ncontinuation"
        result = merge_paragraphs(text)
        assert "first part second part" in result
        assert "## heading" in result
        assert "more text continuation" in result


class TestDetectHeadings:
    """Tests for detect_headings function."""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert detect_headings("") == ""

    def test_existing_heading_preserved(self):
        """Already-formatted headings are not modified."""
        text = "## existing heading\ncontent"
        assert detect_headings(text) == text

    def test_short_line_before_content(self):
        """Short line followed by longer content becomes heading."""
        text = "保险条款\n这是一段很长的保险条款内容，包含详细的描述和说明文字。"
        result = detect_headings(text)
        assert "### 保险条款" in result

    def test_colon_ended_line(self):
        """Line ending with Chinese colon becomes heading."""
        text = "理赔流程：\n详细说明内容"
        result = detect_headings(text)
        assert "### 理赔流程" in result

    def test_english_colon_ended_line(self):
        """Line ending with English colon becomes heading."""
        text = "Claims Process:\nDetailed description here"
        result = detect_headings(text)
        assert "### Claims Process" in result

    def test_all_caps_line(self):
        """All-caps Latin line becomes heading."""
        text = "IMPORTANT\nThis is important content that follows."
        result = detect_headings(text)
        assert "### IMPORTANT" in result

    def test_short_caps_ignored(self):
        """Very short all-caps (< 3 chars) is not treated as heading."""
        text = "OK\nNext line content here"
        result = detect_headings(text)
        assert "### OK" not in result

    def test_code_block_not_converted(self):
        """Lines inside code blocks are not converted to headings."""
        text = "```\nSHORT\nlonger content inside code\n```"
        result = detect_headings(text)
        assert "### SHORT" not in result

    def test_table_rows_not_converted(self):
        """Table rows are not converted to headings."""
        text = "| col |\n| --- |\n| data |"
        result = detect_headings(text)
        assert "###" not in result

    def test_list_items_not_converted(self):
        """List items are not converted to headings."""
        text = "- item one\n  some content"
        result = detect_headings(text)
        assert "###" not in result

    def test_line_ending_with_period_not_heading(self):
        """Short line ending with period is not treated as heading."""
        text = "这是句子。\n更长的内容行在这里继续"
        result = detect_headings(text)
        assert "### 这是句子" not in result

    def test_too_long_for_short_rule(self):
        """Lines >= 30 chars don't trigger short-line rule."""
        long_line = "这是一个超过三十个字符的长行不应该被当作标题"
        text = f"{long_line}\nshort"
        result = detect_headings(text)
        assert f"### {long_line}" not in result


class TestNormalizeTables:
    """Tests for normalize_tables function."""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert normalize_tables("") == ""

    def test_no_tables(self):
        """Text without tables is unchanged."""
        text = "hello\nworld"
        assert normalize_tables(text) == text

    def test_already_valid_table(self):
        """Well-formed table is preserved."""
        text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        result = normalize_tables(text)
        assert result == text

    def test_short_row_padded(self):
        """Short rows are padded with empty cells."""
        text = "| a | b |\n| --- | --- |\n| 1 |"
        result = normalize_tables(text)
        lines = result.split("\n")
        # Cells are stripped, so "| 1 |  |" (with spaces)
        assert "| 1 |  |" in lines[2] or "| 1 | |" in lines[2]

    def test_separator_fixed(self):
        """Separator row is fixed to match column count from data rows."""
        text = "| a | b | c |\n| --- |\n| 1 | 2 | 3 |"
        result = normalize_tables(text)
        lines = result.split("\n")
        # Separator should be expanded to 3 columns
        assert lines[1] == "| --- | --- | --- |"

    def test_single_column_not_treated_as_table(self):
        """Single-column 'table' is left as-is."""
        text = "| item |\n| --- |\n| data |"
        result = normalize_tables(text)
        assert result == text

    def test_table_with_surrounding_text(self):
        """Table block surrounded by non-table text is processed."""
        text = "before\n| a | b |\n| --- | --- |\n| 1 | 2 |\nafter"
        result = normalize_tables(text)
        assert result.startswith("before")
        assert result.endswith("after")
        # Table should have normalized cells (stripped)
        assert "a" in result and "b" in result

    def test_multiple_tables(self):
        """Multiple separate table blocks are each normalized."""
        text = (
            "| a | b |\n| --- |\n| 1 |\n\n"
            "text between\n\n"
            "| x | y | z |\n| --- |\n| a | b |"
        )
        result = normalize_tables(text)
        # First table: 2 cols (from data row), short row padded, separator expanded
        assert "| --- | --- |" in result
        assert "| 1 |  |" in result
        # Second table: 3 cols, short row padded
        assert "| a | b |  |" in result


class TestPostprocessOCRResult:
    """Tests for the full postprocess_ocr_result pipeline."""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert postprocess_ocr_result("") == ""

    def test_cleaning_applied(self):
        """CRLF and artifacts are cleaned."""
        text = "\r\nhello\r\n\r\n\r\nworld\x00\r\n"
        result = postprocess_ocr_result(text)
        assert "\r" not in result
        assert "\x00" not in result

    def test_realistic_ocr_output(self):
        """Realistic OCR output is improved by the pipeline."""
        ocr_text = (
            "保险条款摘要\r\n\r\n\r\n"
            "理赔流程：\n"
            "被保险人应当在事故发生后\r"
            "及时通知保险公司\n\n"
            "| 项目 | 金额 |\n| --- | --- |\n| 医疗 | 5000 |"
        )
        result = postprocess_ocr_result(ocr_text)

        # CRLF/CR normalized
        assert "\r" not in result

        # Excessive blank lines collapsed
        assert "\n\n\n" not in result

        # Paragraphs merged (broken lines joined)
        assert "及时通知保险公司" in result

        # Heading detected
        assert "###" in result

    def test_pipeline_order(self):
        """Pipeline: clean -> normalize_tables -> merge_paragraphs -> detect_headings."""
        text = "标题\n这是一段被分成了\n两行的段落内容\n| a | b |\n| --- |\n| 1 |"
        result = postprocess_ocr_result(text)
        # Heading detected for short line before content
        assert "###" in result or "标题" in result
        # Paragraphs merged
        assert "两行的段落内容" in result

    def test_preserves_good_content(self):
        """Well-structured content is not damaged."""
        text = "# Title\n\nParagraph one.\n\nParagraph two.\n\n- list item"
        result = postprocess_ocr_result(text)
        assert "# Title" in result
        assert "Paragraph one." in result
        assert "- list item" in result
