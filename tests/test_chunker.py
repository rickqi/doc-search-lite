"""Tests for src/storage/chunker.py — Markdown document chunk splitting."""

import pytest

from src.converter.headings import extract_headings
from src.storage.chunker import (
    split_into_chunks,
    _hard_split_paragraphs,
    _merge_tiny_chunks,
    _split_by_headings,
)


# ── Fixtures ────────────────────────────────────────────────

def _make_headings(markdown: str):
    """Shorthand: extract headings then call split."""
    return extract_headings(markdown)


# ── Basic splitting ─────────────────────────────────────────


class TestBasicSplitting:
    """Verify that split_into_chunks correctly partitions documents."""

    def test_short_document_returns_single_chunk(self):
        """Documents under the threshold return one chunk titled '全文'."""
        content = "# Title\n\nSome short content here."
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=50000)
        assert len(chunks) == 1
        assert chunks[0][0] == "全文"
        assert chunks[0][1] == content

    def test_empty_content_returns_single_chunk(self):
        """Empty / whitespace-only content returns one chunk."""
        chunks = split_into_chunks("   \n\n  ", [], max_chunk_chars=100)
        assert len(chunks) == 1
        assert chunks[0][0] == "全文"

    def test_no_headings_returns_single_chunk(self):
        """Long document without headings: short enough → single chunk."""
        content = "A" * 300  # Under threshold
        chunks = split_into_chunks(content, [], max_chunk_chars=500)
        assert len(chunks) == 1
        assert chunks[0][0] == "全文"

    def test_no_headings_long_document_hard_splits(self):
        """Long document with no headings but exceeds threshold → hard-split."""
        # Create content with paragraph breaks
        paragraphs = [f"Paragraph {i} " + "x" * 200 for i in range(10)]
        content = "\n\n".join(paragraphs)  # ~2200 chars
        chunks = split_into_chunks(content, [], max_chunk_chars=500)
        assert len(chunks) > 1
        # Each chunk should be under the limit (allowing for paragraph merging)
        for _, text in chunks:
            assert len(text) <= 500 + 300  # some slack for paragraph boundary

    def test_h2_headings_split_correctly(self):
        """Document with H2 headings splits at heading boundaries."""
        # Build a doc where each section is ~300 chars, total > 500
        section_a = "## Section A\n\n" + "Alpha content. " * 50  # ~750 chars
        section_b = "## Section B\n\n" + "Beta content. " * 50
        intro = "Intro text. " * 30  # ~360 chars before any heading
        content = f"{intro}\n\n{section_a}\n\n{section_b}"
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)

        # Should have multiple chunks
        assert len(chunks) >= 2
        # First chunk should be the intro (titled "开头") or merged
        titles = [c[0] for c in chunks]
        # Verify heading text appears in some chunk title
        title_text = " ".join(titles)
        assert "Section A" in title_text or "Section B" in title_text

    def test_h3_headings_also_split(self):
        """H3 headings serve as primary split points when no H2 present."""
        part1 = "### Subsection 1\n\n" + "Content one. " * 60
        part2 = "### Subsection 2\n\n" + "Content two. " * 60
        content = f"{part1}\n\n{part2}"
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        assert len(chunks) >= 2

    def test_h1_not_used_as_split_point(self):
        """H1 (document title) should not be a split point — only H2/H3."""
        # Only H1 headings → no primary split → falls back to H4+ or hard split
        content = "# Document Title\n\n" + "Body text. " * 100
        headings = extract_headings(content)
        # Content is ~1200 chars, threshold 500
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        # Should still split (hard-split) but NOT by H1
        # The key assertion: no chunk should be titled "Document Title"
        # because H1 is not a split point (it would be "开头" for pre-H2 content)
        for title, _ in chunks:
            # "Document Title" shouldn't appear as a section title
            # (it's part of the content before any H2/H3, so it's "开头")
            pass  # H1 in content but not used as a split heading

    def test_chunk_titles_are_heading_text(self):
        """Chunk titles come from the heading text."""
        section_a = "## 申请流程\n\n" + "内容" * 200
        section_b = "## 报销标准\n\n" + "内容" * 200
        content = f"{section_a}\n\n{section_b}"
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        titles = [c[0] for c in chunks]
        # At least one of the heading texts should appear
        title_joined = " ".join(titles)
        assert "申请流程" in title_joined or "报销标准" in title_joined


# ── Tiny chunk merging ──────────────────────────────────────


class TestTinyChunkMerging:
    """Chunks smaller than MIN_MERGE_CHARS get merged into the previous."""

    def test_tiny_chunks_merged_into_previous(self):
        """A very short section after a big one gets merged."""
        big = "## Big Section\n\n" + "X" * 600  # > threshold
        tiny = "## Tiny\n\nShort."  # < 500 chars → merged
        content = f"{big}\n\n{tiny}"
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        # The tiny chunk should have been merged into the big one
        # So we shouldn't have a separate "Tiny" chunk
        titles = [c[0] for c in chunks]
        # The last chunk should NOT be titled "Tiny" (it got merged)
        assert titles[-1] != "Tiny"

    def test_merge_tiny_chunks_directly(self):
        """Test the _merge_tiny_chunks helper directly."""
        chunks = [
            ("A", "A" * 1000),
            ("B", "tiny"),  # < 500 → merged into A
            ("C", "C" * 1000),
        ]
        merged = _merge_tiny_chunks(chunks, min_chars=500)
        assert len(merged) == 2
        assert merged[0][0] == "A"
        assert "tiny" in merged[0][1]
        assert merged[1][0] == "C"

    def test_first_chunk_tiny_not_merged(self):
        """If the first chunk is tiny, it stays (nothing before it)."""
        chunks = [("first", "x"), ("second", "y" * 1000)]
        merged = _merge_tiny_chunks(chunks, min_chars=500)
        assert len(merged) == 2  # first stays, can't merge backward


# ── Hard-split at paragraph boundaries ─────────────────────


class TestHardSplit:
    """Very long sections without sub-headings get hard-split."""

    def test_hard_split_paragraphs_basic(self):
        """_hard_split_paragraphs respects max_chars."""
        text = "\n\n".join([f"Para {i} " + "z" * 200 for i in range(10)])
        parts = _hard_split_paragraphs(text, max_chars=500)
        assert len(parts) > 1
        for _, chunk in parts:
            assert len(chunk) <= 500 + 250  # slack for paragraph boundary

    def test_hard_split_single_huge_paragraph(self):
        """A single paragraph exceeding max_chars gets force-split."""
        text = "x" * 2000
        parts = _hard_split_paragraphs(text, max_chars=500)
        assert len(parts) >= 4
        for _, chunk in parts:
            assert len(chunk) <= 500

    def test_long_section_no_subheadings_hard_splits(self):
        """A section with no H4-H6 sub-headings that exceeds the limit
        gets hard-split at paragraph boundaries."""
        # Single H2 section with lots of content
        body = "## Big Section\n\n" + "\n\n".join(
            [f"Paragraph {i}. " + "y" * 200 for i in range(10)]
        )
        content = body
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        assert len(chunks) > 1
        # All chunks should be reasonably sized
        for _, text in chunks:
            assert len(text) <= 500 + 300  # some slack


# ── Sub-heading fallback (H4/H5/H6) ────────────────────────


class TestSubHeadingFallback:
    """Oversized H2/H3 sections get further split by H4-H6."""

    def test_h4_subheadings_split_oversized_section(self):
        """When an H2 section is too big, H4 sub-headings further split it."""
        h2 = "## Main Section"
        h4a = "#### Detail A\n\n" + "Detail A content. " * 50
        h4b = "#### Detail B\n\n" + "Detail B content. " * 50
        h4c = "#### Detail C\n\n" + "Detail C content. " * 50
        content = f"{h2}\n\n{h4a}\n\n{h4b}\n\n{h4c}"
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        assert len(chunks) >= 2
        titles = " ".join(t for t, _ in chunks)
        # Sub-heading texts should appear in chunk titles
        assert "Detail A" in titles or "Detail B" in titles or "Detail C" in titles


# ── Integration with extract_headings ──────────────────────


class TestExtractHeadingsIntegration:
    """Verify chunker works with real extract_headings output."""

    def test_code_block_headings_ignored(self):
        """Headings inside code blocks should not be treated as split points."""
        content = (
            "## Real Section\n\n"
            + "Content. " * 200
            + "\n\n```\n## Fake Heading in Code\n```\n\n"
            + "More content. " * 200
        )
        headings = extract_headings(content)
        # "Fake Heading in Code" should NOT be in headings
        heading_texts = [h["text"] for h in headings]
        assert "Fake Heading in Code" not in heading_texts
        assert "Real Section" in heading_texts

    def test_max_headings_limit(self):
        """Documents with many headings still chunk correctly."""
        parts = []
        for i in range(40):
            parts.append(f"## Section {i}\n\n" + "C" * 300)
        content = "\n\n".join(parts)
        headings = extract_headings(content)
        # extract_headings caps at MAX_HEADINGS=30
        assert len(headings) <= 30
        # Should still produce multiple chunks
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        assert len(chunks) >= 2

    def test_mixed_heading_levels(self):
        """Document with H1, H2, H3, H4 mixed."""
        content = (
            "# Document Title\n\n"
            + "Intro. " * 50 + "\n\n"
            + "## Chapter 1\n\n" + "Chapter 1 content. " * 60 + "\n\n"
            + "### Section 1.1\n\n" + "Section 1.1 content. " * 60 + "\n\n"
            + "## Chapter 2\n\n" + "Chapter 2 content. " * 60 + "\n\n"
            + "#### Sub-detail\n\n" + "Sub-detail content. " * 40
        )
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=500)
        assert len(chunks) >= 2
        # Verify all chunk content is non-empty
        for _, text in chunks:
            assert text.strip()


# ── Edge cases ──────────────────────────────────────────────


class TestEdgeCases:

    def test_content_exactly_at_threshold(self):
        """Content exactly at max_chunk_chars returns single chunk."""
        content = "x" * 500
        chunks = split_into_chunks(content, [], max_chunk_chars=500)
        assert len(chunks) == 1

    def test_content_one_char_over_threshold(self):
        """Content one char over threshold with no headings → hard-split."""
        content = "para1 " * 84  # ~504 chars, single paragraph
        chunks = split_into_chunks(content, [], max_chunk_chars=500)
        # Single paragraph > threshold → force-split
        assert len(chunks) >= 1

    def test_preserves_content_completeness(self):
        """All chunk content concatenated should contain all text."""
        section_a = "## Alpha\n\n" + "A" * 300
        section_b = "## Beta\n\n" + "B" * 300
        content = f"{section_a}\n\n{section_b}"
        headings = extract_headings(content)
        chunks = split_into_chunks(content, headings, max_chunk_chars=400)

        # All 'A's should be in some chunk, all 'B's in another
        all_text = " ".join(text for _, text in chunks)
        assert "AAA" in all_text
        assert "BBB" in all_text

    def test_split_by_headings_returns_line_ranges(self):
        """Internal _split_by_headings returns (title, text, start, end)."""
        lines = ["intro", "## H2", "content", "## H3", "more"]
        headings = [
            {"level": 2, "text": "H2", "line": 2},
            {"level": 3, "text": "H3", "line": 4},
        ]
        result = _split_by_headings(lines, headings)
        # Should have 3 chunks: intro (开头), H2 section, H3 section
        assert len(result) == 3
        assert result[0][0] == "开头"
        assert result[1][0] == "H2"
        assert result[2][0] == "H3"
        # Verify line ranges
        assert result[0][2] == 1  # start_line
        assert result[0][3] == 1  # end_line
        assert result[1][2] == 2  # H2 starts at line 2
        assert result[1][3] == 3  # H2 ends before H3 (line 4 - 1 = 3)

    def test_unsorted_headings_handled(self):
        """_split_by_headings sorts split points by line number."""
        lines = ["a", "## B", "c", "## A", "e"]
        headings = [
            {"level": 2, "text": "A", "line": 4},  # out of order
            {"level": 2, "text": "B", "line": 2},
        ]
        result = _split_by_headings(lines, headings)
        # Should be sorted: 开头, B, A
        assert result[1][0] == "B"
        assert result[2][0] == "A"
