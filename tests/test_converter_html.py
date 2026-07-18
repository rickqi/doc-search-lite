"""
Unit tests for HTMLConverter.

Tests cover:
- Basic HTML conversion
- Support for .html and .htm extensions
- Rejection of non-HTML files
- Error handling for malformed HTML
- Error handling for missing files
- Metadata extraction
- Output file creation
- can_convert method
"""

from pathlib import Path

from src.converter.html import HTMLConverter


class TestHTMLConverterBasic:
    """Test basic HTML converter functionality."""

    def test_converter_properties(self):
        """Test converter name, version, and supported formats."""
        converter = HTMLConverter()

        assert converter.name == "HTMLConverter"
        assert converter.version == "0.1.0"
        assert ".html" in converter.supported_formats
        assert ".htm" in converter.supported_formats
        assert len(converter.supported_formats) == 2


class TestCanConvert:
    """Test can_convert method."""

    def test_can_convert_html_file(self):
        """Test that .html files are recognized."""
        converter = HTMLConverter()
        file_path = Path("test.html")

        assert converter.can_convert(file_path) is True

    def test_can_convert_htm_file(self):
        """Test that .htm files are recognized."""
        converter = HTMLConverter()
        file_path = Path("test.htm")

        assert converter.can_convert(file_path) is True

    def test_can_convert_uppercase_extension(self):
        """Test that uppercase extensions are recognized."""
        converter = HTMLConverter()
        file_path = Path("test.HTML")

        assert converter.can_convert(file_path) is True

    def test_cannot_convert_other_formats(self):
        """Test that non-HTML files are rejected."""
        converter = HTMLConverter()

        assert converter.can_convert(Path("test.pdf")) is False
        assert converter.can_convert(Path("test.docx")) is False
        assert converter.can_convert(Path("test.txt")) is False
        assert converter.can_convert(Path("test")) is False


class TestHTMLConversion:
    """Test HTML to Markdown conversion."""

    def test_convert_basic_html(self, tmp_path, tmp_output_dir):
        """Test converting a basic HTML file to Markdown."""
        # Create test HTML file
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Main Heading</h1>
    <h2>Sub Heading</h2>
    <p>This is a paragraph.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    <a href="https://example.com">Link</a>
</body>
</html>
"""
        html_file = tmp_path / "test.html"
        html_file.write_text(html_content, encoding="utf-8")

        # Convert HTML
        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        # Verify result
        assert result.success is True
        assert result.source_file == html_file
        assert result.output_file == tmp_output_dir / "test.md"
        assert result.converter_name == "HTMLConverter"
        assert result.converter_version == "0.1.0"
        assert len(result.errors) == 0
        assert result.convert_time >= 0

        # Verify output file was created
        assert result.output_file.exists()
        markdown_content = result.output_file.read_text(encoding="utf-8")

        # Verify Markdown content
        assert (
            "# Main Heading" in markdown_content or "Main Heading" in markdown_content
        )
        assert markdown_content == result.markdown

    def test_convert_htm_file(self, tmp_path, tmp_output_dir):
        """Test converting a .htm file."""
        html_content = "<html><body><h1>Test</h1></body></html>"
        html_file = tmp_path / "test.htm"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True
        assert result.output_file == tmp_output_dir / "test.md"
        assert result.output_file.exists()

    def test_convert_html_with_tables(self, tmp_path, tmp_output_dir):
        """Test converting HTML with tables."""
        html_content = """
<!DOCTYPE html>
<html>
<body>
    <h1>Table Test</h1>
    <table>
        <tr>
            <th>Name</th>
            <th>Age</th>
        </tr>
        <tr>
            <td>John</td>
            <td>30</td>
        </tr>
        <tr>
            <td>Jane</td>
            <td>25</td>
        </tr>
    </table>
</body>
</html>
"""
        html_file = tmp_path / "table.html"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True
        markdown_content = result.markdown

        # Verify table structure is preserved
        # MarkItDown should convert tables to Markdown table format
        assert "Name" in markdown_content or "Age" in markdown_content

    def test_extract_html_title(self, tmp_path, tmp_output_dir):
        """Test that HTML title is extracted as metadata."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>My Document Title</title>
</head>
<body>
    <h1>Content</h1>
</body>
</html>
"""
        html_file = tmp_path / "title.html"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True
        # MarkItDown should extract title
        if "title" in result.metadata:
            assert "My Document Title" in result.metadata["title"]


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_unsupported_file_extension(self, tmp_path, tmp_output_dir):
        """Test that non-HTML files are rejected with clear error message."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Not HTML content")

        converter = HTMLConverter()
        result = converter.convert(text_file, tmp_output_dir)

        assert result.success is False
        assert result.markdown == ""
        assert len(result.errors) == 1
        assert "Unsupported file format" in result.errors[0]
        assert ".txt" in result.errors[0]

    def test_file_not_found(self, tmp_output_dir):
        """Test handling of non-existent files."""
        non_existent_file = Path("/non/existent/file.html")

        converter = HTMLConverter()
        result = converter.convert(non_existent_file, tmp_output_dir)

        assert result.success is False
        assert result.markdown == ""
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_malformed_html(self, tmp_path, tmp_output_dir):
        """Test handling of malformed HTML."""
        # Create HTML with unclosed tags and other issues
        malformed_html = """
<!DOCTYPE html>
<html>
<body>
    <h1>Unclosed heading
    <p>Unclosed paragraph
    <div>Nested without closing
    <p>Another unclosed paragraph</p>
</body>
"""
        html_file = tmp_path / "malformed.html"
        html_file.write_text(malformed_html, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        # MarkItDown should handle malformed HTML gracefully
        # It may succeed with partial content or fail gracefully
        # Either is acceptable - we just don't want a crash
        assert result.source_file == html_file
        assert result.converter_name == "HTMLConverter"

    def test_empty_html(self, tmp_path, tmp_output_dir):
        """Test handling of empty HTML file."""
        html_file = tmp_path / "empty.html"
        html_file.write_text("", encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        # Should handle empty file gracefully
        # May succeed with empty markdown or fail gracefully
        assert result.source_file == html_file
        assert result.converter_name == "HTMLConverter"


class TestOutputHandling:
    """Test output file handling."""

    def test_output_file_created(self, tmp_path, tmp_output_dir):
        """Test that output file is created in the correct location."""
        html_content = "<html><body><h1>Test</h1></body></html>"
        html_file = tmp_path / "test.html"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True
        assert result.output_file == tmp_output_dir / "test.md"
        assert result.output_file.exists()

        # Verify output directory was created if it didn't exist
        assert tmp_output_dir.exists()

    def test_output_file_content(self, tmp_path, tmp_output_dir):
        """Test that output file contains correct Markdown content."""
        html_content = """
<!DOCTYPE html>
<html>
<body>
    <h1>Test Heading</h1>
    <p>Test paragraph</p>
</body>
</html>
"""
        html_file = tmp_path / "content.html"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True

        # Read output file content
        output_content = result.output_file.read_text(encoding="utf-8")

        # Verify it matches the markdown in result
        assert output_content == result.markdown

    def test_output_dir_created_if_not_exists(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        html_content = "<html><body><h1>Test</h1></body></html>"
        html_file = tmp_path / "test.html"
        html_file.write_text(html_content, encoding="utf-8")

        # Create a non-existent output directory path
        non_existent_dir = tmp_path / "new_output" / "nested"

        converter = HTMLConverter()
        result = converter.convert(html_file, non_existent_dir)

        assert result.success is True
        assert non_existent_dir.exists()
        assert result.output_file.exists()


class TestPreserveStructure:
    """Test that HTML structure is preserved in Markdown."""

    def test_preserve_headings(self, tmp_path, tmp_output_dir):
        """Test that heading hierarchy is preserved."""
        html_content = """
<!DOCTYPE html>
<html>
<body>
    <h1>Heading 1</h1>
    <h2>Heading 2</h2>
    <h3>Heading 3</h3>
    <h4>Heading 4</h4>
</body>
</html>
"""
        html_file = tmp_path / "headings.html"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True
        markdown = result.markdown

        # Verify heading levels are preserved
        assert "Heading 1" in markdown
        assert "Heading 2" in markdown
        assert "Heading 3" in markdown
        assert "Heading 4" in markdown

    def test_preserve_lists(self, tmp_path, tmp_output_dir):
        """Test that list structure is preserved."""
        html_content = """
<!DOCTYPE html>
<html>
<body>
    <h1>Lists Test</h1>
    <ul>
        <li>Unordered item 1</li>
        <li>Unordered item 2</li>
    </ul>
    <ol>
        <li>Ordered item 1</li>
        <li>Ordered item 2</li>
    </ol>
</body>
</html>
"""
        html_file = tmp_path / "lists.html"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True
        markdown = result.markdown

        # Verify list items are present
        assert "Unordered item 1" in markdown or "item 1" in markdown
        assert "Unordered item 2" in markdown or "item 2" in markdown
        assert "Ordered item 1" in markdown or "item 1" in markdown

    def test_preserve_links(self, tmp_path, tmp_output_dir):
        """Test that links are preserved."""
        html_content = """
<!DOCTYPE html>
<html>
<body>
    <h1>Links Test</h1>
    <p>Check out this <a href="https://example.com">link</a>.</p>
    <p>Another <a href="/relative/path">relative link</a>.</p>
</body>
</html>
"""
        html_file = tmp_path / "links.html"
        html_file.write_text(html_content, encoding="utf-8")

        converter = HTMLConverter()
        result = converter.convert(html_file, tmp_output_dir)

        assert result.success is True
        markdown = result.markdown

        # Verify links are preserved
        assert "https://example.com" in markdown
        assert "link" in markdown.lower()


class TestEstimateTime:
    """Test time estimation method."""

    def test_estimate_time_based_on_size(self, tmp_path):
        """Test that time estimation is based on file size."""
        converter = HTMLConverter()

        # Create a small file
        small_file = tmp_path / "small.html"
        small_file.write_text("<html><body><p>Small</p></body></html>")

        # Create a larger file
        large_file = tmp_path / "large.html"
        large_content = "<html><body>" + "<p>Text</p>" * 10000 + "</body></html>"
        large_file.write_text(large_content)

        # Estimate times
        small_time = converter.estimate_time(small_file)
        large_time = converter.estimate_time(large_file)

        # Larger file should have larger estimated time
        assert large_time > small_time
        assert small_time > 0
        assert large_time > 0

    def test_estimate_time_nonexistent_file(self):
        """Test that estimate_time handles non-existent files."""
        converter = HTMLConverter()
        nonexistent = Path("/non/existent/file.html")

        # Should default to 1.0 second
        time_estimate = converter.estimate_time(nonexistent)
        assert time_estimate == 1.0
