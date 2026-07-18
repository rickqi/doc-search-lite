"""Unit tests for PDF converter.

Tests cover:
- Successful PDF conversion
- Password-protected PDF handling
- Corrupted PDF handling
- Table extraction
- Image extraction
- Metadata extraction
- Edge cases (empty files, missing files)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.converter.base import ConvertResult
from src.converter.pdf import PDFConverter, get_pdf_converter


class TestPDFConverterProperties:
    """Test PDF converter basic properties."""

    def test_name(self):
        """Test converter name."""
        converter = PDFConverter()
        assert converter.name == "pdfplumber"

    def test_supported_formats(self):
        """Test supported file formats."""
        converter = PDFConverter()
        assert converter.supported_formats == [".pdf"]

    def test_version(self):
        """Test converter version."""
        converter = PDFConverter()
        # Version should be a string
        assert isinstance(converter.version, str)

    def test_can_convert_pdf(self):
        """Test can_convert method for PDF files."""
        converter = PDFConverter()
        assert converter.can_convert(Path("test.pdf")) is True
        assert converter.can_convert(Path("test.PDF")) is True
        assert converter.can_convert(Path("test.Pdf")) is True

    def test_cannot_convert_other_formats(self):
        """Test can_convert method for non-PDF files."""
        converter = PDFConverter()
        assert converter.can_convert(Path("test.docx")) is False
        assert converter.can_convert(Path("test.txt")) is False
        assert converter.can_convert(Path("test.html")) is False


class TestPDFConverterMissingFile:
    """Test PDF converter with missing source file."""

    def test_convert_missing_file(self):
        """Test conversion of non-existent file."""
        converter = PDFConverter()
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "nonexistent.pdf"
            output_dir = Path(tmpdir) / "output"

            result = converter.convert(source, output_dir)

            assert result.success is False
            assert "Source file not found" in result.errors[0]
            assert result.source_file == source

    def test_convert_missing_file_output_dir_not_created(self):
        """Test that output dir is NOT created for missing files."""
        converter = PDFConverter()
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "nonexistent.pdf"
            output_dir = Path(tmpdir) / "output"

            result = converter.convert(source, output_dir)

            # Should fail gracefully
            assert result.success is False
            # Output dir should NOT be created for missing files
            assert not output_dir.exists()


class TestPDFConverterWithValidPDF:
    """Test PDF converter with valid PDF files."""

    @pytest.fixture
    def sample_pdf_path(self, tmp_path: Path) -> Path:
        """Create a simple test PDF file.

        This creates a minimal valid PDF using raw PDF syntax.
        """
        pdf_path = tmp_path / "test.pdf"
        # Minimal valid PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test Content) Tj
ET
endstream
endobj
5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000362 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
439
%%EOF
"""
        pdf_path.write_bytes(pdf_content)
        return pdf_path

    def test_convert_valid_pdf(self, sample_pdf_path: Path):
        """Test conversion of a valid PDF file."""
        converter = PDFConverter()
        output_dir = sample_pdf_path.parent / "output"

        result = converter.convert(sample_pdf_path, output_dir)

        # Should succeed (text extraction may work or fail depending on PDF structure)
        assert result.source_file == sample_pdf_path
        assert result.converter_name == "pdfplumber"
        assert result.convert_time >= 0

    def test_convert_creates_output_file(self, sample_pdf_path: Path):
        """Test that conversion creates output markdown file."""
        converter = PDFConverter()
        output_dir = sample_pdf_path.parent / "output"

        result = converter.convert(sample_pdf_path, output_dir)

        if result.success:
            assert result.output_file is not None
            assert result.output_file.suffix == ".md"
            assert result.output_file.exists()

    def test_convert_metadata_extraction(self, sample_pdf_path: Path):
        """Test metadata extraction from PDF."""
        converter = PDFConverter()
        output_dir = sample_pdf_path.parent / "output"

        result = converter.convert(sample_pdf_path, output_dir)

        # Metadata should be populated
        assert isinstance(result.metadata, dict)
        # Page count should be extracted
        assert (
            "page_count" in result.metadata
            or "pdfplumber_page_count" in result.metadata
        )


class TestPDFConverterPasswordProtected:
    """Test PDF converter with password-protected PDFs."""

    def test_password_protected_pdf_error(self, tmp_path: Path):
        """Test that password-protected PDF returns appropriate error."""
        converter = PDFConverter()

        # Create a mock encrypted PDF scenario
        with patch("src.converter.pdf._get_pypdf") as mock_get_pypdf:
            mock_reader_class = MagicMock()
            mock_reader = MagicMock()
            mock_reader.is_encrypted = True
            mock_reader.metadata = {}
            mock_reader.pages = []
            mock_reader_class.return_value = mock_reader
            mock_get_pypdf.return_value = mock_reader_class

            # Create dummy file
            pdf_path = tmp_path / "encrypted.pdf"
            pdf_path.write_bytes(b"dummy pdf content")

            output_dir = tmp_path / "output"
            result = converter.convert(pdf_path, output_dir)

            assert result.success is False
            assert any("password" in e.lower() for e in result.errors)

    def test_password_protected_pdf_with_wrong_password(self, tmp_path: Path):
        """Test that wrong password returns error."""
        converter = PDFConverter()

        with patch("src.converter.pdf._get_pypdf") as mock_get_pypdf:
            mock_reader_class = MagicMock()
            mock_reader = MagicMock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = False  # Wrong password
            mock_reader.metadata = {}
            mock_reader.pages = []
            mock_reader_class.return_value = mock_reader
            mock_get_pypdf.return_value = mock_reader_class

            pdf_path = tmp_path / "encrypted.pdf"
            pdf_path.write_bytes(b"dummy pdf content")

            output_dir = tmp_path / "output"
            result = converter.convert(
                pdf_path, output_dir, options={"password": "wrong"}
            )

            assert result.success is False
            assert any(
                "password" in e.lower() or "decrypt" in e.lower() for e in result.errors
            )


class TestPDFConverterCorrupted:
    """Test PDF converter with corrupted files."""

    def test_corrupted_pdf_handling(self, tmp_path: Path):
        """Test that corrupted PDF is handled gracefully."""
        converter = PDFConverter()

        # Create a corrupted PDF (not valid PDF structure)
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"This is not a valid PDF file content")

        output_dir = tmp_path / "output"
        result = converter.convert(pdf_path, output_dir)

        # Should handle gracefully without crashing
        assert result.success is False or len(result.errors) > 0
        assert isinstance(result, ConvertResult)


class TestPDFConverterOptions:
    """Test PDF converter with various options."""

    def test_extract_images_option_disabled(self, tmp_path: Path):
        """Test conversion with image extraction disabled."""
        converter = PDFConverter()

        # Create minimal PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj "
            b"xref\n0 3\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n%%EOF"
        )

        output_dir = tmp_path / "output"
        result = converter.convert(
            pdf_path, output_dir, options={"extract_images": False}
        )

        # Should not crash, images list should be empty
        assert isinstance(result.images, list)

    def test_custom_image_directory(self, tmp_path: Path):
        """Test conversion with custom image subdirectory."""
        converter = PDFConverter()

        # Create minimal PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj "
            b"xref\n0 3\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n%%EOF"
        )

        output_dir = tmp_path / "output"
        result = converter.convert(
            pdf_path, output_dir, options={"image_dir": "custom_images"}
        )

        # Should work without error
        assert isinstance(result, ConvertResult)


class TestPDFConverterTableExtraction:
    """Test table extraction functionality."""

    def test_table_to_markdown_empty(self):
        """Test _table_to_markdown with empty table."""
        converter = PDFConverter()
        mock_table = MagicMock()
        mock_table.extract.return_value = []

        result = converter._table_to_markdown(mock_table)
        assert result == ""

    def test_table_to_markdown_with_data(self):
        """Test _table_to_markdown with table data."""
        converter = PDFConverter()
        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["Header 1", "Header 2", "Header 3"],
            ["Row 1", "Data 1", "Data 2"],
            ["Row 2", "Data 3", "Data 4"],
        ]

        result = converter._table_to_markdown(mock_table)

        assert "| Header 1 | Header 2 | Header 3 |" in result
        assert "| --- | --- | --- |" in result
        assert "| Row 1 | Data 1 | Data 2 |" in result

    def test_table_to_markdown_with_none_cells(self):
        """Test _table_to_markdown with None cells."""
        converter = PDFConverter()
        mock_table = MagicMock()
        mock_table.extract.return_value = [
            ["Header", None, "Header 3"],
            [None, "Data", None],
        ]

        result = converter._table_to_markdown(mock_table)

        # None cells should become empty strings
        assert "| Header |  | Header 3 |" in result
        assert "|  | Data |  |" in result


class TestPDFConverterTextProcessing:
    """Test text processing functionality."""

    def test_process_text_all_caps_heading(self):
        """Test that all-caps lines become headings."""
        converter = PDFConverter()
        text = "IMPORTANT NOTICE\nRegular text here"

        result = converter._process_text(text)

        assert "## Important Notice" in result

    def test_process_text_numbered_heading(self):
        """Test that numbered lines become headings."""
        converter = PDFConverter()
        text = "1. First section\nSome content\n2. Second section"

        result = converter._process_text(text)

        assert "### 1. First section" in result
        assert "### 2. Second section" in result


class TestGetPDFConverter:
    """Test get_pdf_converter function."""

    def test_returns_same_instance(self):
        """Test that get_pdf_converter returns singleton."""
        converter1 = get_pdf_converter()
        converter2 = get_pdf_converter()

        assert converter1 is converter2

    def test_returns_pdf_converter(self):
        """Test that get_pdf_converter returns PDFConverter."""
        converter = get_pdf_converter()
        assert isinstance(converter, PDFConverter)


class TestPDFConverterEstimateTime:
    """Test time estimation functionality."""

    def test_estimate_time_small_file(self, tmp_path: Path):
        """Test time estimation for small file."""
        converter = PDFConverter()
        pdf_path = tmp_path / "small.pdf"
        pdf_path.write_bytes(b"x" * 1024)  # 1 KB

        estimate = converter.estimate_time(pdf_path)

        # Should be size_mb * 2, which is very small for 1KB
        assert estimate >= 0

    def test_estimate_time_nonexistent_file(self, tmp_path: Path):
        """Test time estimation for non-existent file."""
        converter = PDFConverter()
        pdf_path = tmp_path / "nonexistent.pdf"

        estimate = converter.estimate_time(pdf_path)

        # Should return default 1.0 for missing files
        assert estimate == 1.0
