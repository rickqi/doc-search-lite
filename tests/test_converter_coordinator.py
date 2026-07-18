"""Unit tests for ConverterCoordinator.

Tests cover:
- Converter selection based on file extension
- Conversion delegation to appropriate converters
- Scanned PDF detection and OCR fallback
- Unsupported format handling
- Custom converter registration
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.converter.base import ConvertResult, Converter
from src.converter.coordinator import (
    ConverterCoordinator,
    UnsupportedFormatError,
    get_coordinator,
)
from src.converter.html import HTMLConverter
from src.converter.office import OfficeConverter
from src.converter.pdf import PDFConverter


class MockConverter(Converter):
    """Mock converter for testing custom converters."""

    def __init__(self, extensions: list[str], name: str = "MockConverter"):
        self._extensions = [ext.lower() for ext in extensions]
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_formats(self) -> list[str]:
        return self._extensions

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: dict | None = None,
    ) -> ConvertResult:
        return ConvertResult(
            success=True,
            markdown="Mock content",
            source_file=source,
            converter_name=self.name,
            converter_version=self.version,
        )


class TestConverterCoordinatorProperties:
    """Test ConverterCoordinator basic properties."""

    def test_supported_extensions(self):
        """Test that supported_extensions returns all registered extensions."""
        coordinator = ConverterCoordinator()
        extensions = coordinator.supported_extensions

        # Should include PDF, Office, and HTML extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".pptx" in extensions
        assert ".xlsx" in extensions
        assert ".html" in extensions
        assert ".htm" in extensions

    def test_can_convert_supported_formats(self):
        """Test can_convert for supported formats."""
        coordinator = ConverterCoordinator()

        assert coordinator.can_convert(Path("test.pdf")) is True
        assert coordinator.can_convert(Path("test.PDF")) is True
        assert coordinator.can_convert(Path("test.docx")) is True
        assert coordinator.can_convert(Path("test.pptx")) is True
        assert coordinator.can_convert(Path("test.xlsx")) is True
        assert coordinator.can_convert(Path("test.doc")) is True  # Legacy Word
        assert coordinator.can_convert(Path("test.xls")) is True  # Legacy Excel
        assert coordinator.can_convert(Path("test.html")) is True
        assert coordinator.can_convert(Path("test.htm")) is True

    def test_cannot_convert_unsupported_formats(self):
        """Test can_convert for unsupported formats."""
        coordinator = ConverterCoordinator()

        assert coordinator.can_convert(Path("test.rtf")) is False
        assert coordinator.can_convert(Path("test.exe")) is False
        # .zip is now supported via ArchiveConverter
        assert coordinator.can_convert(Path("test.zip")) is True
        assert coordinator.can_convert(Path("test.xml")) is False
        assert coordinator.can_convert(Path("test.json")) is False


class TestConverterCoordinatorGetConverter:
    """Test get_converter method."""

    def test_get_converter_pdf(self):
        """Test get_converter returns PDFConverter for .pdf files."""
        coordinator = ConverterCoordinator()
        converter = coordinator.get_converter(Path("document.pdf"))
        assert isinstance(converter, PDFConverter)

    def test_get_converter_docx(self):
        """Test get_converter returns OfficeConverter for .docx files."""
        coordinator = ConverterCoordinator()
        converter = coordinator.get_converter(Path("document.docx"))
        assert isinstance(converter, OfficeConverter)

    def test_get_converter_pptx(self):
        """Test get_converter returns OfficeConverter for .pptx files."""
        coordinator = ConverterCoordinator()
        converter = coordinator.get_converter(Path("presentation.pptx"))
        assert isinstance(converter, OfficeConverter)

    def test_get_converter_xlsx(self):
        """Test get_converter returns OfficeConverter for .xlsx files."""
        coordinator = ConverterCoordinator()
        converter = coordinator.get_converter(Path("spreadsheet.xlsx"))
        assert isinstance(converter, OfficeConverter)

    def test_get_converter_html(self):
        """Test get_converter returns HTMLConverter for .html files."""
        coordinator = ConverterCoordinator()
        converter = coordinator.get_converter(Path("page.html"))
        assert isinstance(converter, HTMLConverter)

    def test_get_converter_htm(self):
        """Test get_converter returns HTMLConverter for .htm files."""
        coordinator = ConverterCoordinator()
        converter = coordinator.get_converter(Path("page.htm"))
        assert isinstance(converter, HTMLConverter)

    def test_get_converter_case_insensitive(self):
        """Test get_converter is case-insensitive for extensions."""
        coordinator = ConverterCoordinator()

        assert isinstance(coordinator.get_converter(Path("test.PDF")), PDFConverter)
        assert isinstance(coordinator.get_converter(Path("test.Pdf")), PDFConverter)
        assert isinstance(coordinator.get_converter(Path("test.DOCX")), OfficeConverter)
        assert isinstance(coordinator.get_converter(Path("test.HTML")), HTMLConverter)

    def test_get_converter_unsupported_format(self):
        """Test get_converter raises UnsupportedFormatError for unsupported formats."""
        coordinator = ConverterCoordinator()

        with pytest.raises(UnsupportedFormatError) as exc_info:
            coordinator.get_converter(Path("document.xyz"))

        assert ".xyz" in str(exc_info.value)
        assert "Unsupported" in str(exc_info.value)


class TestConverterCoordinatorConvert:
    """Test convert method."""

    def test_convert_missing_file(self, tmp_path: Path):
        """Test conversion of non-existent file."""
        coordinator = ConverterCoordinator()
        source = tmp_path / "nonexistent.pdf"
        output_dir = tmp_path / "output"

        result = coordinator.convert(source, output_dir)

        assert result.success is False
        assert "Source file not found" in result.errors[0]
        assert result.source_file == source

    def test_convert_unsupported_format(self, tmp_path: Path):
        """Test conversion of unsupported file format."""
        coordinator = ConverterCoordinator()

        # Create a file with unsupported extension
        source = tmp_path / "document.xyz"
        source.write_text("test content")
        output_dir = tmp_path / "output"

        result = coordinator.convert(source, output_dir)

        assert result.success is False
        assert any("Unsupported" in e for e in result.errors)

    def test_convert_delegates_to_pdf_converter(self, tmp_path: Path):
        """Test that convert delegates to PDFConverter for PDF files."""
        coordinator = ConverterCoordinator()

        # Create a minimal PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
trailer<</Size 4/Root 1 0 R>>
startxref
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        output_dir = tmp_path / "output"

        result = coordinator.convert(pdf_path, output_dir)

        assert result.source_file == pdf_path
        assert "coordinator_converter" in result.metadata
        # PDFConverter name is "pdfplumber"
        assert result.metadata["coordinator_converter"] == "pdfplumber"

    def test_convert_delegates_to_html_converter(self, tmp_path: Path):
        """Test that convert delegates to HTMLConverter for HTML files."""
        coordinator = ConverterCoordinator()

        html_path = tmp_path / "test.html"
        html_path.write_text("<html><body><h1>Test</h1></body></html>")
        output_dir = tmp_path / "output"

        result = coordinator.convert(html_path, output_dir)

        assert result.source_file == html_path
        assert "coordinator_converter" in result.metadata
        assert result.metadata["coordinator_converter"] == "HTMLConverter"

    def test_convert_sets_coordinator_metadata(self, tmp_path: Path):
        """Test that convert adds coordinator metadata to result."""
        coordinator = ConverterCoordinator()

        html_path = tmp_path / "test.html"
        html_path.write_text("<html><body>Test</body></html>")
        output_dir = tmp_path / "output"

        result = coordinator.convert(html_path, output_dir)

        assert result.metadata.get("coordinator_used") is True


class TestConverterCoordinatorScannedPDFDetection:
    """Test scanned PDF detection and OCR fallback."""

    def test_is_scanned_pdf_with_minimal_text(self):
        """Test _is_scanned_pdf returns True for minimal text."""
        coordinator = ConverterCoordinator(scanned_pdf_threshold=50)

        # Create a result with minimal text
        result = ConvertResult(
            success=True,
            markdown="Short text",  # Less than 50 chars
            metadata={},
        )

        assert coordinator._is_scanned_pdf(result, page_count=1) is True

    def test_is_scanned_pdf_with_sufficient_text(self):
        """Test _is_scanned_pdf returns False for sufficient text."""
        coordinator = ConverterCoordinator(scanned_pdf_threshold=50)

        # Create a result with sufficient text
        result = ConvertResult(
            success=True,
            markdown="x" * 100,  # More than 50 chars
            metadata={},
        )

        assert coordinator._is_scanned_pdf(result, page_count=1) is False

    def test_is_scanned_pdf_multi_page_average(self):
        """Test _is_scanned_pdf uses per-page average."""
        coordinator = ConverterCoordinator(scanned_pdf_threshold=50)

        # 100 chars across 3 pages = ~33 chars per page (below threshold)
        result = ConvertResult(
            success=True,
            markdown="x" * 100,
            metadata={},
        )

        assert coordinator._is_scanned_pdf(result, page_count=3) is True

        # 200 chars across 3 pages = ~67 chars per page (above threshold)
        result2 = ConvertResult(
            success=True,
            markdown="x" * 200,
            metadata={},
        )

        assert coordinator._is_scanned_pdf(result2, page_count=3) is False

    def test_is_scanned_pdf_unsuccessful_result(self):
        """Test _is_scanned_pdf returns False for unsuccessful result."""
        coordinator = ConverterCoordinator()

        result = ConvertResult(
            success=False,
            markdown="",
            errors=["Some error"],
        )

        assert coordinator._is_scanned_pdf(result, page_count=1) is False

    def test_ocr_disabled_no_fallback(self, tmp_path: Path):
        """Test that OCR fallback is skipped when disabled."""
        coordinator = ConverterCoordinator(enable_ocr_fallback=False)

        # Create a minimal PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
trailer<</Size 4/Root 1 0 R>>
startxref
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        output_dir = tmp_path / "output"

        result = coordinator.convert(pdf_path, output_dir)

        # Should not use OCR
        assert result.ocr_used is False

    def test_disable_ocr_fallback_option(self, tmp_path: Path):
        """Test disable_ocr_fallback option."""
        coordinator = ConverterCoordinator(enable_ocr_fallback=True)

        # Create a minimal PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
trailer<</Size 4/Root 1 0 R>>
startxref
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        output_dir = tmp_path / "output"

        result = coordinator.convert(
            pdf_path, output_dir, options={"disable_ocr_fallback": True}
        )

        # Should not use OCR
        assert result.ocr_used is False

    def test_scanned_pdf_without_ocr_config(self, tmp_path: Path):
        """Test scanned PDF handling when OCR is not configured."""
        coordinator = ConverterCoordinator(
            scanned_pdf_threshold=100,  # High threshold
            enable_ocr_fallback=True,
            ocr_config=None,  # No OCR config
        )

        # Create a minimal PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
trailer<</Size 4/Root 1 0 R>>
startxref
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        output_dir = tmp_path / "output"

        result = coordinator.convert(pdf_path, output_dir)

        # Should complete without OCR (OCR service is None)
        assert isinstance(result, ConvertResult)
        # If PDF appears scanned but no OCR available, should have error message
        if result.metadata.get("page_count", 1) == 1:
            # Minimal text PDF without OCR config
            pass  # Just verify it doesn't crash


class TestConverterCoordinatorCustomConverter:
    """Test custom converter registration."""

    def test_register_custom_converter(self):
        """Test registering a custom converter."""
        coordinator = ConverterCoordinator()
        custom_converter = MockConverter([".xyz", ".abc"])

        coordinator.register_custom_converter(custom_converter)

        assert coordinator.can_convert(Path("test.xyz")) is True
        assert coordinator.can_convert(Path("test.abc")) is True
        assert coordinator.get_converter(Path("test.xyz")) == custom_converter

    def test_register_custom_converter_override(self):
        """Test overriding existing converter with custom converter."""
        coordinator = ConverterCoordinator()
        custom_converter = MockConverter([".pdf"], name="CustomPDF")

        coordinator.register_custom_converter(custom_converter, override=True)

        # Custom converter should now handle .pdf
        assert coordinator.get_converter(Path("test.pdf")) == custom_converter

    def test_register_custom_converter_no_override_error(self):
        """Test that registering without override raises error for existing extension."""
        coordinator = ConverterCoordinator()
        custom_converter = MockConverter([".pdf"], name="CustomPDF")

        with pytest.raises(ValueError) as exc_info:
            coordinator.register_custom_converter(custom_converter, override=False)

        assert "already registered" in str(exc_info.value)


class TestConverterCoordinatorEstimateTime:
    """Test time estimation."""

    def test_estimate_time_delegates_to_converter(self, tmp_path: Path):
        """Test that estimate_time delegates to appropriate converter."""
        coordinator = ConverterCoordinator()

        # Create a test file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"x" * 1024)

        estimate = coordinator.estimate_time(pdf_path)

        # Should return a non-negative value
        assert estimate >= 0

    def test_estimate_time_unsupported_format(self, tmp_path: Path):
        """Test estimate_time raises error for unsupported format."""
        coordinator = ConverterCoordinator()

        unsupported_path = tmp_path / "test.xyz"
        unsupported_path.write_text("test")

        with pytest.raises(UnsupportedFormatError):
            coordinator.estimate_time(unsupported_path)


class TestGetCoordinator:
    """Test get_coordinator function."""

    def test_returns_coordinator_instance(self):
        """Test that get_coordinator returns ConverterCoordinator."""
        coordinator = get_coordinator()
        assert isinstance(coordinator, ConverterCoordinator)

    def test_returns_same_instance(self):
        """Test that get_coordinator returns singleton."""
        # Reset the singleton
        import src.converter.coordinator as coordinator_module

        coordinator_module._default_coordinator = None

        coordinator1 = get_coordinator()
        coordinator2 = get_coordinator()

        assert coordinator1 is coordinator2


class TestUnsupportedFormatError:
    """Test UnsupportedFormatError exception."""

    def test_error_message_includes_extension(self):
        """Test that error message includes the unsupported extension."""
        error = UnsupportedFormatError(".xyz", [".pdf", ".docx"])

        assert ".xyz" in str(error)
        assert "Unsupported" in str(error)

    def test_error_message_includes_supported_extensions(self):
        """Test that error message includes supported extensions."""
        error = UnsupportedFormatError(".xyz", [".pdf", ".docx", ".html"])

        assert ".pdf" in str(error)
        assert ".docx" in str(error)
        assert ".html" in str(error)

    def test_error_attributes(self):
        """Test error attributes are set correctly."""
        error = UnsupportedFormatError(".xyz", [".pdf", ".docx"])

        assert error.extension == ".xyz"
        assert error.supported_extensions == [".pdf", ".docx"]


class TestConverterCoordinatorHTMLDetection:
    """Test HTML content detection for mismatched file extensions."""

    def test_is_html_content_with_doctype(self, tmp_path: Path):
        """Test _is_html_content detects <!DOCTYPE> prefix."""
        coordinator = ConverterCoordinator()

        # Create file with HTML content
        html_file = tmp_path / "test.html"
        html_file.write_text("<!DOCTYPE html><html><body>Test</body></html>")

        assert coordinator._is_html_content(html_file) is True

    def test_is_html_content_with_lowercase_html(self, tmp_path: Path):
        """Test _is_html_content detects <html prefix."""
        coordinator = ConverterCoordinator()

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        assert coordinator._is_html_content(html_file) is True

    def test_is_html_content_with_uppercase_html(self, tmp_path: Path):
        """Test _is_html_content detects <HTML prefix."""
        coordinator = ConverterCoordinator()

        html_file = tmp_path / "test.html"
        html_file.write_text("<HTML><body>Test</body></HTML>")

        assert coordinator._is_html_content(html_file) is True

    def test_is_html_content_with_xml_declaration(self, tmp_path: Path):
        """Test _is_html_content detects <?xml> prefix."""
        coordinator = ConverterCoordinator()

        html_file = tmp_path / "test.html"
        html_file.write_text('<?xml version="1.0"?><html><body>Test</body></html>')

        assert coordinator._is_html_content(html_file) is True

    def test_is_html_content_with_pdf(self, tmp_path: Path):
        """Test _is_html_content returns False for real PDF."""
        coordinator = ConverterCoordinator()

        # Create a minimal PDF
        pdf_file = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
trailer<</Size 4/Root 1 0 R>>
startxref
%%EOF"""
        pdf_file.write_bytes(pdf_content)

        assert coordinator._is_html_content(pdf_file) is False

    def test_is_html_content_with_text_file(self, tmp_path: Path):
        """Test _is_html_content returns False for plain text."""
        coordinator = ConverterCoordinator()

        text_file = tmp_path / "test.txt"
        text_file.write_text("This is just plain text")

        assert coordinator._is_html_content(text_file) is False

    def test_is_html_content_with_missing_file(self, tmp_path: Path):
        """Test _is_html_content returns False for non-existent file."""
        coordinator = ConverterCoordinator()

        missing_file = tmp_path / "nonexistent.txt"

        # Should not raise exception, just return False
        assert coordinator._is_html_content(missing_file) is False

    def test_convert_html_in_pdf_extension(self, tmp_path: Path):
        """Test that HTML content in .pdf file is routed to HTMLConverter."""
        coordinator = ConverterCoordinator()

        # Create a .pdf file with HTML content
        html_pdf = tmp_path / "document.pdf"
        html_pdf.write_text(
            "<!DOCTYPE html><html><body><h1>HTML Content</h1></body></html>"
        )
        output_dir = tmp_path / "output"

        result = coordinator.convert(html_pdf, output_dir)

        # Should succeed
        assert result.success is True
        # Should use HTMLConverter
        assert result.metadata["coordinator_converter"] == "HTMLConverter"
        # Should NOT use OCR (because we routed to HTMLConverter, not PDFConverter)
        assert result.ocr_used is False

    def test_convert_real_pdf_not_html(self, tmp_path: Path):
        """Test that real PDF files are still routed to PDFConverter."""
        coordinator = ConverterCoordinator()

        # Create a real PDF
        real_pdf = tmp_path / "document.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
trailer<</Size 4/Root 1 0 R>>
startxref
%%EOF"""
        real_pdf.write_bytes(pdf_content)
        output_dir = tmp_path / "output"

        result = coordinator.convert(real_pdf, output_dir)

        # Should succeed
        assert result.success is True
        # Should use PDFConverter (pdfplumber)
        assert result.metadata["coordinator_converter"] == "pdfplumber"

    def test_convert_html_file_normal(self, tmp_path: Path):
        """Test that regular .html files are still handled normally."""
        coordinator = ConverterCoordinator()

        html_file = tmp_path / "document.html"
        html_file.write_text("<!DOCTYPE html><html><body>Test</body></html>")
        output_dir = tmp_path / "output"

        result = coordinator.convert(html_file, output_dir)

        # Should succeed
        assert result.success is True
        # Should use HTMLConverter
        assert result.metadata["coordinator_converter"] == "HTMLConverter"

    def test_convert_html_in_pdf_various_markers(self, tmp_path: Path):
        """Test HTML-in-PDF detection with various HTML markers."""
        coordinator = ConverterCoordinator()
        output_dir = tmp_path / "output"

        # Test with <!DOCTYPE>
        pdf1 = tmp_path / "doctype.pdf"
        pdf1.write_text("<!DOCTYPE html><html><body>Test</body></html>")
        result1 = coordinator.convert(pdf1, output_dir)
        assert result1.success is True
        assert result1.metadata["coordinator_converter"] == "HTMLConverter"

        # Test with <html>
        pdf2 = tmp_path / "lowercase.pdf"
        pdf2.write_text("<html><body>Test</body></html>")
        result2 = coordinator.convert(pdf2, output_dir)
        assert result2.success is True
        assert result2.metadata["coordinator_converter"] == "HTMLConverter"

        # Test with <HTML>
        pdf3 = tmp_path / "uppercase.pdf"
        pdf3.write_text("<HTML><body>Test</body></HTML>")
        result3 = coordinator.convert(pdf3, output_dir)
        assert result3.success is True
        assert result3.metadata["coordinator_converter"] == "HTMLConverter"

        # Test with <?xml>
        pdf4 = tmp_path / "xml.pdf"
        pdf4.write_text('<?xml version="1.0"?><html><body>Test</body></html>')
        result4 = coordinator.convert(pdf4, output_dir)
        assert result4.success is True
        assert result4.metadata["coordinator_converter"] == "HTMLConverter"
