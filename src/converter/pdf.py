"""PDF to Markdown converter using pdfplumber and pypdf.

This module provides PDF conversion functionality with:
- Text extraction preserving structure
- Table extraction as Markdown tables
- Image extraction and saving
- Password-protected PDF detection
- Corrupted PDF handling
"""

import hashlib
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from src.converter.base import Converter, ConvertResult

from .table_fix import fix_table_alignment

# Suppress pdfminer's verbose FontBBox warnings.
# These occur when PDFs embed fonts with missing/invalid FontBBox descriptors
# (e.g., None instead of 4 floats). The warnings are harmless — text extraction
# still works — but they flood stderr with hundreds of identical messages.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Lazy imports to handle missing optional dependencies
_pdfplumber = None
_pypdf = None
_PILImage = None


def _get_pdfplumber():
    """Lazy load pdfplumber module."""
    global _pdfplumber
    if _pdfplumber is None:
        try:
            import pdfplumber

            _pdfplumber = pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF conversion. "
                "Install it with: pip install pdfplumber"
            )
    return _pdfplumber


def _get_pypdf():
    """Lazy load pypdf module."""
    global _pypdf
    if _pypdf is None:
        try:
            from pypdf import PdfReader

            _pypdf = PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF conversion. "
                "Install it with: pip install pypdf"
            )
    return _pypdf


def _get_pil():
    """Lazy load PIL Image module."""
    global _PILImage
    if _PILImage is None:
        try:
            from PIL import Image

            _PILImage = Image
        except ImportError:
            raise ImportError(
                "Pillow is required for image extraction. "
                "Install it with: pip install pillow"
            )
    return _PILImage


def _sanitize_metadata_value(value) -> str:
    """Convert pypdf ByteStringObject and other non-serializable types to str."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return value.decode("latin-1")
    return str(value) if value is not None else ""


class PDFConverter(Converter):
    """PDF to Markdown converter using pdfplumber.

    Features:
    - Text extraction with structure preservation
    - Table extraction as Markdown tables
    - Image extraction and saving to output directory
    - Password-protected PDF detection (returns error, doesn't crash)
    - Corrupted PDF handling
    - Heading hierarchy preservation

    Attributes:
        name: Converter name ("pdfplumber")
        version: Version string
        supported_formats: List of supported file extensions
    """

    @property
    def name(self) -> str:
        """Get the converter name."""
        return "pdfplumber"

    @property
    def version(self) -> str:
        """Get the converter version."""
        try:
            import pdfplumber

            return getattr(pdfplumber, "__version__", "0.11.9")
        except ImportError:
            return "unknown"

    @property
    def supported_formats(self) -> list[str]:
        """Get list of supported file extensions."""
        return [".pdf"]

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: dict | None = None,
    ) -> ConvertResult:
        """Convert a PDF file to Markdown.

        Args:
            source: Path to the source PDF file.
            output_dir: Directory to save output files (markdown and images).
            options: Optional conversion options:
                - extract_images: bool (default: False) - Extract images from PDF
                - image_dir: str (default: "images") - Subdirectory for images
                - password: str (optional) - Password for encrypted PDFs
                - dpi: int (default: 150) - DPI for image extraction

        Returns:
            ConvertResult containing:
            - success: True if conversion succeeded
            - markdown: Converted markdown content
            - images: List of extracted image paths
            - errors: List of error messages if any
            - metadata: PDF metadata
        """
        start_time = time.time()
        options = options or {}
        errors: list[str] = []
        images: list[Path] = []
        metadata: dict = {}

        # Validate source file
        if not source.exists():
            return ConvertResult(
                success=False,
                markdown="",
                errors=[f"Source file not found: {source}"],
                source_file=source,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Image extraction settings
        extract_images = options.get("extract_images", False)
        image_subdir = options.get("image_dir", "images")
        password = options.get("password")
        dpi = options.get("dpi", 150)

        try:
            # First check if PDF is encrypted using pypdf
            PdfReader = _get_pypdf()
            reader = None
            try:
                reader = PdfReader(str(source))

                # Check if encrypted
                if reader.is_encrypted:
                    # Try password dictionary if no explicit password given
                    if password is None:
                        from src.utils.password_dict import PasswordDictionary
                        pdict = PasswordDictionary()
                        found_pwd = None
                        for candidate in pdict:
                            try:
                                if reader.decrypt(candidate) > 0:
                                    found_pwd = candidate
                                    logger.info(
                                        "PDF decrypted with dictionary password (len=%d): %s",
                                        len(candidate),
                                        "***" if candidate else "(empty)",
                                    )
                                    break
                            except Exception:
                                continue

                        if found_pwd is None:
                            if hasattr(reader, 'stream') and reader.stream:
                                reader.stream.close()
                            return ConvertResult(
                                success=False,
                                markdown="",
                                errors=[
                                    "PDF is password-protected. "
                                    "None of the %d dictionary passwords worked. "
                                    "Provide password in options['password']."
                                    % len(pdict)
                                ],
                                source_file=source,
                                converter_name=self.name,
                                converter_version=self.version,
                                convert_time=time.time() - start_time,
                                metadata={"encrypted": True},
                            )
                        password = found_pwd
                    else:
                        # Try to decrypt with provided password
                        if not reader.decrypt(password):
                            if hasattr(reader, 'stream') and reader.stream:
                                reader.stream.close()
                            return ConvertResult(
                                success=False,
                                markdown="",
                                errors=["Failed to decrypt PDF. Invalid password."],
                                source_file=source,
                                converter_name=self.name,
                                converter_version=self.version,
                                convert_time=time.time() - start_time,
                                metadata={"encrypted": True},
                            )

                # Extract metadata
                if reader.metadata:
                    metadata["pdf_metadata"] = {
                        key: _sanitize_metadata_value(reader.metadata.get(key, ""))
                        for key in [
                            "/Title",
                            "/Author",
                            "/Subject",
                            "/Creator",
                            "/Producer",
                        ]
                    }
                metadata["page_count"] = len(reader.pages)

            except Exception as e:
                # pypdf failed, but pdfplumber might still work
                errors.append(f"Warning: pypdf metadata extraction failed: {e}")
            finally:
                if reader is not None and hasattr(reader, 'stream') and reader.stream:
                    reader.stream.close()

            # Main conversion using pdfplumber
            pdfplumber = _get_pdfplumber()
            markdown_parts: list[str] = []

            try:
                with pdfplumber.open(str(source), password=password) as pdf:
                    metadata["pdfplumber_page_count"] = len(pdf.pages)

                    # Extract metadata from pdfplumber
                    if pdf.metadata:
                        metadata["pdfplumber_metadata"] = {
                            k: _sanitize_metadata_value(v)
                            for k, v in pdf.metadata.items()
                        }

                    # Create image directory if extracting images
                    image_dir = None
                    if extract_images:
                        image_dir = output_dir / image_subdir
                        image_dir.mkdir(parents=True, exist_ok=True)

                    # Process each page
                    for page_num, page in enumerate(pdf.pages, start=1):
                        page_markdown = self._process_page(
                            page=page,
                            page_num=page_num,
                            extract_images=extract_images,
                            image_dir=image_dir if extract_images else None,
                            images_list=images,
                            source_hash=hashlib.md5(str(source).encode()).hexdigest()[
                                :8
                            ],
                            dpi=dpi,
                        )
                        markdown_parts.append(page_markdown)

                    markdown_content = "\n\n".join(markdown_parts)
                    # Fix table alignment issues
                    markdown_content = fix_table_alignment(markdown_content)

            except Exception as e:
                error_msg = str(e).lower()

                # Check for common error patterns
                if "password" in error_msg or "encrypted" in error_msg:
                    return ConvertResult(
                        success=False,
                        markdown="",
                        errors=["PDF is password-protected and cannot be opened."],
                        source_file=source,
                        converter_name=self.name,
                        converter_version=self.version,
                        convert_time=time.time() - start_time,
                        metadata={"encrypted": True},
                    )
                elif (
                    "corrupt" in error_msg
                    or "invalid" in error_msg
                    or "damaged" in error_msg
                ):
                    return ConvertResult(
                        success=False,
                        markdown="",
                        errors=[f"PDF file appears to be corrupted: {e}"],
                        source_file=source,
                        converter_name=self.name,
                        converter_version=self.version,
                        convert_time=time.time() - start_time,
                    )
                else:
                    # Unknown error, but try to continue with partial results
                    errors.append(f"Warning: PDF processing error: {e}")
                    markdown_content = (
                        "\n\n".join(markdown_parts) if markdown_parts else ""
                    )

            # Determine success
            success = len(markdown_content.strip()) > 0

            # Save output markdown file
            output_file = output_dir / f"{source.stem}.md"
            if success:
                output_file.write_text(markdown_content, encoding="utf-8")

            return ConvertResult(
                success=success,
                markdown=markdown_content,
                images=images,
                metadata=metadata,
                errors=errors,
                source_file=source,
                output_file=output_file if success else None,
                convert_time=time.time() - start_time,
                converter_name=self.name,
                converter_version=self.version,
            )

        except Exception as e:
            return ConvertResult(
                success=False,
                markdown="",
                errors=[f"Unexpected error during PDF conversion: {e}"],
                source_file=source,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

    def _process_page(
        self,
        page,
        page_num: int,
        extract_images: bool,
        image_dir: Path | None,
        images_list: list[Path],
        source_hash: str,
        dpi: int,
    ) -> str:
        """Process a single PDF page and convert to Markdown.

        Args:
            page: pdfplumber page object
            page_num: Page number (1-indexed)
            extract_images: Whether to extract images
            image_dir: Directory to save images (if extracting)
            images_list: List to append extracted image paths
            source_hash: Hash of source file for unique image naming
            dpi: DPI for image extraction

        Returns:
            Markdown string for the page
        """
        parts: list[str] = []

        # Add page header
        parts.append(f"<!-- Page {page_num} -->")

        # Extract tables first (they have positions)
        tables = page.find_tables()
        table_regions = set()

        for _table_idx, table in enumerate(tables):
            try:
                table_md = self._table_to_markdown(table)
                if table_md:
                    parts.append(f"\n{table_md}\n")
                # Record table bbox to exclude from text
                if table.bbox:
                    table_regions.add(table.bbox)
            except Exception:
                pass  # Skip failed table extractions

        # Extract text (excluding table regions)
        try:
            text = page.extract_text()
            if text:
                # Clean up text and detect headings
                processed_text = self._process_text(text)
                parts.append(processed_text)
        except Exception:
            pass  # Skip failed text extraction

        # Extract images
        if extract_images and image_dir:
            try:
                page_images = self._extract_page_images(
                    page=page,
                    page_num=page_num,
                    image_dir=image_dir,
                    source_hash=source_hash,
                    dpi=dpi,
                )
                images_list.extend(page_images)
            except Exception:
                pass  # Skip failed image extraction

        return "\n".join(parts)

    def _table_to_markdown(self, table) -> str:
        """Convert a pdfplumber table to Markdown format.

        Args:
            table: pdfplumber table object

        Returns:
            Markdown table string
        """
        try:
            extracted = table.extract()
            if not extracted or len(extracted) == 0:
                return ""

            lines: list[str] = []

            # Process header row
            header = extracted[0]
            if header:
                # Clean header cells
                header = [str(cell).strip() if cell else "" for cell in header]
                lines.append("| " + " | ".join(header) + " |")
                lines.append("| " + " | ".join(["---"] * len(header)) + " |")

            # Process data rows
            for row in extracted[1:]:
                if row:
                    # Clean row cells
                    row = [
                        str(cell).strip().replace("\n", " ") if cell else ""
                        for cell in row
                    ]
                    lines.append("| " + " | ".join(row) + " |")

            return "\n".join(lines)
        except Exception:
            return ""

    def _process_text(self, text: str) -> str:
        """Process extracted text and detect/convert headings.

        Args:
            text: Raw extracted text

        Returns:
            Processed text with potential heading markers
        """
        lines = text.split("\n")
        processed_lines: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect potential headings based on patterns
            # All caps lines (likely titles)
            if line.isupper() and len(line) < 100:
                processed_lines.append(f"## {line.title()}")
            # Lines starting with numbers followed by period or space (numbered headings)
            elif len(line) > 0 and line[0].isdigit():
                if re.match(r"^\d+[\.\)]\s+", line):
                    processed_lines.append(f"### {line}")
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _extract_page_images(
        self,
        page,
        page_num: int,
        image_dir: Path,
        source_hash: str,
        dpi: int,
    ) -> list[Path]:
        """Extract images from a PDF page.

        Args:
            page: pdfplumber page object
            page_num: Page number (1-indexed)
            image_dir: Directory to save images
            source_hash: Hash of source file for unique naming
            dpi: DPI for image extraction

        Returns:
            List of extracted image file paths
        """
        images: list[Path] = []
        _get_pil()

        # Get page images
        try:
            # Convert page to image
            im = page.to_image(resolution=dpi)
            if im and im.original:
                # Save full page image
                image_filename = f"{source_hash}_page_{page_num:03d}.png"
                image_path = image_dir / image_filename
                im.original.save(image_path, "PNG")
                images.append(image_path)
        except Exception:
            pass  # Image extraction not critical

        return images


# Module-level converter instance for convenience
_default_converter: PDFConverter | None = None


def get_pdf_converter() -> PDFConverter:
    """Get the default PDF converter instance.

    Returns:
        PDFConverter instance
    """
    global _default_converter
    if _default_converter is None:
        _default_converter = PDFConverter()
    return _default_converter
