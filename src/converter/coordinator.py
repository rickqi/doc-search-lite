"""
Converter coordinator that automatically selects the appropriate converter
based on file extension and handles scanned PDF detection with OCR fallback.

This module provides the ConverterCoordinator class which:
- Maps file extensions to appropriate converters
- Auto-selects converter based on file extension
- Detects scanned PDFs (empty/minimal text content) and triggers OCR
- Handles unsupported formats with clear error messages
"""

import gc
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from .base import ConvertResult, Converter
from .csv import CSVConverter
from .html import HTMLConverter
from .image import ImageConverter
from .msg import MsgConverter
from .ocr import OCRService, OCRServiceConfig
from .office import OfficeConverter
from .pdf import PDFConverter
from .text import TextConverter
from .archive import ArchiveConverter
from .tag_extractor import TagExtractor
from .ocr_postprocess import postprocess_ocr_result
from src.storage.convert_db import PIPELINE_VERSION

if TYPE_CHECKING:
    from src.stats.usage_tracker import UsageTracker

logger = logging.getLogger(__name__)

# Lazy imports for OCR image generation
_coordinator_pdfplumber = None
_coordinator_PIL_Image = None


def _get_coordinator_pdfplumber():
    """Lazy load pdfplumber for page rendering."""
    global _coordinator_pdfplumber
    if _coordinator_pdfplumber is None:
        try:
            import pdfplumber

            _coordinator_pdfplumber = pdfplumber
        except ImportError:
            pass
    return _coordinator_pdfplumber


def _get_coordinator_pil():
    """Lazy load PIL Image for page rendering."""
    global _coordinator_PIL_Image
    if _coordinator_PIL_Image is None:
        try:
            from PIL import Image

            _coordinator_PIL_Image = Image
        except ImportError:
            pass
    return _coordinator_PIL_Image


class UnsupportedFormatError(Exception):
    """Exception raised when file format is not supported."""

    def __init__(self, extension: str, supported_extensions: List[str]):
        self.extension = extension
        self.supported_extensions = supported_extensions
        super().__init__(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {', '.join(supported_extensions)}"
        )


class ConverterCoordinator:
    """
    Coordinates document converters based on file extension.

    This class automatically selects the appropriate converter for a given
    file based on its extension. It also supports detecting scanned PDFs
    and falling back to OCR when text extraction yields minimal content.

    Attributes:
        converters: Dictionary mapping file extensions to converter instances
        pdf_converter: PDFConverter instance
        office_converter: OfficeConverter instance
        html_converter: HTMLConverter instance
        ocr_service: Optional OCRService for scanned PDF handling

    Example:
        >>> coordinator = ConverterCoordinator()
        >>> result = coordinator.convert(Path("document.pdf"), Path("output"))
        >>> print(result.success)
    """

    # Default threshold for detecting scanned PDFs (characters per page)
    SCANNED_PDF_THRESHOLD: int = 50

    def __init__(
        self,
        ocr_config: Optional[OCRServiceConfig] = None,
        scanned_pdf_threshold: int = SCANNED_PDF_THRESHOLD,
        enable_ocr_fallback: bool = True,
        usage_tracker: Optional["UsageTracker"] = None,
    ):
        """
        Initialize the converter coordinator.

        Args:
            ocr_config: Optional OCR service configuration for scanned PDF handling
            scanned_pdf_threshold: Minimum characters per page to consider PDF as
                                   not scanned (default: 50)
            enable_ocr_fallback: Whether to enable OCR fallback for scanned PDFs
            usage_tracker: Optional UsageTracker for recording OCR token usage
                           with extended schema (cost_millicents, source_dir, session_id)
        """
        self._pdf_converter = PDFConverter()
        self._office_converter = OfficeConverter()
        self._html_converter = HTMLConverter()
        self._msg_converter = MsgConverter()

        self._ocr_config = ocr_config
        self._ocr_service: Optional[OCRService] = None
        self._scanned_pdf_threshold = scanned_pdf_threshold
        self._enable_ocr_fallback = enable_ocr_fallback
        self._usage_tracker = usage_tracker

        # Build extension to converter mapping
        self._converters: Dict[str, Converter] = {}
        self._register_converter(self._pdf_converter)
        self._register_converter(self._office_converter)
        self._register_converter(self._html_converter)
        self._register_converter(self._msg_converter)

        # Register additional converters
        self._image_converter = ImageConverter()
        self._csv_converter = CSVConverter()
        self._text_converter = TextConverter()
        self._register_converter(self._image_converter)
        self._register_converter(self._csv_converter)
        self._register_converter(self._text_converter)

        # Archive converter (needs coordinator reference for recursive conversion)
        self._archive_converter = ArchiveConverter(coordinator=self)
        self._register_converter(self._archive_converter)

    def _register_converter(self, converter: Converter) -> None:
        """
        Register a converter for its supported formats.

        Args:
            converter: Converter instance to register
        """
        for ext in converter.supported_formats:
            self._converters[ext.lower()] = converter

    @property
    def supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.

        Returns:
            List of supported file extensions (e.g., ['.pdf', '.docx', '.html'])
        """
        return list(self._converters.keys())

    def get_converter(self, source: Path) -> Converter:
        """
        Get the appropriate converter for a file based on its extension.

        Args:
            source: Path to the source file

        Returns:
            Converter instance that can handle the file

        Raises:
            UnsupportedFormatError: If no converter supports the file format
        """
        extension = source.suffix.lower()

        if extension not in self._converters:
            raise UnsupportedFormatError(extension, self.supported_extensions)

        return self._converters[extension]

    def can_convert(self, source: Path) -> bool:
        """
        Check if the coordinator can convert the given file.

        Args:
            source: Path to the source file

        Returns:
            True if the file format is supported
        """
        return source.suffix.lower() in self._converters

    def _get_ocr_service(self) -> Optional[OCRService]:
        """
        Get or create OCR service lazily.

        Returns:
            OCRService instance if configured, None otherwise
        """
        if not self._enable_ocr_fallback:
            return None

        if self._ocr_service is None and self._ocr_config is not None:
            self._ocr_service = OCRService(self._ocr_config)

        return self._ocr_service

    def _is_scanned_pdf(self, result: ConvertResult, page_count: int) -> bool:
        """
        Check if a PDF conversion result indicates a scanned document.

        A PDF is considered scanned if:
        - Conversion succeeded but yielded minimal text content
        - Average text per page is below threshold

        Args:
            result: ConvertResult from PDF conversion
            page_count: Number of pages in the PDF

        Returns:
            True if PDF appears to be scanned
        """
        if not result.success:
            return False

        # Get text content length
        text_length = len(result.markdown.strip())

        # Calculate average characters per page
        if page_count <= 0:
            page_count = 1

        avg_chars_per_page = text_length / page_count

        return avg_chars_per_page < self._scanned_pdf_threshold

    def _is_html_content(self, source: Path) -> bool:
        """
        Check if a file's content starts with HTML markup.

        This detects files that have been misnamed with the wrong extension
        (e.g., HTML files with .pdf extension).

        Args:
            source: Path to the file to check

        Returns:
            True if the file starts with HTML markers, False otherwise
        """
        try:
            # Read first 100 bytes to check for HTML markers
            with open(source, "rb") as f:
                header = f.read(100)

            # Check for common HTML markers
            html_markers = [
                b"<!DOCTYPE",
                b"<html",
                b"<HTML",
                b"<?xml",  # Some HTML files start with XML declaration
            ]

            for marker in html_markers:
                if header.startswith(marker):
                    return True

            return False
        except (IOError, OSError):
            # If we can't read the file, assume it's not HTML
            return False

    def _render_pdf_pages(
        self,
        source: Path,
        output_dir: Path,
        dpi: int = 150,
    ) -> tuple:
        """Render PDF pages to images for OCR.

        Returns:
            Tuple of (image_paths, temp_dir) where temp_dir should be
            cleaned up after use.
        """
        pdfplumber = _get_coordinator_pdfplumber()
        if pdfplumber is None:
            return [], None

        _get_coordinator_pil()  # Ensure PIL is available for .save()
        if _coordinator_PIL_Image is None:
            return [], None

        temp_dir = output_dir / "_ocr_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        image_paths: List[Path] = []
        try:
            with pdfplumber.open(str(source)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        im = page.to_image(resolution=dpi)
                        if im and im.original:
                            image_path = temp_dir / f"page_{page_num:03d}.png"
                            im.original.save(str(image_path), "PNG")
                            image_paths.append(image_path)
                    except Exception:
                        pass
        except Exception:
            pass

        return image_paths, temp_dir

    def _convert_scanned_pdf(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict],
        initial_result: ConvertResult,
    ) -> ConvertResult:
        """
        Handle scanned PDF by converting pages to images and running OCR.

        For large scanned PDFs (>50 MB), tries direct PDF submission to the
        layout_parsing API first (single API call for the entire PDF). Falls
        back to per-page render+OCR streaming if direct submission fails.

        Uses streaming mode: opens the PDF once, renders one page at a time,
        runs OCR immediately, then deletes the page image before proceeding
        to the next page. This keeps peak memory bounded to a single page
        image instead of accumulating all pages.

        Falls back to batch render-then-OCR if pdfplumber is not available
        for the streaming path.

        Args:
            source: Path to the source PDF
            output_dir: Directory to save output files
            options: Conversion options
            initial_result: Initial conversion result

        Returns:
            ConvertResult with OCR-extracted text
        """
        ocr_service = self._get_ocr_service()

        if ocr_service is None:
            # No OCR service available, return initial result
            initial_result.errors.append(
                "PDF appears to be scanned but OCR is not configured. "
                "Provide ocr_config to enable OCR fallback."
            )
            return initial_result

        # ── Large PDF direct submission path ────────────────────────────
        # For large scanned PDFs, send the entire PDF to layout_parsing API
        # in one call instead of rendering each page to PNG+OCR individually.
        # This is dramatically faster for 50+ page scanned documents.
        LARGE_PDF_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB
        if source.stat().st_size >= LARGE_PDF_THRESHOLD_BYTES:
            file_size_mb = source.stat().st_size // (1024 * 1024)
            logger.info(
                "Large PDF detected (%d MB), trying direct PDF OCR for %s",
                file_size_mb, source.name,
            )
            ocr_result = ocr_service.recognize_pdf(source)
            if ocr_result.success and ocr_result.text.strip():
                logger.info(
                    "Direct PDF OCR succeeded for %s (%.1fs)",
                    source.name, ocr_result.processing_time,
                )
                ocr_start_time = time.time()
                self._finalize_ocr_result(
                    ocr_texts=[ocr_result.text],
                    ocr_pages=1,
                    ocr_start_time=ocr_start_time,
                    total_token_usage=ocr_result.token_usage or {},
                    initial_result=initial_result,
                    source=source,
                    output_dir=output_dir,
                )
                return initial_result
            else:
                logger.warning(
                    "Direct PDF OCR failed for %s (%s), falling back to per-page OCR",
                    source.name, ocr_result.error or "unknown error",
                )

        # ── Streaming (interleaved render+OCR) path ────────────────────
        pdfplumber = _get_coordinator_pdfplumber()
        _get_coordinator_pil()

        if pdfplumber is not None and _coordinator_PIL_Image is not None:
            return self._convert_scanned_pdf_streaming(
                source=source,
                output_dir=output_dir,
                ocr_service=ocr_service,
                initial_result=initial_result,
            )

        # Fallback: batch render all pages then OCR (legacy path)
        return self._convert_scanned_pdf_batch(
            source=source,
            output_dir=output_dir,
            ocr_service=ocr_service,
            initial_result=initial_result,
        )

    def _convert_scanned_pdf_streaming(
        self,
        source: Path,
        output_dir: Path,
        ocr_service: OCRService,
        initial_result: ConvertResult,
        dpi: int = 150,
    ) -> ConvertResult:
        """
        Stream scanned PDF: render one page → OCR → cleanup → next page.

        Opens the PDF once in pdfplumber. For each page:
        1. Render to a temp PNG image
        2. Run OCR on that single image
        3. Delete the temp PNG immediately
        4. Explicitly release page objects and trigger GC

        This keeps peak memory at ~1 page image (~12 MB at 150 DPI)
        plus the pdfplumber PDF object, regardless of total page count.
        """
        pdfplumber = _get_coordinator_pdfplumber()

        temp_dir = output_dir / "_ocr_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        ocr_texts: List[str] = []
        ocr_start_time = time.time()
        ocr_pages = 0
        total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        try:
            with pdfplumber.open(str(source)) as pdf:  # type: ignore[union-attr]
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    image_path: Optional[Path] = None
                    try:
                        # Render single page to image
                        im = page.to_image(resolution=dpi)
                        if im and im.original:
                            image_path = temp_dir / f"page_{page_num:03d}.png"
                            im.original.save(str(image_path), "PNG")
                            # Release PIL image immediately
                            del im
                    except Exception:
                        del_im = locals().get("im")
                        if del_im is not None:
                            del del_im
                        continue

                    if image_path is None:
                        continue

                    # OCR the single page
                    try:
                        ocr_result = ocr_service.recognize(image_path)
                        if ocr_result.success and ocr_result.text.strip():
                            ocr_texts.append(ocr_result.text)
                            ocr_pages += 1
                        if ocr_result.token_usage:
                            total_token_usage["input_tokens"] += ocr_result.token_usage.get("input_tokens", 0)
                            total_token_usage["output_tokens"] += ocr_result.token_usage.get("output_tokens", 0)
                            total_token_usage["total_tokens"] += ocr_result.token_usage.get("total_tokens", 0)
                    except Exception as e:
                        initial_result.errors.append(f"OCR failed for page {page_num}/{total_pages}: {e}")

                    # Delete page image immediately after OCR
                    try:
                        image_path.unlink()
                    except Exception:
                        pass

                    # Periodic GC every 50 pages to release pdfplumber page caches
                    if page_num % 50 == 0:
                        gc.collect()

                        logger.debug(
                            "OCR streaming progress: %d/%d pages, %d succeeded",
                            page_num, total_pages, ocr_pages,
                        )

        except Exception as e:
            initial_result.errors.append(f"PDF rendering failed: {e}")
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

        # Combine OCR results and update initial_result
        if ocr_texts:
            self._finalize_ocr_result(
                ocr_texts=ocr_texts,
                ocr_pages=ocr_pages,
                ocr_start_time=ocr_start_time,
                total_token_usage=total_token_usage,
                initial_result=initial_result,
                source=source,
                output_dir=output_dir,
            )

        return initial_result

    def _convert_scanned_pdf_batch(
        self,
        source: Path,
        output_dir: Path,
        ocr_service: OCRService,
        initial_result: ConvertResult,
    ) -> ConvertResult:
        """
        Batch render-then-OCR path (legacy fallback).

        Renders ALL pages first, then OCRs sequentially.
        Used when streaming path is unavailable.
        """
        image_paths, temp_dir = self._render_pdf_pages(source, output_dir)

        if image_paths:
            try:
                ocr_texts: List[str] = []
                ocr_start_time = time.time()
                ocr_pages = 0

                total_token_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
                for image_path in image_paths:
                    try:
                        ocr_result = ocr_service.recognize(image_path)
                        if ocr_result.success and ocr_result.text.strip():
                            ocr_texts.append(ocr_result.text)
                            ocr_pages += 1
                        if ocr_result.token_usage:
                            total_token_usage["input_tokens"] += ocr_result.token_usage.get("input_tokens", 0)
                            total_token_usage["output_tokens"] += ocr_result.token_usage.get("output_tokens", 0)
                            total_token_usage["total_tokens"] += ocr_result.token_usage.get("total_tokens", 0)
                    except Exception as e:
                        initial_result.errors.append(f"OCR failed for {image_path}: {e}")

                if ocr_texts:
                    self._finalize_ocr_result(
                        ocr_texts=ocr_texts,
                        ocr_pages=ocr_pages,
                        ocr_start_time=ocr_start_time,
                        total_token_usage=total_token_usage,
                        initial_result=initial_result,
                        source=source,
                        output_dir=output_dir,
                    )

            finally:
                if temp_dir and temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass

        return initial_result

    def _finalize_ocr_result(
        self,
        ocr_texts: List[str],
        ocr_pages: int,
        ocr_start_time: float,
        total_token_usage: Dict[str, int],
        initial_result: ConvertResult,
        source: Path,
        output_dir: Path,
    ) -> None:
        """Combine OCR texts, postprocess, and update the ConvertResult in-place."""
        combined_text = "\n\n".join(ocr_texts)
        combined_text = postprocess_ocr_result(combined_text)

        # Update result
        initial_result.markdown = combined_text
        initial_result.ocr_used = True
        initial_result.ocr_model = (
            self._ocr_config.model if self._ocr_config else ""
        )
        initial_result.ocr_pages = ocr_pages
        initial_result.ocr_time = time.time() - ocr_start_time
        initial_result.convert_time += time.time() - ocr_start_time
        initial_result.token_usage = total_token_usage

        # Record OCR usage via UsageTracker (extended schema)
        if self._usage_tracker is not None:
            try:
                self._usage_tracker.record_ocr(
                    model=initial_result.ocr_model,
                    input_tokens=total_token_usage.get("input_tokens", 0),
                    output_tokens=total_token_usage.get("output_tokens", 0),
                    total_tokens=total_token_usage.get("total_tokens", 0),
                )
            except Exception as e:
                logger.warning("UsageTracker record_ocr failed: %s", e)

        # Update output file
        output_file = output_dir / f"{source.stem}_ocr.md"
        output_file.write_text(combined_text, encoding="utf-8")
        initial_result.output_file = output_file

        # Update metadata
        initial_result.metadata["ocr_fallback"] = True
        initial_result.metadata["ocr_pages"] = ocr_pages

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict] = None,
    ) -> ConvertResult:
        """
        Convert a document file to markdown using the appropriate converter.

        This method:
        1. Selects the appropriate converter based on file extension
        2. Performs the conversion
        3. For PDFs, checks if the document is scanned and falls back to OCR

        Args:
            source: Path to the source document file
            output_dir: Directory to save output files
            options: Optional conversion options:
                - extract_images: bool - Extract images from PDF (default: False)
                - force_ocr: bool - Force OCR even for non-scanned PDFs (default: False)
                - disable_ocr_fallback: bool - Disable OCR fallback (default: False)

        Returns:
            ConvertResult containing conversion results

        Raises:
            UnsupportedFormatError: If file format is not supported
        """
        start_time = time.time()
        options = options or {}
        errors = []

        # Validate source file
        if not source.exists():
            return ConvertResult(
                success=False,
                markdown="",
                errors=[f"Source file not found: {source}"],
                source_file=source,
                converter_name="ConverterCoordinator",
                converter_version="0.1.0",
                convert_time=time.time() - start_time,
            )

        # Detect files with mismatched extension (e.g., HTML content in .pdf file)
        temp_source = None
        if source.suffix.lower() == ".pdf" and self._is_html_content(source):
            converter = self._html_converter
            # Create a temporary file with .html extension
            # since HTMLConverter validates file extension
            try:
                    temp_source = Path(tempfile.mktemp(suffix=".html"))
                    shutil.copy2(source, temp_source)
                    convert_source = temp_source
            except Exception as e:
                errors.append(f"Failed to create temporary file for HTML-in-PDF: {e}")
                return ConvertResult(
                    success=False,
                    markdown="",
                    errors=errors,
                    source_file=source,
                    converter_name="ConverterCoordinator",
                    converter_version="0.1.0",
                    convert_time=time.time() - start_time,
                )
        else:
            # Get appropriate converter
            try:
                converter = self.get_converter(source)
                convert_source = source
            except UnsupportedFormatError as e:
                return ConvertResult(
                    success=False,
                    markdown="",
                    errors=[str(e)],
                    source_file=source,
                    converter_name="ConverterCoordinator",
                    converter_version="0.1.0",
                    convert_time=time.time() - start_time,
                )

        # Perform conversion
        result = converter.convert(convert_source, output_dir, options)

        # Clean up temporary file if we created one
        if temp_source is not None:
            try:
                temp_source.unlink()
            except Exception:
                # Best effort cleanup - don't fail if cleanup fails
                pass

        # Check if this is a PDF that needs OCR fallback
        # Only consider it a PDF for OCR purposes if it's using PDFConverter
        # (not HTMLConverter, which we might have routed to for HTML-in-PDF files)
        is_pdf = source.suffix.lower() == ".pdf" and converter == self._pdf_converter
        force_ocr = options.get("force_ocr", False)
        disable_ocr_fallback = options.get("disable_ocr_fallback", False)

        if is_pdf and not disable_ocr_fallback:
            page_count = result.metadata.get("page_count", 1)

            # Check if PDF is scanned or OCR is forced
            if force_ocr or self._is_scanned_pdf(result, page_count):
                result = self._convert_scanned_pdf(
                    source=source,
                    output_dir=output_dir,
                    options=options,
                    initial_result=result,
                )

        # Update converter info in result
        result.metadata["coordinator_used"] = True
        result.metadata["coordinator_converter"] = converter.name

        # Extract tags from converted content (zero cost, keyword-based)
        if result.success and result.markdown.strip():
            try:
                tag_extractor = TagExtractor()
                tag_result = tag_extractor.extract(
                    markdown=result.markdown,
                    filename=str(source.name) if source else "",
                )
                result.metadata["tags"] = tag_result.tags
                result.metadata["doc_type"] = tag_result.doc_type
                result.metadata["keywords"] = tag_result.keywords
                result.metadata["tag_confidence"] = tag_result.confidence
            except Exception as e:
                logger.warning("TagExtractor failed for %s: %s", source, e)

            # Extract heading structure for Agent navigation (zero LLM, regex only)
            # NOTE: Must run BEFORE frontmatter injection so line numbers are
            # based on pure content (no YAML header offset).
            try:
                from .headings import extract_headings
                result.metadata["headings"] = extract_headings(result.markdown)
            except Exception as e:
                logger.warning("Headings extraction failed for %s: %s", source, e)

            # Inject YAML frontmatter into .md content (OKF-compatible)
            # headings are already extracted above with correct line numbers
            try:
                from .frontmatter import inject_frontmatter
                from datetime import datetime as _dt

                frontmatter_meta = {
                    "title": source.stem if source else "",
                    "doc_type": result.metadata.get("doc_type", "document"),
                    "source": str(source.name) if source else "",
                    "tags": result.metadata.get("tags", []),
                    "converted_at": _dt.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "headings": result.metadata.get("headings", []),
                }
                result.markdown = inject_frontmatter(
                    result.markdown, frontmatter_meta
                )
            except Exception as e:
                logger.warning("Frontmatter injection failed for %s: %s", source, e)

        # Set pipeline version on successful conversion
        if result.success:
            result.metadata["pipeline_version"] = PIPELINE_VERSION

        return result

    def estimate_time(self, source: Path) -> float:
        """
        Estimate conversion time for a file.

        Args:
            source: Path to the file to estimate time for

        Returns:
            Estimated time in seconds

        Raises:
            UnsupportedFormatError: If file format is not supported
        """
        converter = self.get_converter(source)
        return converter.estimate_time(source)

    def register_custom_converter(
        self,
        converter: Converter,
        override: bool = False,
    ) -> None:
        """
        Register a custom converter for its supported formats.

        Args:
            converter: Custom converter instance to register
            override: Whether to override existing converter for same extension

        Raises:
            ValueError: If extension is already registered and override is False
        """
        for ext in converter.supported_formats:
            ext_lower = ext.lower()
            if ext_lower in self._converters and not override:
                raise ValueError(
                    f"Extension {ext} is already registered. "
                    f"Use override=True to replace."
                )
            self._converters[ext_lower] = converter


# Module-level coordinator instance for convenience
_default_coordinator: Optional[ConverterCoordinator] = None


def get_coordinator(
    ocr_config: Optional[OCRServiceConfig] = None,
    usage_tracker: Optional["UsageTracker"] = None,
    **kwargs,
) -> ConverterCoordinator:
    """
    Get the default coordinator instance.

    Args:
        ocr_config: Optional OCR service configuration
        usage_tracker: Optional UsageTracker for recording OCR token usage
        **kwargs: Additional arguments passed to ConverterCoordinator

    Returns:
        ConverterCoordinator instance
    """
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = ConverterCoordinator(
            ocr_config=ocr_config, usage_tracker=usage_tracker, **kwargs
        )
    return _default_coordinator
