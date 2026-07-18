"""
Office document converter using MarkItDown.

Converts DOCX, DOC, PPTX, XLSX, and XLS files to Markdown format,
preserving document structure including headings, lists, and tables.

Legacy .doc format is bridged via LibreOffice:
    .doc → soffice --headless --convert-to docx → MarkItDown → .md
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from markitdown import MarkItDown, MissingDependencyException, UnsupportedFormatException

from .base import ConvertResult, Converter
from .table_fix import fix_merged_tables_with_html, fix_table_alignment

logger = logging.getLogger(__name__)


class OfficeConverter(Converter):
    """
    Converter for Office files using MarkItDown library.

    Supports .docx (Word), .doc (Word Legacy), .pptx (PowerPoint),
    .xlsx (Excel), and .xls (Excel Legacy) file extensions.
    Converts Office documents to Markdown format while preserving structure.

    Legacy .doc format is bridged via LibreOffice:
        .doc → soffice --headless --convert-to docx → MarkItDown → .md
    """

    # Mapping of file extensions to document types for metadata
    _EXTENSION_TYPES: Dict[str, str] = {
        ".docx": "Word Document",
        ".doc": "Word Document (Legacy)",
        ".pptx": "PowerPoint Presentation",
        ".xlsx": "Excel Spreadsheet",
        ".xls": "Excel Spreadsheet (Legacy)",
    }

    @property
    def name(self) -> str:
        """Get the converter name."""
        return "OfficeConverter"

    @property
    def version(self) -> str:
        """Get the converter version."""
        return "0.1.0"

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".docx", ".doc", ".pptx", ".xlsx", ".xls"]

    @staticmethod
    def _find_soffice() -> Optional[str]:
        """Find LibreOffice soffice executable (cross-platform).

        Search order:
            1. LIBREOFFICE_PATH environment variable
            2. PATH lookup via shutil.which
            3. Known installation directories (Windows / Linux / macOS)
        """
        # 1. Environment variable override
        env_path = os.getenv("LIBREOFFICE_PATH")
        if env_path and Path(env_path).is_file():
            return env_path

        # 2. System PATH
        found = shutil.which("soffice")
        if found:
            return found

        # 3. Known installation paths
        candidates = [
            # Windows
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            # Linux
            "/usr/bin/soffice",
            "/usr/bin/libreoffice",
            "/snap/bin/libreoffice",
            # macOS
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
        for p in candidates:
            if Path(p).is_file():
                return p

        return None

    @staticmethod
    def _convert_doc_to_docx(source: Path, work_dir: Path) -> Path:
        """Convert legacy .doc to .docx via LibreOffice.

        Args:
            source: Path to the .doc file.
            work_dir: Temporary working directory for the output .docx.

        Returns:
            Path to the converted .docx file.

        Raises:
            RuntimeError: If LibreOffice is not available or conversion fails.
        """
        soffice = OfficeConverter._find_soffice()
        if not soffice:
            raise RuntimeError(
                "LibreOffice (soffice) not found. "
                "Install libreoffice to convert legacy .doc files:\n"
                "  Ubuntu/Debian: sudo apt install libreoffice-writer-nogui\n"
                "  macOS:         brew install libreoffice\n"
                "  Windows:       winget install TheDocumentFoundation.LibreOffice\n"
                "  Or set LIBREOFFICE_PATH env var to soffice executable."
            )

        cmd = [
            soffice,
            "--headless",
            "--convert-to", "docx",
            "--outdir", str(work_dir),
            str(source),
        ]
        logger.info("Converting .doc → .docx: %s", source.name)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"LibreOffice timed out converting {source.name} "
                f"(file may be too large or corrupted)"
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"LibreOffice conversion failed for {source.name} "
                f"(exit code {result.returncode}): {stderr}"
            )

        # Find the output .docx (LibreOffice uses the stem name)
        docx_path = work_dir / f"{source.stem}.docx"
        if not docx_path.exists():
            # Some versions produce uppercase extension
            docx_path = work_dir / f"{source.stem}.DOCX"
        if not docx_path.exists():
            raise RuntimeError(
                f"LibreOffice ran but no .docx output found for {source.name}"
            )

        logger.info("Converted %s → %s (%d bytes)",
                     source.name, docx_path.name, docx_path.stat().st_size)
        return docx_path

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict] = None,
    ) -> ConvertResult:
        """
        Convert an Office document file to Markdown.

        Args:
            source: Path to the source Office document file.
            output_dir: Directory to save output files.
            options: Optional conversion options (currently unused).

        Returns:
            ConvertResult containing conversion results.
        """
        options = options or {}
        errors: List[str] = []
        metadata: Dict = {}
        start_time = time.time()

        # Validate file extension
        if not self.can_convert(source):
            supported = ", ".join(self.supported_formats)
            error_msg = (
                f"Unsupported file format: {source.suffix}. "
                f"Expected one of: {supported}"
            )
            return ConvertResult(
                success=False,
                markdown="",
                source_file=source,
                output_file=None,
                errors=[error_msg],
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        # Validate file exists
        if not source.exists():
            error_msg = f"Source file does not exist: {source}"
            return ConvertResult(
                success=False,
                markdown="",
                source_file=source,
                output_file=None,
                errors=[error_msg],
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        # Check if file is readable (not corrupted/locked)
        try:
            # Try to read a small portion to check if file is accessible
            with open(source, "rb") as f:
                f.read(8)  # Read header bytes to verify file integrity
        except IOError as e:
            error_msg = f"Cannot read source file (may be corrupted or locked): {e}"
            return ConvertResult(
                success=False,
                markdown="",
                source_file=source,
                output_file=None,
                errors=[error_msg],
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Bridge legacy .doc → .docx via LibreOffice
            tmpdir: Optional[tempfile.TemporaryDirectory] = None
            source_to_convert = source
            try:
                if source.suffix.lower() == ".doc":
                    tmpdir = tempfile.TemporaryDirectory(prefix="docsearch_doc_")
                    source_to_convert = self._convert_doc_to_docx(
                        source, Path(tmpdir.name)
                    )

                # Initialize MarkItDown converter
                md = MarkItDown()

                # Convert the Office document
                result = md.convert(str(source_to_convert))

                # Extract markdown content
                markdown_content = result.markdown

                # Fix table alignment issues from MarkItDown/markdownify
                markdown_content = fix_table_alignment(markdown_content)
                # Replace merged-cell tables with HTML for structural accuracy
                markdown_content = fix_merged_tables_with_html(source, markdown_content)

                # Extract metadata
                if hasattr(result, "title") and result.title:
                    metadata["title"] = result.title

                # Add document type to metadata
                extension = source.suffix.lower()
                if extension in self._EXTENSION_TYPES:
                    metadata["document_type"] = self._EXTENSION_TYPES[extension]

                # Add file size to metadata
                metadata["file_size_bytes"] = source.stat().st_size

                # Determine output file path
                output_file = output_dir / f"{source.stem}.md"

                # Write markdown to output file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)

                convert_time = time.time() - start_time

                return ConvertResult(
                    success=True,
                    markdown=markdown_content,
                    source_file=source,
                    output_file=output_file,
                    metadata=metadata,
                    errors=errors,
                    converter_name=self.name,
                    converter_version=self.version,
                    convert_time=convert_time,
                )
            finally:
                if tmpdir is not None:
                    tmpdir.cleanup()

        except UnsupportedFormatException as e:
            error_msg = f"Unsupported Office format: {e}"
            errors.append(error_msg)
            return ConvertResult(
                success=False,
                markdown="",
                source_file=source,
                output_file=None,
                errors=errors,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        except MissingDependencyException as e:
            error_msg = f"Unsupported Office format: {e}"
            errors.append(error_msg)
            return ConvertResult(
                success=False,
                markdown="",
                source_file=source,
                output_file=None,
                errors=errors,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        except Exception as e:
            # Catch any other exceptions gracefully (corrupted files, etc.)
            error_msg = f"Failed to convert Office file: {e}"
            errors.append(error_msg)
            return ConvertResult(
                success=False,
                markdown="",
                source_file=source,
                output_file=None,
                errors=errors,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )
