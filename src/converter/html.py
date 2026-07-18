"""
HTML document converter using MarkItDown.

Converts HTML files to Markdown format, preserving document structure
including headings, lists, links, and tables.
"""

from pathlib import Path

from markitdown import MarkItDown, MissingDependencyException, UnsupportedFormatException

from .base import Converter, ConvertResult
from .table_fix import fix_merged_tables_with_html, fix_table_alignment


class HTMLConverter(Converter):
    """
    Converter for HTML files using MarkItDown library.

    Supports .html and .htm file extensions.
    Converts HTML content to Markdown format while preserving structure.
    """

    @property
    def name(self) -> str:
        """Get the converter name."""
        return "HTMLConverter"

    @property
    def version(self) -> str:
        """Get the converter version."""
        return "0.1.0"

    @property
    def supported_formats(self) -> list[str]:
        """Get list of supported file extensions."""
        return [".html", ".htm"]

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: dict | None = None,
    ) -> ConvertResult:
        """
        Convert an HTML file to Markdown.

        Args:
            source: Path to the source HTML file.
            output_dir: Directory to save output files.
            options: Optional conversion options (currently unused).

        Returns:
            ConvertResult containing conversion results.
        """
        import time

        options = options or {}
        errors = []
        metadata = {}
        start_time = time.time()

        # Validate file extension
        if not self.can_convert(source):
            error_msg = (
                f"Unsupported file format: {source.suffix}. Expected .html or .htm"
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

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize MarkItDown converter
            md = MarkItDown()

            # Convert the HTML file
            result = md.convert(str(source))

            # Extract markdown content
            markdown_content = result.markdown

            # Apply table fixes (same as OfficeConverter)
            markdown_content = fix_table_alignment(markdown_content)
            markdown_content = fix_merged_tables_with_html(source, markdown_content)

            # Extract metadata
            if hasattr(result, "title") and result.title:
                metadata["title"] = result.title

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

        except MissingDependencyException as e:
            error_msg = f"Missing dependency for HTML conversion: {e}"
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

        except UnsupportedFormatException as e:
            error_msg = f"Unsupported HTML format: {e}"
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
            # Catch any other exceptions gracefully
            error_msg = f"Failed to convert HTML file: {e}"
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
