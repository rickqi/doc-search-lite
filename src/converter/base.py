from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ConvertResult:
    """Result of a document conversion operation."""

    success: bool
    """Whether the conversion was successful."""

    markdown: str
    """Converted markdown content."""

    images: List[Path] = field(default_factory=list)
    """List of extracted image file paths."""

    metadata: Dict = field(default_factory=dict)
    """Additional metadata about the conversion."""

    errors: List[str] = field(default_factory=list)
    """List of error messages if conversion failed."""

    source_file: Optional[Path] = None
    """Source file path."""

    output_file: Optional[Path] = None
    """Output file path."""

    convert_time: float = 0.0
    """Time taken for conversion in seconds."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When the conversion was performed."""

    converter_name: str = ""
    """Name of the converter used."""

    converter_version: str = ""
    """Version of the converter used."""

    ocr_used: bool = False
    """Whether OCR was used during conversion."""

    ocr_model: str = ""
    """OCR model used if applicable."""

    ocr_pages: int = 0
    """Number of pages processed with OCR."""

    ocr_time: float = 0.0
    """Time taken for OCR processing in seconds."""

    token_usage: Dict = field(default_factory=dict)
    """Token usage from LLM/OCR API, e.g. {'input_tokens': 100, 'output_tokens': 500, 'total_tokens': 600}."""


class Converter(ABC):
    """Abstract base class for document converters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the converter name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Get the converter version."""

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Get list of supported file extensions (e.g., ['.pdf', '.docx'])."""

    @abstractmethod
    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict] = None,
    ) -> ConvertResult:
        """
        Convert a document file to markdown.

        Args:
            source: Path to the source document file.
            output_dir: Directory to save output files.
            options: Optional conversion options.

        Returns:
            ConvertResult containing conversion results.
        """

    def can_convert(self, file_path: Path) -> bool:
        """
        Check if this converter can handle the given file.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file extension is in supported_formats.
        """
        return file_path.suffix.lower() in [
            fmt.lower() for fmt in self.supported_formats
        ]

    def estimate_time(self, file_path: Path) -> float:
        """
        Estimate conversion time for a file.

        Args:
            file_path: Path to the file to estimate time for.

        Returns:
            Estimated time in seconds (defaults to size_mb * 2).
        """
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return size_mb * 2
        except (OSError, AttributeError):
            return 1.0  # Default to 1 second if can't get file size
