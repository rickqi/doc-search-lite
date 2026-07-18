"""OCR service using GLM-4V vision model for text extraction from images.

Also includes local PaddleOCR and remote HTTP PaddleOCR services.
"""

import base64
import json as _json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.utils.config import Config

logger = logging.getLogger(__name__)


class OCRErrorCode(Enum):
    """Error codes for OCR operations."""

    INVALID_API_KEY = "invalid_api_key"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    INVALID_IMAGE = "invalid_image"
    FILE_NOT_FOUND = "file_not_found"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"


class OCRMode(Enum):
    """OCR processing mode."""

    LAYOUT_PARSING = "layout_parsing"  # Default, uses layout_parsing API
    VISION_CHAT = "vision_chat"  # Uses chat completions with vision model


class OCRError(Exception):
    """Exception raised for OCR errors."""

    def __init__(
        self,
        message: str,
        code: OCRErrorCode = OCRErrorCode.UNKNOWN,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.original_error = original_error


@dataclass
class OCRResult:
    """Result of an OCR operation."""

    success: bool
    """Whether the OCR operation was successful."""

    text: str = ""
    """Extracted text content."""

    confidence: float = 0.0
    """Confidence score (0.0 to 1.0)."""

    processing_time: float = 0.0
    """Time taken for OCR processing in seconds."""

    error: str | None = None
    """Error message if operation failed."""

    error_code: OCRErrorCode | None = None
    """Error code if operation failed."""

    pages: int = 1
    """Number of pages processed."""

    model: str = ""
    """Model used for OCR."""

    source: str = ""
    """Source image path or URL."""

    token_usage: dict[str, int] = field(default_factory=dict)
    """Token usage from API response, e.g. {'input_tokens': 100, 'output_tokens': 500, 'total_tokens': 600}."""


@dataclass
class OCRServiceConfig:
    """Configuration for OCR service."""

    api_key: str
    """GLM API key."""

    base_url: str | None = None
    """Optional base URL for GLM API."""

    model: str = "glm-ocr"
    """Model to use for OCR (default: glm-ocr)."""

    max_retries: int = 3
    """Maximum number of retry attempts."""

    retry_delay: float = 1.0
    """Initial retry delay in seconds."""

    retry_multiplier: float = 2.0
    """Multiplier for exponential backoff."""

    timeout: float = 60.0
    """Request timeout in seconds."""

    prompt: str = (
        "请识别并提取图片中的所有文字内容。只输出提取的文字，不要添加任何解释或说明。"
    )
    """Prompt for OCR extraction."""

    mode: "OCRMode" = None  # type: ignore[assignment]
    """OCR mode: LAYOUT_PARSING (default) or VISION_CHAT."""

    vision_model: str = "glm-5-turbo"
    """Model for vision chat mode."""

    vision_prompt: str = (
        "请识别并提取图片中的所有内容，按以下规则输出：\n"
        "1. 保持原文的结构和布局，包括标题、段落、列表\n"
        "2. 表格内容用 Markdown 表格格式输出，保持行列对应\n"
        "3. 如果存在合并单元格，用 HTML table 标签输出\n"
        "4. 保持原文的语义连贯性，不要打乱文字顺序\n"
        "5. 对于保险/金融类文档，确保金额、条款编号等关键信息完整准确"
    )
    """Prompt for vision chat mode OCR."""

    def __post_init__(self):
        """Set default mode if not provided."""
        if self.mode is None:
            self.mode = OCRMode.LAYOUT_PARSING

    @classmethod
    def from_config(cls, config: Config) -> "OCRServiceConfig":
        """Create OCRServiceConfig from application Config.

        Args:
            config: Application configuration

        Returns:
            OCRServiceConfig instance
        """
        return cls(
            api_key=config.glm_api_key,
            base_url=config.glm_base_url,
        )


class OCRService:
    """OCR service using GLM-4V vision model.

    This service provides OCR capabilities by using the GLM-4V multimodal
    model to extract text from images. It supports both local image files
    and remote image URLs.

    Example:
        >>> config = OCRServiceConfig(api_key="your-api-key")
        >>> service = OCRService(config)
        >>> result = service.recognize("path/to/image.png")
        >>> if result.success:
        ...     print(result.text)
    """

    def __init__(self, config: OCRServiceConfig):
        """Initialize OCR service.

        Args:
            config: OCR service configuration
        """
        self._config = config
        self._client = None

    def _get_client(self):
        """Get or create ZhipuAiClient client lazily.

        Returns:
            ZhipuAiClient client instance

        Raises:
            OCRError: If client initialization fails
        """
        if self._client is None:
            try:
                from zai import ZhipuAiClient

                self._client = ZhipuAiClient(api_key=self._config.api_key)
            except ImportError as e:
                raise OCRError(
                    "zai-sdk library not installed. Install with: pip install zai-sdk",
                    code=OCRErrorCode.UNKNOWN,
                    original_error=e,
                )
        return self._client

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL.

        Args:
            source: Source string to check

        Returns:
            True if source is a URL
        """
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _encode_image(self, image_path: Path) -> str:
        """Encode image file to base64.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string

        Raises:
            OCRError: If file cannot be read
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError as e:
            raise OCRError(
                f"Image file not found: {image_path}",
                code=OCRErrorCode.FILE_NOT_FOUND,
                original_error=e,
            )
        except Exception as e:
            raise OCRError(
                f"Failed to read image file: {e}",
                code=OCRErrorCode.INVALID_IMAGE,
                original_error=e,
            )

    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type from file extension.

        Args:
            path: File path

        Returns:
            MIME type string
        """
        suffix = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }
        return mime_map.get(suffix, "image/jpeg")

    def _classify_error(self, error: Exception) -> OCRErrorCode:
        """Classify error into OCRErrorCode.

        Args:
            error: Original exception

        Returns:
            Appropriate OCRErrorCode
        """
        error_str = str(error).lower()

        if (
            "api_key" in error_str
            or "authentication" in error_str
            or "unauthorized" in error_str
        ):
            return OCRErrorCode.INVALID_API_KEY

        if "rate" in error_str or "limit" in error_str or "quota" in error_str:
            return OCRErrorCode.RATE_LIMITED

        if (
            "network" in error_str
            or "connection" in error_str
            or "timeout" in error_str
        ):
            return OCRErrorCode.NETWORK_ERROR

        if "image" in error_str or "format" in error_str or "invalid" in error_str:
            return OCRErrorCode.INVALID_IMAGE

        return OCRErrorCode.API_ERROR

    def _call_api_with_retry(self, file_input: str | Path) -> tuple:
        """Call API with retry logic, dispatching based on OCR mode.

        Args:
            file_input: Image file path or URL

        Returns:
            Tuple of (extracted_text, token_usage_dict).
            token_usage_dict may be empty {} if API doesn't return usage.

        Raises:
            OCRError: If all retries fail
        """
        if self._config.mode == OCRMode.VISION_CHAT:
            return self._call_vision_api(file_input)

        return self._call_layout_parsing_api(file_input)

    def _prepare_file_param(self, file_input: str | Path) -> str:
        """Convert file input to a data URL or pass through URL.

        Args:
            file_input: Image file path or URL string.

        Returns:
            Data URL string or original URL.

        Raises:
            OCRError: If local file not found.
        """
        if isinstance(file_input, Path):
            mime_type = self._get_mime_type(file_input)
            base64_image = self._encode_image(file_input)
            return f"data:{mime_type};base64,{base64_image}"

        if not self._is_url(str(file_input)):
            path = Path(str(file_input))
            if not path.exists():
                raise OCRError(
                    f"Image file not found: {file_input}",
                    code=OCRErrorCode.FILE_NOT_FOUND,
                )
            mime_type = self._get_mime_type(path)
            base64_image = self._encode_image(path)
            return f"data:{mime_type};base64,{base64_image}"

        return str(file_input)

    def _call_layout_parsing_api(self, file_input: str | Path) -> tuple:
        """Call layout_parsing API with retry logic.

        Args:
            file_input: Image file path or URL.

        Returns:
            Tuple of (extracted_text, token_usage_dict).

        Raises:
            OCRError: If all retries fail.
        """
        client = self._get_client()
        last_error: Exception | None = None

        file_param = self._prepare_file_param(file_input)

        for attempt in range(self._config.max_retries):
            try:
                response = client.layout_parsing.create(
                    model=self._config.model,
                    file=file_param,
                )

                # Extract token usage from response
                token_usage: dict[str, int] = {}
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
                    outp = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
                    total = getattr(usage, "total_tokens", None) or (inp + outp)
                    token_usage = {
                        "input_tokens": inp,
                        "output_tokens": outp,
                        "total_tokens": total,
                    }

                # Handle different response formats from zai-sdk
                if hasattr(response, "md_results"):
                    text = response.md_results or ""
                elif hasattr(response, "text"):
                    text = response.text
                elif isinstance(response, str):
                    text = response
                else:
                    text = str(response)

                return text, token_usage

            except Exception as e:
                last_error = e
                error_code = self._classify_error(e)

                # Don't retry for certain error types
                if error_code in (
                    OCRErrorCode.INVALID_API_KEY,
                    OCRErrorCode.INVALID_IMAGE,
                ):
                    raise OCRError(
                        str(e),
                        code=error_code,
                        original_error=e,
                    )

                # Calculate retry delay with exponential backoff
                if attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (
                        self._config.retry_multiplier**attempt
                    )
                    time.sleep(delay)

        # All retries failed
        raise OCRError(
            f"OCR failed after {self._config.max_retries} attempts: {last_error}",
            code=self._classify_error(last_error)
            if last_error
            else OCRErrorCode.UNKNOWN,
            original_error=last_error,
        )

    def _call_vision_api(self, file_input: str | Path) -> tuple:
        """Call chat completions API with vision model for OCR.

        Uses litellm to call a vision-capable model (e.g. glm-5-turbo)
        with the image and a prompt that instructs structured extraction.

        Args:
            file_input: Image file path or URL.

        Returns:
            Tuple of (extracted_text, token_usage_dict).

        Raises:
            OCRError: If all retries fail.
        """
        import litellm

        last_error: Exception | None = None

        # Build image content for litellm
        if isinstance(file_input, Path) or not self._is_url(str(file_input)):
            path = Path(str(file_input)) if not isinstance(file_input, Path) else file_input
            if not path.exists():
                raise OCRError(
                    f"Image file not found: {file_input}",
                    code=OCRErrorCode.FILE_NOT_FOUND,
                )
            mime_type = self._get_mime_type(path)
            b64 = self._encode_image(path)
            image_url = f"data:{mime_type};base64,{b64}"
        else:
            image_url = str(file_input)

        model = f"zai/{self._config.vision_model}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._config.vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ]

        for attempt in range(self._config.max_retries):
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    timeout=self._config.timeout,
                    api_key=self._config.api_key,
                    api_base=self._config.base_url,
                )

                # Extract text from response
                text = ""
                if response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, "message") and choice.message:
                        text = choice.message.content or ""

                # Extract token usage
                token_usage: dict[str, int] = {}
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    token_usage = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }

                return text, token_usage

            except Exception as e:
                last_error = e
                error_code = self._classify_error(e)

                if error_code in (
                    OCRErrorCode.INVALID_API_KEY,
                    OCRErrorCode.INVALID_IMAGE,
                ):
                    raise OCRError(
                        str(e),
                        code=error_code,
                        original_error=e,
                    )

                if attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (
                        self._config.retry_multiplier**attempt
                    )
                    time.sleep(delay)

        raise OCRError(
            f"Vision OCR failed after {self._config.max_retries} attempts: {last_error}",
            code=self._classify_error(last_error)
            if last_error
            else OCRErrorCode.UNKNOWN,
            original_error=last_error,
        )

    def recognize(self, source: str | Path) -> OCRResult:
        """Recognize text from an image.

        Args:
            source: Image source - either a file path or URL

        Returns:
            OCRResult with extracted text or error information

        Example:
            >>> result = service.recognize("document.png")
            >>> result = service.recognize("https://example.com/image.jpg")
        """
        start_time = time.time()
        source_str = str(source)

        try:
            # Call API - file handling is now done inside _call_api_with_retry
            text, token_usage = self._call_api_with_retry(source)

            return OCRResult(
                success=True,
                text=text,
                confidence=0.9,  # GLM-OCR doesn't provide confidence, use default
                processing_time=time.time() - start_time,
                model=self._config.model,
                source=source_str,
                token_usage=token_usage,
            )

        except OCRError as e:
            return OCRResult(
                success=False,
                error=str(e),
                error_code=e.code,
                processing_time=time.time() - start_time,
                source=source_str,
            )
        except Exception as e:
            return OCRResult(
                success=False,
                error=f"Unexpected error: {e}",
                error_code=OCRErrorCode.UNKNOWN,
                processing_time=time.time() - start_time,
                source=source_str,
            )

    def recognize_batch(
        self,
        sources: list[str | Path],
    ) -> list[OCRResult]:
        """Recognize text from multiple images.

        Args:
            sources: List of image sources (file paths or URLs)

        Returns:
            List of OCRResult for each source
        """
        return [self.recognize(source) for source in sources]

    def recognize_pdf(self, pdf_path: Path, max_pages_per_chunk: int = 30) -> OCRResult:
        """Recognize text from a PDF by sending it directly to the layout_parsing API.

        Unlike recognize() which handles individual rendered page images, this method
        sends the entire PDF file to the API in a single call (for small PDFs) or
        splits it into page-range chunks (for large PDFs). This is dramatically
        faster for large scanned documents (100+ pages) because:
        - The API processes document structure internally (no per-page renders)
        - No pdfplumber rendering to PNG (saves ~12MB/page I/O)
        - ~30 pages per API call vs 1 page per call in streaming mode

        For PDFs larger than ~30 MB, the file is split into page-range chunks
        using PyPDF to avoid base64 encoding issues with oversized payloads.

        Args:
            pdf_path: Path to the PDF file.
            max_pages_per_chunk: Max pages per API call when chunking large PDFs.

        Returns:
            OCRResult with extracted text from all pages.
        """
        start_time = time.time()
        source_str = str(pdf_path)

        # ── Threshold: send whole PDF directly if small enough ──────
        DIRECT_SUBMIT_MAX_BYTES = 30 * 1024 * 1024  # 30 MB

        if pdf_path.stat().st_size < DIRECT_SUBMIT_MAX_BYTES:
            try:
                text, token_usage = self._call_layout_parsing_api(pdf_path)
                return OCRResult(
                    success=True,
                    text=text,
                    confidence=0.9,
                    processing_time=time.time() - start_time,
                    model=self._config.model,
                    source=source_str,
                    token_usage=token_usage,
                )
            except OCRError as e:
                return OCRResult(
                    success=False,
                    error=str(e),
                    error_code=e.code,
                    processing_time=time.time() - start_time,
                    source=source_str,
                )
            except Exception as e:
                return OCRResult(
                    success=False,
                    error=f"PDF OCR failed: {e}",
                    error_code=OCRErrorCode.UNKNOWN,
                    processing_time=time.time() - start_time,
                    source=source_str,
                )

        # ── Large PDF: split into page-range chunks ─────────────────
        # Sending a 176 MB PDF as a single base64 data URI (~235 MB)
        # causes HTTP connection errors.  Instead, split the PDF into
        # smaller page-range PDFs (~3-9 MB each) and process separately.
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            return OCRResult(
                success=False,
                error="pypdf library required for large PDF chunked OCR. "
                      "Install with: pip install pypdf",
                error_code=OCRErrorCode.UNKNOWN,
                processing_time=time.time() - start_time,
                source=source_str,
            )

        import shutil
        import tempfile

        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        logger.info(
            "Splitting large PDF %s (%d MB, %d pages) into %d-page chunks",
            pdf_path.name,
            pdf_path.stat().st_size // (1024 * 1024),
            total_pages,
            max_pages_per_chunk,
        )

        temp_dir = Path(tempfile.mkdtemp())
        all_texts: list[str] = []
        total_token_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        chunk_count = 0

        try:
            for chunk_start in range(0, total_pages, max_pages_per_chunk):
                chunk_end = min(chunk_start + max_pages_per_chunk, total_pages)
                chunk_path = temp_dir / f"chunk_{chunk_start:04d}-{chunk_end:04d}.pdf"

                # Write chunk PDF
                writer = PdfWriter()
                for page_num in range(chunk_start, chunk_end):
                    writer.add_page(reader.pages[page_num])
                with open(chunk_path, "wb") as f:
                    writer.write(f)

                # Send chunk to layout_parsing API
                try:
                    text, token_usage = self._call_layout_parsing_api(chunk_path)
                    if text and text.strip():
                        all_texts.append(text)
                    for k in total_token_usage:
                        total_token_usage[k] += token_usage.get(k, 0)
                    chunk_count += 1
                    logger.debug(
                        "Chunk %d/%d (%d-%d) OCR complete",
                        chunk_count,
                        (total_pages + max_pages_per_chunk - 1) // max_pages_per_chunk,
                        chunk_start + 1,
                        chunk_end,
                    )
                except Exception as e:
                    logger.warning(
                        "Chunk %d-%d OCR failed: %s", chunk_start + 1, chunk_end, e,
                    )

                # Cleanup chunk PDF immediately
                try:
                    chunk_path.unlink()
                except Exception:
                    pass

        except Exception as e:
            logger.error("PDF chunking failed: %s", e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if not all_texts:
            return OCRResult(
                success=False,
                error="PDF chunked OCR produced no text from any chunk",
                error_code=OCRErrorCode.UNKNOWN,
                processing_time=time.time() - start_time,
                source=source_str,
            )

        combined_text = "\n\n".join(all_texts)
        logger.info(
            "PDF chunked OCR complete: %d chunks, %d/%d pages, %.1fs",
            chunk_count, total_pages, total_pages,
            time.time() - start_time,
        )

        return OCRResult(
            success=True,
            text=combined_text,
            confidence=0.9,
            processing_time=time.time() - start_time,
            model=self._config.model,
            source=source_str,
            token_usage=total_token_usage,
        )


class PaddleOCRService:
    """OCR service using local PaddleOCR (PP-OCRv5) engine.

    Provides the same recognize() interface as OCRService, enabling
    drop-in replacement for offline OCR without API calls.
    Uses module-level singleton to avoid repeated model loading.

    Example:
        >>> result = get_paddleocr_service().recognize("image.png")
        >>> print(result.text)
    """

    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            import os
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr

    def recognize(self, source: str | Path) -> OCRResult:
        start_time = time.time()
        source_str = str(source)

        try:
            ocr = self._get_ocr()
            result = ocr.predict(source_str)

            res = result[0]
            data = res.json if hasattr(res, "json") else res
            rec_texts = data.get("res", {}).get("rec_texts", [])
            rec_scores = data.get("res", {}).get("rec_scores", [])

            filtered = [t for t, s in zip(rec_texts, rec_scores) if s >= 0.3]
            text = "\n".join(filtered)

            avg_conf = (
                sum(float(s) for s in rec_scores) / len(rec_scores)
                if rec_scores
                else 0.0
            )

            return OCRResult(
                success=True,
                text=text,
                confidence=avg_conf,
                processing_time=time.time() - start_time,
                model="PP-OCRv5",
                source=source_str,
            )
        except Exception as e:
            return OCRResult(
                success=False,
                error=str(e),
                error_code=OCRErrorCode.UNKNOWN,
                processing_time=time.time() - start_time,
                source=source_str,
            )

    def recognize_batch(
        self,
        sources: list[str | Path],
    ) -> list[OCRResult]:
        return [self.recognize(source) for source in sources]


_paddleocr_service: PaddleOCRService | None = None


def get_paddleocr_service() -> PaddleOCRService:
    """Get or create the module-level PaddleOCRService singleton."""
    global _paddleocr_service
    if _paddleocr_service is None:
        _paddleocr_service = PaddleOCRService()
    return _paddleocr_service


class PaddleOCRHTTPService:
    """OCR via remote PaddleOCR HTTP API (e.g., WSL2 GPU server).

    Connects to an external PaddleOCR FastAPI server that exposes
    POST /ocr for image recognition and GET /health for status.
    This is the recommended approach for Windows hosts where
    PaddlePaddle GPU is only available in WSL2.

    The server must implement the API contract:
        GET  /health → {"status": "ok", "gpu": true}
        POST /ocr (multipart/form-data, field: "file")
            → {"texts": [{"text": "...", "confidence": 0.99}],
               "count": N, "time_ms": 268}

    Configuration:
        PADDLEOCR_HTTP_URL — base URL of the OCR server (default: http://localhost:8868)
    """

    DEFAULT_URL = "http://localhost:8868"

    def __init__(self, base_url: str | None = None):
        self._base_url = (base_url or os.environ.get("PADDLEOCR_HTTP_URL", self.DEFAULT_URL)).rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base_url

    def health(self) -> bool:
        """Check if the remote OCR server is reachable and healthy.

        Returns:
            True if server responds with status=ok.
        """
        try:
            req = Request(f"{self._base_url}/health")
            with urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
                return data.get("status") == "ok"
        except Exception:
            return False

    def recognize(self, source: str | Path) -> OCRResult:
        """Recognize text from an image via remote PaddleOCR /ocr endpoint.

        Uses POST /ocr for fast plain-text extraction (no layout/table/formula).
        For structured output, use recognize_structure().

        Args:
            source: Path to local image file.

        Returns:
            OCRResult with extracted text or error information.
        """
        start_time = time.time()
        source_str = str(source)

        try:
            path = Path(source_str)
            if not path.exists():
                return OCRResult(
                    success=False,
                    error=f"Image file not found: {source_str}",
                    error_code=OCRErrorCode.FILE_NOT_FOUND,
                    processing_time=time.time() - start_time,
                    source=source_str,
                )

            content_type, body = self._encode_multipart(path)

            req = Request(
                f"{self._base_url}/ocr",
                data=body,
                headers={"Content-Type": content_type},
            )

            with urlopen(req, timeout=600) as resp:
                data = _json.loads(resp.read().decode())

            texts = data.get("texts", [])
            text_lines = [item["text"] for item in texts if item.get("text")]
            confidences = [item.get("confidence", 0.0) for item in texts]

            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            elapsed = data.get("time_ms", 0) / 1000.0

            return OCRResult(
                success=True,
                text="\n".join(text_lines),
                confidence=round(avg_confidence, 4),
                processing_time=elapsed or (time.time() - start_time),
                model="PP-OCRv5-HTTP",
                source=source_str,
            )

        except Exception as e:
            return OCRResult(
                success=False,
                error=f"PaddleOCR HTTP error: {e}",
                error_code=OCRErrorCode.NETWORK_ERROR,
                processing_time=time.time() - start_time,
                source=source_str,
            )

    def recognize_structure(
        self,
        source: str | Path,
        use_formula: bool = True,
        use_seal: bool = True,
        use_chart: bool = False,  # chart adds 3-10x latency, only enable for chart-heavy docs
    ) -> OCRResult:
        """Recognize text via PP-StructureV3 /structure/download?format=md.

        Returns clean markdown with layout preservation, table detection,
        formula recognition, and seal detection.

        Args:
            source: Path to local image file.
            use_formula: Enable formula recognition pipeline.
            use_seal: Enable seal/stamp detection pipeline.
            use_chart: Enable chart/figure analysis pipeline.

        Returns:
            OCRResult with structured markdown text.
        """
        start_time = time.time()
        source_str = str(source)

        try:
            path = Path(source_str)
            if not path.exists():
                return OCRResult(
                    success=False,
                    error=f"Image file not found: {source_str}",
                    error_code=OCRErrorCode.FILE_NOT_FOUND,
                    processing_time=time.time() - start_time,
                    source=source_str,
                )

            content_type, body = self._encode_multipart(path)

            # Build query string for optional pipelines
            params = ["format=md"]
            if use_formula:
                params.append("use_formula=true")
            if use_seal:
                params.append("use_seal=true")
            if use_chart:
                params.append("use_chart=true")
            qs = "&".join(params)
            url = f"{self._base_url}/structure/download?{qs}"

            req = Request(url, data=body, headers={"Content-Type": content_type})

            with urlopen(req, timeout=600) as resp:
                text = resp.read().decode("utf-8")

            elapsed = time.time() - start_time

            return OCRResult(
                success=True,
                text=text,
                confidence=0.9,
                processing_time=elapsed,
                model="PP-StructureV3-HTTP",
                source=source_str,
            )

        except Exception as e:
            return OCRResult(
                success=False,
                error=f"PaddleOCR Structure HTTP error: {e}",
                error_code=OCRErrorCode.NETWORK_ERROR,
                processing_time=time.time() - start_time,
                source=source_str,
            )

    def recognize_vl(self, source: str | Path) -> OCRResult:
        """Recognize text via PaddleOCRVL /vl endpoint (Vision-Language Model).

        Uses POST /vl for VLM-based document parsing — best quality
        for complex documents with mixed content (text, tables, images).

        Args:
            source: Path to local image file.

        Returns:
            OCRResult with VLM-generated markdown text.
        """
        start_time = time.time()
        source_str = str(source)

        try:
            path = Path(source_str)
            if not path.exists():
                return OCRResult(
                    success=False,
                    error=f"Image file not found: {source_str}",
                    error_code=OCRErrorCode.FILE_NOT_FOUND,
                    processing_time=time.time() - start_time,
                    source=source_str,
                )

            content_type, body = self._encode_multipart(path)

            req = Request(
                f"{self._base_url}/vl",
                data=body,
                headers={"Content-Type": content_type},
            )

            with urlopen(req, timeout=600) as resp:
                data = _json.loads(resp.read().decode())

            pages = data.get("pages", [])
            text = "\n".join(p.get("markdown", "") for p in pages)

            return OCRResult(
                success=True,
                text=text,
                confidence=0.9,
                processing_time=time.time() - start_time,
                model="PaddleOCRVL-HTTP",
                source=source_str,
            )

        except Exception as e:
            return OCRResult(
                success=False,
                error=f"PaddleOCR VL HTTP error: {e}",
                error_code=OCRErrorCode.NETWORK_ERROR,
                processing_time=time.time() - start_time,
                source=source_str,
            )

    def recognize_batch(
        self,
        sources: list[str | Path],
    ) -> list[OCRResult]:
        """Recognize text from multiple images (sequential)."""
        return [self.recognize(source) for source in sources]

    @staticmethod
    def _encode_multipart(file_path: Path) -> tuple[str, bytes]:
        """Build multipart/form-data body for file upload.

        Returns:
            Tuple of (content_type header, body bytes).
        """
        boundary = f"----PaddleOCRHttp{uuid.uuid4().hex[:12]}"
        filename = file_path.name

        body_parts = [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
            b"Content-Type: application/octet-stream",
            b"",
            file_path.read_bytes(),
            f"--{boundary}--".encode(),
            b"",
        ]

        body = b"\r\n".join(body_parts)
        content_type = f"multipart/form-data; boundary={boundary}"

        return content_type, body


_paddleocr_http_service: PaddleOCRHTTPService | None = None


def get_paddleocr_http_service() -> PaddleOCRHTTPService:
    """Get or create the module-level PaddleOCRHTTPService singleton."""
    global _paddleocr_http_service
    if _paddleocr_http_service is None:
        _paddleocr_http_service = PaddleOCRHTTPService()
    return _paddleocr_http_service


class PPStructureV3Service:
    """Structured document parsing via PaddleOCR PP-StructureV3.

    Detects layout, tables, formulas, seals, and charts, outputting
    formatted Markdown with proper table structures.

    Requires >4GB VRAM for multi-model pipeline.
    """

    def __init__(self):
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PPStructureV3

            self._pipeline = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_table_recognition=True,
                use_formula_recognition=True,
                use_seal_recognition=False,
                use_chart_recognition=False,
                format_block_content=True,
            )
        return self._pipeline

    def recognize(self, source: str | Path) -> OCRResult:
        start_time = time.time()
        source_str = str(source)

        try:
            import shutil
            import tempfile
            from pathlib import Path as _Path

            pipeline = self._get_pipeline()
            output = pipeline.predict(source_str)
            res = output[0]

            tmpdir = tempfile.mkdtemp()
            try:
                res.save_to_markdown(tmpdir)
                md_files = list(_Path(tmpdir).glob("*.md"))
                text = md_files[0].read_text(encoding="utf-8") if md_files else ""
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            return OCRResult(
                success=True,
                text=text,
                confidence=0.9,
                processing_time=time.time() - start_time,
                model="PP-StructureV3",
                source=source_str,
            )
        except Exception as e:
            return OCRResult(
                success=False,
                error=str(e),
                error_code=OCRErrorCode.UNKNOWN,
                processing_time=time.time() - start_time,
                source=source_str,
            )

    def recognize_batch(self, sources):
        return [self.recognize(s) for s in sources]


_ppstructurev3_service: PPStructureV3Service | None = None


def get_ppstructurev3_service() -> PPStructureV3Service:
    global _ppstructurev3_service
    if _ppstructurev3_service is None:
        _ppstructurev3_service = PPStructureV3Service()
    return _ppstructurev3_service
