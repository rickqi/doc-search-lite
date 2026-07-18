"""
Image to Markdown converter using OCR service.

Converts image files (PNG, JPG, BMP, WEBP, GIF) to Markdown text
using GLM-OCR vision model for text extraction.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.converter.base import ConvertResult, Converter
from src.converter.ocr import (
    OCRService,
    OCRServiceConfig,
    get_paddleocr_service,
    get_paddleocr_http_service,
    get_ppstructurev3_service,
)
from src.converter.ocr_postprocess import postprocess_ocr_result

logger = logging.getLogger(__name__)


class ImageConverter(Converter):
    """
    Converter for image files using OCR (GLM-OCR).

    Supports .png, .jpg, .jpeg, .bmp, .webp, .gif files.
    Converts images to Markdown text using OCR service.
    """

    @property
    def name(self) -> str:
        """Get the converter name."""
        return "ImageConverter"

    @property
    def version(self) -> str:
        """Get the converter version."""
        return "0.1.0"

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"]

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict] = None,
    ) -> ConvertResult:
        """
        Convert an image file to Markdown using OCR.

        Args:
            source: Path to the source image file.
            output_dir: Directory to save output files.
            options: Optional conversion options.
                - ocr_api_key: str - API key for OCR service
                - ocr_base_url: str - Base URL for OCR API
                - ocr_model: str - Model name (default: "glm-ocr")
                - ocr_prompt: str - Custom prompt for OCR

        Returns:
            ConvertResult containing conversion results.
        """
        options = options or {}
        errors: List[str] = []
        metadata: Dict = {}
        start_time = time.time()

        # 验证文件格式
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

        # 验证文件存在
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

        # 检查文件是否可读
        try:
            with open(source, "rb") as f:
                f.read(8)
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

        # 检查 OCR API key (仅 zhipu 引擎需要)
        ocr_engine = options.get("ocr_engine", "zhipu")
        if ocr_engine == "zhipu":
            api_key = options.get("ocr_api_key", "")
            if not api_key:
                error_msg = (
                    "OCR API key not provided. "
                    "Pass 'ocr_api_key' in options to enable image conversion."
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

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            ocr_engine = options.get("ocr_engine", "zhipu")
            api_key = options.get("ocr_api_key", "")

            if ocr_engine == "paddleocr":
                ocr_service = get_paddleocr_service()
                ocr_model = "PP-OCRv5"
                ocr_result = ocr_service.recognize(source)
            elif ocr_engine == "paddleocr-http":
                ocr_service = get_paddleocr_http_service()
                ocr_result = ocr_service.recognize_vl(source)
                if ocr_result.success and ocr_result.text.strip():
                    ocr_model = "PaddleOCRVL-HTTP"
                else:
                    ocr_result = ocr_service.recognize_structure(source)
                    if ocr_result.success and ocr_result.text.strip():
                        ocr_model = "PP-StructureV3-HTTP"
                    else:
                        ocr_result = ocr_service.recognize(source)
                        ocr_model = "PP-OCRv5-HTTP"
            elif ocr_engine == "ppstructurev3":
                ocr_service = get_ppstructurev3_service()
                ocr_model = "PP-StructureV3"
            else:
                ocr_config = OCRServiceConfig(
                    api_key=api_key,
                    base_url=options.get("ocr_base_url"),
                    model=options.get("ocr_model", "glm-ocr"),
                    prompt=options.get("ocr_prompt", OCRServiceConfig.prompt),
                )
                ocr_service = OCRService(ocr_config)
                ocr_model = ocr_config.model
                ocr_result = ocr_service.recognize(source)

            if not ocr_result.success:
                error_msg = f"OCR recognition failed: {ocr_result.error}"
                return ConvertResult(
                    success=False,
                    markdown="",
                    source_file=source,
                    output_file=None,
                    errors=[error_msg],
                    converter_name=self.name,
                    converter_version=self.version,
                    convert_time=time.time() - start_time,
                    ocr_used=True,
                    ocr_model=ocr_model,
                    ocr_time=ocr_result.processing_time,
                )

            ocr_text = postprocess_ocr_result(ocr_result.text)
            markdown_content = f"# {source.stem}\n\n{ocr_text}"

            metadata["file_size_bytes"] = source.stat().st_size
            metadata["image_format"] = source.suffix.lower()
            metadata["ocr_model"] = ocr_model

            output_file = output_dir / f"{source.stem}.md"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            convert_time = time.time() - start_time

            token_usage = ocr_result.token_usage if ocr_result.token_usage else {}

            usage_tracker = options.get("usage_tracker")
            if usage_tracker is not None and token_usage:
                try:
                    usage_tracker.record_ocr(
                        model=ocr_model,
                        input_tokens=token_usage.get("input_tokens", 0),
                        output_tokens=token_usage.get("output_tokens", 0),
                        total_tokens=token_usage.get("total_tokens", 0),
                    )
                except Exception as e:
                    logger.warning("UsageTracker record_ocr failed: %s", e)

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
                ocr_used=True,
                ocr_model=ocr_model,
                ocr_time=ocr_result.processing_time,
                token_usage=token_usage,
            )

        except Exception as e:
            # 捕获所有其他异常
            error_msg = f"Failed to convert image file: {e}"
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
