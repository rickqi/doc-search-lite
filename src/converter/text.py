"""
Text to Markdown passthrough converter.

Converts plain text files (.txt) to Markdown format with minimal
formatting - adds a title header and preserves original content.
Supports .md files as pass-through (direct copy).
Supports automatic encoding detection for Chinese text files.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

from src.converter.base import ConvertResult, Converter


class TextConverter(Converter):
    """
    Converter for plain text files.

    Supports .txt files. Reads text content and saves as .md
    with minimal formatting (add title header).
    """

    @property
    def name(self) -> str:
        """Get the converter name."""
        return "TextConverter"

    @property
    def version(self) -> str:
        """Get the converter version."""
        return "0.1.0"

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".txt", ".md"]

    def _read_text(self, source: Path) -> tuple:
        """读取文本文件，自动检测编码。

        按优先级尝试 utf-8 -> gbk -> latin-1

        Args:
            source: 文本文件路径

        Returns:
            (text_content, encoding_used) 元组
        """
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
        last_error: Optional[Exception] = None

        for enc in encodings:
            try:
                with open(source, "r", encoding=enc) as f:
                    content = f.read()
                return content, enc
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue

        raise ValueError(
            f"Failed to read text file with any encoding. Last error: {last_error}"
        )

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict] = None,
    ) -> ConvertResult:
        """
        Convert a plain text file to Markdown.

        Args:
            source: Path to the source text file.
            output_dir: Directory to save output files.
            options: Optional conversion options.
                - encoding: str - Force specific encoding (default: auto-detect)

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

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # .md files: pass-through — copy directly without modification
            if source.suffix.lower() == ".md":
                import shutil
                markdown_content = source.read_text(encoding="utf-8")
                if not markdown_content.strip():
                    error_msg = f"Empty markdown file: {source}"
                    return ConvertResult(
                        success=False, markdown="", source_file=source,
                        output_file=None, errors=[error_msg],
                        converter_name=self.name, converter_version=self.version,
                        convert_time=time.time() - start_time,
                    )
                output_file = output_dir / source.name
                shutil.copy2(source, output_file)
                metadata["file_size_bytes"] = source.stat().st_size
                metadata["line_count"] = markdown_content.count("\n") + 1
                metadata["char_count"] = len(markdown_content)
                metadata["passthrough"] = True
                convert_time = time.time() - start_time
                return ConvertResult(
                    success=True, markdown=markdown_content,
                    source_file=source, output_file=output_file,
                    metadata=metadata, errors=errors,
                    converter_name=self.name, converter_version=self.version,
                    convert_time=convert_time,
                )

            # .txt files: read and add title header
            # 读取文本内容
            force_encoding = options.get("encoding")
            if force_encoding:
                try:
                    with open(source, "r", encoding=force_encoding) as f:
                        text_content = f.read()
                    encoding_used = force_encoding
                except (UnicodeDecodeError, UnicodeError):
                    text_content, encoding_used = self._read_text(source)
            else:
                text_content, encoding_used = self._read_text(source)

            # 构建 Markdown 内容 - 添加标题头
            markdown_content = f"# {source.stem}\n\n{text_content}"

            # 添加元数据
            metadata["file_size_bytes"] = source.stat().st_size
            metadata["encoding"] = encoding_used
            metadata["line_count"] = text_content.count("\n") + 1
            metadata["char_count"] = len(text_content)

            # 确定输出文件路径
            output_file = output_dir / f"{source.stem}.md"

            # 写入 Markdown 文件
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

        except Exception as e:
            # 捕获所有其他异常
            error_msg = f"Failed to convert text file: {e}"
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
