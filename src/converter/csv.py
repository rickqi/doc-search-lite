"""
CSV to Markdown converter using pandas.

Converts CSV files to Markdown tables with automatic encoding detection,
supporting Chinese encodings (GBK, GB2312) and large file truncation.
"""

import time
from pathlib import Path

from src.converter.base import Converter, ConvertResult


class CSVConverter(Converter):
    """
    Converter for CSV files to Markdown tables.

    Supports .csv files. Converts each CSV to a Markdown table
    with automatic encoding detection and large file truncation.
    """

    # 最大行数限制，超过则截断
    _MAX_ROWS_DEFAULT = 10000

    @property
    def name(self) -> str:
        """Get the converter name."""
        return "CSVConverter"

    @property
    def version(self) -> str:
        """Get the converter version."""
        return "0.1.0"

    @property
    def supported_formats(self) -> list[str]:
        """Get list of supported file extensions."""
        return [".csv"]

    def _read_csv(self, source: Path, options: dict) -> tuple:
        """读取 CSV 文件，自动检测编码。

        Args:
            source: CSV 文件路径
            options: 转换选项

        Returns:
            (DataFrame, encoding_used) 元组

        Raises:
            Exception: 所有编码尝试失败时抛出
        """
        import pandas as pd

        encoding = options.get("encoding")
        delimiter = options.get("delimiter")
        max_rows = options.get("max_rows", 0)

        # 如果指定了编码，直接使用
        if encoding:
            read_kwargs: dict = {"encoding": encoding}
            if delimiter:
                read_kwargs["sep"] = delimiter
            if max_rows > 0:
                read_kwargs["nrows"] = max_rows
            df = pd.read_csv(str(source), **read_kwargs)
            return df, encoding

        # 自动检测编码 - 按优先级尝试
        encodings_to_try = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
        last_error: Exception | None = None

        for enc in encodings_to_try:
            try:
                read_kwargs = {"encoding": enc}
                if delimiter:
                    read_kwargs["sep"] = delimiter
                if max_rows > 0:
                    read_kwargs["nrows"] = max_rows
                df = pd.read_csv(str(source), **read_kwargs)
                return df, enc
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue

        # 所有编码失败，尝试 chardet 检测（可选依赖）
        try:
            import chardet

            with open(source, "rb") as f:
                raw = f.read(10000)
                detected = chardet.detect(raw)
                detected_encoding = detected.get("encoding", "utf-8")

            read_kwargs = {"encoding": detected_encoding}
            if delimiter:
                read_kwargs["sep"] = delimiter
            if max_rows > 0:
                read_kwargs["nrows"] = max_rows
            df = pd.read_csv(str(source), **read_kwargs)
            return df, detected_encoding
        except ImportError:
            pass
        except Exception as e:
            last_error = e

        raise ValueError(
            f"Failed to read CSV with any encoding. Last error: {last_error}"
        )

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: dict | None = None,
    ) -> ConvertResult:
        """
        Convert a CSV file to Markdown table.

        Args:
            source: Path to the source CSV file.
            output_dir: Directory to save output files.
            options: Optional conversion options.
                - encoding: str - CSV encoding (default: auto-detect)
                - delimiter: str - CSV delimiter (default: auto-detect)
                - max_rows: int - Maximum rows to convert (0 = all)

        Returns:
            ConvertResult containing conversion results.
        """
        options = options or {}
        errors: list[str] = []
        metadata: dict = {}
        start_time = time.time()

        # 验证文件格式
        if not self.can_convert(source):
            error_msg = (
                f"Unsupported file format: {source.suffix}. Expected .csv"
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
        except OSError as e:
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
            # 读取 CSV — 当 max_rows 为 0 时使用默认限制避免读取整个文件
            read_options = dict(options)
            if read_options.get("max_rows", 0) == 0:
                read_options["max_rows"] = self._MAX_ROWS_DEFAULT
            df, encoding_used = self._read_csv(source, read_options)
            total_rows = len(df)

            # 大文件截断
            truncated = False
            max_rows = options.get("max_rows", 0)
            if max_rows == 0 and total_rows > self._MAX_ROWS_DEFAULT:
                df = df.head(self._MAX_ROWS_DEFAULT)
                truncated = True

            # 转换为 Markdown 表格
            markdown_table = df.to_markdown(index=False, tablefmt="github")

            # 构建完整 Markdown 内容
            parts = [f"# {source.stem}\n\n"]

            if truncated:
                parts.append(
                    f"> 注意：CSV 文件共有 {total_rows} 行，"
                    f"已截断为前 {self._MAX_ROWS_DEFAULT} 行。\n\n"
                )

            parts.append(str(markdown_table))
            parts.append(f"\n\n<!-- 行数: {total_rows}, 列数: {len(df.columns)} -->")

            markdown_content = "".join(parts)

            # 添加元数据
            metadata["row_count"] = total_rows
            metadata["col_count"] = len(df.columns)
            metadata["columns"] = list(df.columns)
            metadata["encoding_used"] = encoding_used
            metadata["file_size_bytes"] = source.stat().st_size
            metadata["truncated"] = truncated

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
            error_msg = f"Failed to convert CSV file: {e}"
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
