"""
Outlook MSG email to Markdown converter.

Uses olefile to parse .msg files (OLE2 Compound Binary Format).
Extracts: subject, sender, recipients, date, body (text/HTML), attachments.
"""

import re
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.converter.base import ConvertResult, Converter

# OLE stream paths for MSG properties (MAPI property tags)
_STREAM_SUBJECT = "__substg1.0_0037001F"  # PR_SUBJECT (Unicode)
_STREAM_SUBJECT_ASCII = "__substg1.0_0037001E"  # PR_SUBJECT (ASCII)
_STREAM_FROM_NAME = "__substg1.0_0C1A001F"  # PR_SENDER_NAME
_STREAM_FROM_EMAIL = "__substg1.0_0C1F001F"  # PR_SENDER_EMAIL_ADDRESS
_STREAM_TO = "__substg1.0_0E04001F"  # PR_DISPLAY_TO
_STREAM_CC = "__substg1.0_0E03001F"  # PR_DISPLAY_CC
_STREAM_DATE = "__substg1.0_00390040"  # PR_CLIENT_SUBMIT_TIME (FILETIME)
_STREAM_BODY = "__substg1.0_1000001F"  # PR_BODY (plain text, Unicode)
_STREAM_BODY_ASCII = "__substg1.0_1000001E"  # PR_BODY (plain text, ASCII)
_STREAM_HTML = "__substg1.0_10130102"  # PR_HTML (binary)
_STREAM_ATTACH_LONG_FILENAME = "__substg1.0_3707001F"  # PR_ATTACH_LONG_FILENAME
_STREAM_ATTACH_FILENAME = "__substg1.0_3704001F"  # PR_ATTACH_FILENAME

# FILETIME epoch: 1601-01-01 00:00:00 UTC
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
_FILETIME_TICKS_PER_SECOND = 10_000_000  # 100-nanosecond intervals


class MsgConverter(Converter):
    """
    Converter for Microsoft Outlook .msg email files.

    Uses olefile to parse the OLE2 Compound Binary Format.
    Extracts headers (subject, sender, recipients, date), body (plain/HTML),
    and attachment filenames.
    """

    @property
    def name(self) -> str:
        """Get the converter name."""
        return "MsgConverter"

    @property
    def version(self) -> str:
        """Get the converter version."""
        return "0.1.0"

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".msg"]

    def _read_stream(self, msg, stream_path: str) -> Optional[str]:
        """Read and decode an OLE stream.

        Tries UTF-16-LE first (Unicode streams with 001F suffix),
        falls back to UTF-8 (ASCII streams with 001E suffix).

        Args:
            msg: OleFileIO instance.
            stream_path: OLE stream path.

        Returns:
            Decoded string, or None if stream doesn't exist.
        """
        if not msg.exists(stream_path):
            return None
        try:
            data = msg.openstream(stream_path).read()
            if not data:
                return None
            # Try UTF-16-LE first (Unicode streams end with null bytes)
            try:
                text = data.decode("utf-16-le").rstrip("\x00")
                if text:
                    return text
            except (UnicodeDecodeError, UnicodeError):
                pass
            # Fallback to UTF-8
            text = data.decode("utf-8", errors="replace").rstrip("\x00")
            return text if text else None
        except Exception:
            return None

    def _read_binary_stream(self, msg, stream_path: str) -> Optional[bytes]:
        """Read a binary OLE stream.

        Args:
            msg: OleFileIO instance.
            stream_path: OLE stream path.

        Returns:
            Raw bytes, or None if stream doesn't exist.
        """
        if not msg.exists(stream_path):
            return None
        try:
            return msg.openstream(stream_path).read()
        except Exception:
            return None

    def _parse_filetime(self, data: bytes) -> Optional[str]:
        """Parse FILETIME bytes to ISO datetime string.

        FILETIME is 8 bytes, little-endian, representing 100-nanosecond
        intervals since 1601-01-01 00:00:00 UTC.

        Args:
            data: 8 bytes of FILETIME data.

        Returns:
            ISO format datetime string, or None if parsing fails.
        """
        if not data or len(data) < 8:
            return None
        try:
            ticks = struct.unpack("<Q", data[:8])[0]
            if ticks == 0:
                return None
            seconds = ticks / _FILETIME_TICKS_PER_SECOND
            dt = _FILETIME_EPOCH + timedelta(seconds=seconds)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (struct.error, OverflowError, OSError):
            return None

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags to extract plain text.

        Args:
            html: HTML content string.

        Returns:
            Plain text with HTML tags removed.
        """
        # Remove <style> and <script> blocks entirely
        text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Convert <br> and block elements to newlines
        text = re.sub(r"<br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|tr|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
        # Remove remaining tags
        text = re.sub(r"<[^>]+>", "", text)
        # Decode common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _get_attachments(self, msg) -> List[str]:
        """Enumerate attachment names from MSG file.

        Scans for __attach_version1.0_#000{NNN}/ directories and extracts
        filenames from the OLE stream properties.

        Args:
            msg: OleFileIO instance.

        Returns:
            List of attachment filenames.
        """
        attachments: List[str] = []
        if not msg.exists("__attach_version1.0_#00000000"):
            return attachments

        # Find all attachment directories
        attach_dirs: set[str] = set()
        for entry in msg.listdir():
            if entry and entry[0].startswith("__attach_version1.0_#"):
                attach_dirs.add(entry[0])

        for attach_dir in sorted(attach_dirs):
            # Try long filename first, then short filename
            long_name_stream = f"{attach_dir}/{_STREAM_ATTACH_LONG_FILENAME}"
            short_name_stream = f"{attach_dir}/{_STREAM_ATTACH_FILENAME}"

            filename = self._read_stream(msg, long_name_stream)
            if not filename:
                filename = self._read_stream(msg, short_name_stream)
            if filename:
                attachments.append(filename)
            else:
                attachments.append(f"<attachment_{attach_dir}>")

        return attachments

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict] = None,
    ) -> ConvertResult:
        """
        Convert an Outlook .msg email file to Markdown.

        Args:
            source: Path to the .msg file.
            output_dir: Directory to save output files.
            options: Optional conversion options.

        Returns:
            ConvertResult containing conversion results.
        """
        options = options or {}
        errors: List[str] = []
        metadata: Dict = {}
        start_time = time.time()

        # Validate file format
        if not self.can_convert(source):
            error_msg = (
                f"Unsupported file format: {source.suffix}. Expected .msg"
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
            # Lazy import olefile
            import olefile  # noqa: F811

            # Open MSG file
            try:
                msg = olefile.OleFileIO(str(source))
            except (OSError, IOError) as e:
                error_msg = f"Cannot open MSG file (invalid OLE format): {e}"
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

            try:
                # Extract email fields
                subject = self._read_stream(msg, _STREAM_SUBJECT)
                if not subject:
                    subject = self._read_stream(msg, _STREAM_SUBJECT_ASCII)
                if not subject:
                    subject = source.stem

                from_name = self._read_stream(msg, _STREAM_FROM_NAME)
                from_email = self._read_stream(msg, _STREAM_FROM_EMAIL)

                to_line = self._read_stream(msg, _STREAM_TO)
                cc_line = self._read_stream(msg, _STREAM_CC)

                # Parse date
                date_data = self._read_binary_stream(msg, _STREAM_DATE)
                date_str = self._parse_filetime(date_data) if date_data else None

                # Extract body — prefer plain text, fallback to HTML
                body_text = self._read_stream(msg, _STREAM_BODY)
                if not body_text:
                    body_text = self._read_stream(msg, _STREAM_BODY_ASCII)
                if not body_text:
                    html_data = self._read_binary_stream(msg, _STREAM_HTML)
                    if html_data:
                        try:
                            html_text = html_data.decode("utf-8", errors="replace")
                            body_text = self._strip_html(html_text)
                        except Exception:
                            body_text = None

                # Get attachments
                attachments = self._get_attachments(msg)

                # Build Markdown
                parts: List[str] = []

                # Title
                parts.append(f"# {subject}")
                parts.append("")

                # Email headers blockquote
                sender_display = f"{from_name} <{from_email}>" if from_name and from_email else (from_name or from_email or "未知")
                parts.append(f"> **发件人:** {sender_display}")

                if to_line:
                    parts.append(f"> **收件人:** {to_line}")
                if cc_line:
                    parts.append(f"> **抄送:** {cc_line}")
                if date_str:
                    parts.append(f"> **日期:** {date_str}")

                attach_count = len(attachments)
                parts.append(f"> **附件:** {attach_count} 个")

                # List attachment names if any
                if attachments:
                    parts.append(">")
                    parts.append("> **附件列表:**")
                    for att in attachments:
                        parts.append(f"> - {att}")

                parts.append("")
                parts.append("---")
                parts.append("")

                # Body content
                if body_text:
                    parts.append(body_text)
                else:
                    parts.append("*(邮件正文为空)*")

                markdown_content = "\n".join(parts)

                # Build metadata
                metadata["msg_subject"] = subject
                metadata["msg_sender"] = from_name or ""
                metadata["msg_sender_email"] = from_email or ""
                metadata["msg_date"] = date_str or ""
                metadata["attachment_count"] = attach_count
                if attachments:
                    metadata["attachment_names"] = attachments
                if to_line:
                    metadata["msg_to"] = to_line
                if cc_line:
                    metadata["msg_cc"] = cc_line

                # Write output file
                output_file = output_dir / f"{source.stem}.md"
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
                msg.close()

        except Exception as e:
            error_msg = f"Failed to convert MSG file: {e}"
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
