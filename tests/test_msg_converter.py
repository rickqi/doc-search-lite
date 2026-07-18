"""Unit tests for MsgConverter — Outlook MSG email to Markdown converter."""

import struct
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.converter.msg import MsgConverter


@pytest.fixture
def converter():
    return MsgConverter()


@pytest.fixture
def msg_file(tmp_path):
    """Create a minimal .msg file path (content doesn't matter, olefile is mocked)."""
    return tmp_path / "test_email.msg"


def _encode_utf16le(text: str) -> bytes:
    """Encode text as UTF-16-LE with null terminator."""
    return text.encode("utf-16-le") + b"\x00\x00"


def _encode_filetime(year=2024, month=6, day=15, hour=10, minute=30, second=0):
    """Encode a datetime as FILETIME bytes (8 bytes LE)."""
    from datetime import datetime, timedelta, timezone

    epoch = datetime(1601, 1, 1, tzinfo=timezone.utc)
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    ticks = int((dt - epoch).total_seconds() * 10_000_000)
    return struct.pack("<Q", ticks)


def _make_mock_olefile(
    subject="Test Subject",
    from_name="Alice",
    from_email="alice@example.com",
    to_line="Bob <bob@example.com>",
    cc_line="Charlie",
    body="Hello,\n\nThis is the body.",
    date_bytes=None,
    attachments=None,
    html_body=None,
):
    """Create a mock OleFileIO with configurable fields.

    Returns a MagicMock that behaves like olefile.OleFileIO.
    """
    msg = MagicMock()

    # Build the set of attachment directories for exists/openstream
    _attach_dirs = []
    if attachments:
        for i in range(len(attachments)):
            _attach_dirs.append(f"__attach_version1.0_#0000000{i}")

    def exists(stream_path):
        stream_map = {
            "__substg1.0_0037001F": subject is not None,
            "__substg1.0_0C1A001F": from_name is not None,
            "__substg1.0_0C1F001F": from_email is not None,
            "__substg1.0_0E04001F": to_line is not None,
            "__substg1.0_0E03001F": cc_line is not None,
            "__substg1.0_00390040": date_bytes is not None,
            "__substg1.0_1000001F": body is not None,
            "__substg1.0_10130102": html_body is not None,
            "__attach_version1.0_#00000000": bool(attachments),
        }
        if stream_path in stream_map:
            return stream_map[stream_path]
        # Attachment stream paths
        for i, att in enumerate(attachments or []):
            dir_name = f"__attach_version1.0_#0000000{i}"
            if stream_path == f"{dir_name}/__substg1.0_3707001F":
                return True
            if stream_path == f"{dir_name}/__substg1.0_3704001F":
                return False  # fallback short name not available
        return False

    msg.exists = MagicMock(side_effect=exists)

    def openstream(stream_path):
        mock_stream = MagicMock()

        # Standard stream data
        data_map = {
            "__substg1.0_0037001F": _encode_utf16le(subject) if subject else b"",
            "__substg1.0_0C1A001F": _encode_utf16le(from_name) if from_name else b"",
            "__substg1.0_0C1F001F": _encode_utf16le(from_email) if from_email else b"",
            "__substg1.0_0E04001F": _encode_utf16le(to_line) if to_line else b"",
            "__substg1.0_0E03001F": _encode_utf16le(cc_line) if cc_line else b"",
            "__substg1.0_00390040": date_bytes if date_bytes else b"",
            "__substg1.0_1000001F": _encode_utf16le(body) if body else b"",
            "__substg1.0_10130102": html_body if html_body else b"",
        }

        # Check attachment streams first
        for i, att in enumerate(attachments or []):
            dir_name = f"__attach_version1.0_#0000000{i}"
            if stream_path == f"{dir_name}/__substg1.0_3707001F":
                mock_stream.read.return_value = _encode_utf16le(att)
                return mock_stream

        mock_stream.read.return_value = data_map.get(stream_path, b"")
        return mock_stream

    msg.openstream = MagicMock(side_effect=openstream)

    if attachments:
        listdir_entries = []
        for i in range(len(attachments)):
            dir_name = f"__attach_version1.0_#0000000{i}"
            listdir_entries.append([dir_name])
        msg.listdir = MagicMock(return_value=listdir_entries)
    else:
        msg.listdir = MagicMock(return_value=[])

    msg.close = MagicMock()
    return msg


def _encode_utf16lele(text):
    """Alias for consistency."""
    return _encode_utf16le(text)


class TestMsgConverterProperties:
    """Test MsgConverter basic properties."""

    def test_name(self, converter):
        assert converter.name == "MsgConverter"

    def test_version(self, converter):
        assert converter.version == "0.1.0"

    def test_supported_formats(self, converter):
        assert converter.supported_formats == [".msg"]

    def test_can_convert_msg(self, converter, tmp_path):
        f = tmp_path / "email.msg"
        f.write_text("dummy")
        assert converter.can_convert(f) is True

    def test_cannot_convert_non_msg(self, converter, tmp_path):
        f = tmp_path / "email.docx"
        f.write_text("dummy")
        assert converter.can_convert(f) is False

    def test_can_convert_case_insensitive(self, converter, tmp_path):
        f = tmp_path / "email.MSG"
        f.write_text("dummy")
        assert converter.can_convert(f) is True


class TestMsgConverterConvert:
    """Test convert() method."""

    def test_convert_full_email(self, converter, msg_file, tmp_path):
        mock_msg = _make_mock_olefile(
            subject="Quarterly Report",
            from_name="Alice",
            from_email="alice@example.com",
            to_line="Bob <bob@example.com>",
            cc_line="Charlie",
            body="Please find attached the quarterly report.",
            date_bytes=_encode_filetime(2024, 6, 15, 10, 30),
            attachments=["report.xlsx", "summary.pdf"],
        )
        msg_file.write_bytes(b"dummy msg content")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("olefile.OleFileIO", return_value=mock_msg):
            result = converter.convert(msg_file, output_dir)

        assert result.success is True
        assert "Quarterly Report" in result.markdown
        assert "Alice" in result.markdown
        assert "alice@example.com" in result.markdown
        assert "Bob" in result.markdown
        assert "Charlie" in result.markdown
        assert "2024-06-15" in result.markdown
        assert "quarterly report" in result.markdown.lower()
        assert "report.xlsx" in result.markdown
        assert "summary.pdf" in result.markdown
        assert result.output_file is not None
        assert result.output_file.exists()

    def test_convert_minimal_email(self, converter, msg_file, tmp_path):
        mock_msg = _make_mock_olefile(
            subject=None,
            from_name=None,
            from_email=None,
            to_line=None,
            cc_line=None,
            body="Simple body.",
            date_bytes=None,
            attachments=None,
        )
        msg_file.write_bytes(b"dummy")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("olefile.OleFileIO", return_value=mock_msg):
            result = converter.convert(msg_file, output_dir)

        assert result.success is True
        # Subject falls back to file stem
        assert "test_email" in result.markdown
        assert "Simple body." in result.markdown
        # Sender falls back to "未知"
        assert "未知" in result.markdown

    def test_convert_html_body_fallback(self, converter, msg_file, tmp_path):
        html = "<html><body><p>Hello from HTML</p></body></html>"
        mock_msg = _make_mock_olefile(
            subject="HTML Email",
            body=None,  # no plain text body
            html_body=html.encode("utf-8"),
        )
        msg_file.write_bytes(b"dummy")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("olefile.OleFileIO", return_value=mock_msg):
            result = converter.convert(msg_file, output_dir)

        assert result.success is True
        assert "Hello from HTML" in result.markdown

    def test_convert_empty_body(self, converter, msg_file, tmp_path):
        mock_msg = _make_mock_olefile(
            subject="Empty",
            body=None,
            html_body=None,
        )
        msg_file.write_bytes(b"dummy")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("olefile.OleFileIO", return_value=mock_msg):
            result = converter.convert(msg_file, output_dir)

        assert result.success is True
        assert "邮件正文为空" in result.markdown

    def test_convert_unsupported_format(self, converter, tmp_path):
        bad_file = tmp_path / "email.docx"
        bad_file.write_text("not an msg")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(bad_file, output_dir)

        assert result.success is False
        assert "Unsupported" in result.errors[0]

    def test_convert_nonexistent_file(self, converter, tmp_path):
        missing = tmp_path / "nonexistent.msg"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(missing, output_dir)

        assert result.success is False
        assert "does not exist" in result.errors[0]

    @patch("olefile.OleFileIO")
    def test_convert_invalid_ole(self, mock_olefile_cls, converter, msg_file, tmp_path):
        mock_olefile_cls.side_effect = OSError("Not a valid OLE file")
        msg_file.write_bytes(b"garbage")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(msg_file, output_dir)

        assert result.success is False
        assert "OLE" in result.errors[0] or "Cannot open" in result.errors[0]

    def test_convert_metadata(self, converter, msg_file, tmp_path):
        mock_msg = _make_mock_olefile(
            subject="Meta Test",
            from_name="Sender",
            from_email="sender@test.com",
            to_line="receiver@test.com",
            body="content",
            date_bytes=_encode_filetime(2024, 1, 1),
            attachments=["doc.pdf"],
        )
        msg_file.write_bytes(b"dummy")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("olefile.OleFileIO", return_value=mock_msg):
            result = converter.convert(msg_file, output_dir)

        assert result.success is True
        assert result.metadata["msg_subject"] == "Meta Test"
        assert result.metadata["msg_sender"] == "Sender"
        assert result.metadata["msg_sender_email"] == "sender@test.com"
        assert result.metadata["msg_to"] == "receiver@test.com"
        assert result.metadata["attachment_count"] == 1
        assert "doc.pdf" in result.metadata["attachment_names"]


class TestReadStream:
    """Test _read_stream internal method."""

    def test_read_existing_stream(self, converter):
        msg = MagicMock()
        msg.exists.return_value = True
        mock_stream = MagicMock()
        mock_stream.read.return_value = "Hello".encode("utf-16-le") + b"\x00\x00"
        msg.openstream.return_value = mock_stream

        result = converter._read_stream(msg, "__substg1.0_0037001F")
        assert result == "Hello"

    def test_read_missing_stream(self, converter):
        msg = MagicMock()
        msg.exists.return_value = False

        result = converter._read_stream(msg, "__substg1.0_0037001F")
        assert result is None

    def test_read_empty_stream(self, converter):
        msg = MagicMock()
        msg.exists.return_value = True
        mock_stream = MagicMock()
        mock_stream.read.return_value = b""
        msg.openstream.return_value = mock_stream

        result = converter._read_stream(msg, "__substg1.0_0037001F")
        assert result is None


class TestParseFiletime:
    """Test _parse_filetime internal method."""

    def test_valid_filetime(self, converter):
        data = _encode_filetime(2024, 6, 15, 10, 30, 0)
        result = converter._parse_filetime(data)
        assert result is not None
        assert "2024-06-15" in result
        assert "10:30:00" in result

    def test_zero_filetime(self, converter):
        data = struct.pack("<Q", 0)
        result = converter._parse_filetime(data)
        assert result is None

    def test_none_data(self, converter):
        result = converter._parse_filetime(None)
        assert result is None

    def test_short_data(self, converter):
        result = converter._parse_filetime(b"\x01\x02\x03")
        assert result is None

    def test_empty_data(self, converter):
        result = converter._parse_filetime(b"")
        assert result is None


class TestStripHtml:
    """Test _strip_html internal method."""

    def test_basic_tags(self, converter):
        html = "<p>Hello <b>world</b></p>"
        result = converter._strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_style_removal(self, converter):
        html = "<style>body{color:red}</style><p>Content</p>"
        result = converter._strip_html(html)
        assert "color" not in result
        assert "Content" in result

    def test_script_removal(self, converter):
        html = "<script>alert('x')</script><p>Content</p>"
        result = converter._strip_html(html)
        assert "alert" not in result
        assert "Content" in result

    def test_br_to_newline(self, converter):
        html = "Line1<br>Line2<br/>Line3"
        result = converter._strip_html(html)
        assert "Line1\nLine2\nLine3" in result

    def test_html_entities(self, converter):
        html = "&amp; &lt; &gt; &quot; &nbsp;"
        result = converter._strip_html(html)
        assert "&" in result
        assert "<" in result
        assert ">" in result
        assert '"' in result
        assert " " in result

    def test_empty_input(self, converter):
        result = converter._strip_html("")
        assert result == ""


class TestGetAttachments:
    """Test _get_attachments internal method."""

    def test_no_attachments(self, converter):
        msg = MagicMock()
        msg.exists.return_value = False
        result = converter._get_attachments(msg)
        assert result == []

    def test_with_attachments(self, converter):
        msg = MagicMock()
        msg.exists.return_value = True
        msg.listdir.return_value = [
            ["__attach_version1.0_#00000000"],
            ["__attach_version1.0_#00000001"],
        ]

        def openstream_side_effect(path):
            mock_stream = MagicMock()
            if "00000000/__substg1.0_3707001F" in path:
                mock_stream.read.return_value = _encode_utf16le("file1.pdf")
            elif "00000001/__substg1.0_3707001F" in path:
                mock_stream.read.return_value = _encode_utf16le("file2.xlsx")
            else:
                mock_stream.read.return_value = b""
            return mock_stream

        msg.openstream = MagicMock(side_effect=openstream_side_effect)

        result = converter._get_attachments(msg)
        assert len(result) == 2
        assert "file1.pdf" in result
        assert "file2.xlsx" in result
