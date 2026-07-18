"""Tests for hash utility functions — calculate_hash and calculate_content_hash."""

import hashlib
from pathlib import Path

import pytest

from src.utils.hash import calculate_content_hash, calculate_hash


class TestCalculateHash:
    """File hashing via calculate_hash()."""

    def test_file_hash_consistent(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        h1 = calculate_hash(f)
        h2 = calculate_hash(f)
        assert h1 == h2

    def test_different_files_different_hash(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("content a", encoding="utf-8")
        b.write_text("content b", encoding="utf-8")
        assert calculate_hash(a) != calculate_hash(b)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            calculate_hash(Path("/nonexistent/file.txt"))

    def test_unsupported_algorithm(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            calculate_hash(f, algorithm="md999")

    def test_default_sha256(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        h = calculate_hash(f)
        assert len(h) == 64  # SHA256 hexdigest length
        assert all(c in "0123456789abcdef" for c in h)

    def test_str_path_input(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data", encoding="utf-8")
        h = calculate_hash(str(f))
        assert len(h) == 64

    def test_large_file(self, tmp_path):
        f = tmp_path / "large.bin"
        f.write_bytes(b"x" * 1024 * 1024)  # 1MB
        h = calculate_hash(f)
        assert len(h) == 64

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        h = calculate_hash(f)
        assert len(h) == 64

    def test_sha1_algorithm(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        h = calculate_hash(f, algorithm="sha1")
        assert len(h) == 40  # SHA1 hexdigest length


class TestCalculateContentHash:
    """String content hashing via calculate_content_hash()."""

    def test_content_hash_consistent(self):
        h1 = calculate_content_hash("hello world")
        h2 = calculate_content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        assert calculate_content_hash("abc") != calculate_content_hash("xyz")

    def test_empty_string(self):
        h = calculate_content_hash("")
        assert len(h) == 64

    def test_chinese_characters(self):
        h = calculate_content_hash("中文搜索系统")
        assert len(h) == 64

    def test_unicode_content(self):
        h = calculate_content_hash("héllo wörld 🔍")
        assert len(h) == 64

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Content must be a string"):
            calculate_content_hash(123)  # type: ignore[arg-type]

    def test_none_raises(self):
        with pytest.raises(ValueError):
            calculate_content_hash(None)  # type: ignore[arg-type]

    def test_default_sha256(self):
        h = calculate_content_hash("test")
        assert len(h) == 64
        # Verify it's actually SHA256
        expected = hashlib.sha256(b"test").hexdigest()
        assert h == expected

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            calculate_content_hash("test", algorithm="bad-algo")

    def test_sha1_algorithm(self):
        h = calculate_content_hash("test", algorithm="sha1")
        assert len(h) == 40

    def test_numeric_string(self):
        h = calculate_content_hash("12345")
        assert len(h) == 64

    def test_multiline_content(self):
        content = "line1\nline2\nline3"
        h = calculate_content_hash(content)
        assert len(h) == 64
