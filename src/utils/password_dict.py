"""Password dictionary manager for encrypted document conversion.

Provides a default list of common passwords for enterprise documents,
and supports loading external password dictionary files for extension.

Usage:
    pdict = PasswordDictionary()
    pdict.load("/path/to/custom_passwords.txt")  # Optional: extend with custom file
    for password in pdict.passwords:
        ...  # Try each password

Environment variables:
    PASSWORD_DICT_PATH: Path to an external password dictionary file (one password per line).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default password list for enterprise document decryption.
# Covers: empty password (owner-only encryption), numeric sequences,
# common Chinese/English patterns, and typical corporate passwords.
# Inspired by oletools DEFAULT_PASSWORDS and common enterprise patterns.
_DEFAULT_PASSWORDS: List[str] = [
    # Empty password (owner-only encryption — very common in PDFs)
    "",
    # Numeric sequences (most common in Chinese enterprises)
    "123456",
    "1234",
    "123",
    "12345",
    "1234567",
    "12345678",
    "123456789",
    "1234567890",
    "000000",
    "111111",
    "666666",
    "888888",
    "999999",
    "0000",
    "1111",
    "6666",
    "8888",
    "9999",
    "4321",
    "321",
    # Common English passwords
    "password",
    "admin",
    "test",
    "guest",
    "root",
    # Corporate / document-specific patterns
    "admin123",
    "password123",
    "changeme",
    "letmein",
    "welcome",
    "test123",
    "abc123",
    "qwerty",
    # Common Chinese pinyin passwords
    "mima",
    "ceshi",
    # Company name patterns (often used as archive passwords)
    "cigna",
    "cmb",
    # Mixed case variants of above
    "Admin",
    "Admin123",
    "Password",
    "Password123",
    "P@ssw0rd",
    "P@ssword1",
    "Admin@123",
    "Aa123456",
    # Office transparent passwords (used by PowerPoint for protection)
    # See: https://docs.microsoft.com/en-us/openspecs/office_file_formats/ms-offcrypto/
]


class PasswordDictionary:
    """Manages password dictionaries for encrypted document conversion.

    Combines a built-in default password list with optional external
    dictionary files. Passwords are deduplicated while preserving order
    (defaults first, then custom additions).
    """

    def __init__(
        self,
        dict_path: Optional[str] = None,
        include_defaults: bool = True,
    ):
        """Initialize the password dictionary.

        Args:
            dict_path: Path to an external password dictionary file.
                       One password per line, UTF-8 encoding.
                       Lines starting with # are comments.
                       Blank lines are ignored.
            include_defaults: Whether to include the built-in default passwords.
                              Set to False to use ONLY the external dictionary.
        """
        self._passwords: List[str] = []
        self._seen: set = set()
        self._dict_path: Optional[str] = dict_path
        self._include_defaults = include_defaults
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load passwords on first access."""
        if self._loaded:
            return
        self._loaded = True

        # 1. Add default passwords
        if self._include_defaults:
            for pwd in _DEFAULT_PASSWORDS:
                self._add(pwd)

        # 2. Load external dictionary from env var
        env_path = os.environ.get("PASSWORD_DICT_PATH", "")
        if env_path and os.path.isfile(env_path):
            self._load_file(env_path)

        # 3. Load explicitly passed dictionary (overrides env)
        if self._dict_path and os.path.isfile(self._dict_path):
            self._load_file(self._dict_path)

    def _add(self, password: str) -> None:
        """Add a password if not already present."""
        if password not in self._seen:
            self._seen.add(password)
            self._passwords.append(password)

    def _load_file(self, path: str) -> None:
        """Load passwords from a text file.

        Format: one password per line, UTF-8.
        Lines starting with # are comments.
        Blank lines and trailing whitespace are ignored.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    # Skip empty lines and comments
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    self._add(stripped)
            logger.debug("Loaded %d passwords from %s", len(self._passwords), path)
        except Exception as e:
            logger.warning("Failed to load password dictionary from %s: %s", path, e)

    def load(self, path: str) -> None:
        """Load additional passwords from an external file.

        Can be called multiple times to merge multiple dictionary files.
        Passwords are deduplicated.

        Args:
            path: Path to the password dictionary file (UTF-8, one per line).
        """
        self._ensure_loaded()
        if os.path.isfile(path):
            self._load_file(path)
        else:
            logger.warning("Password dictionary file not found: %s", path)

    def add(self, password: str) -> None:
        """Add a single password to the dictionary.

        Args:
            password: The password to add.
        """
        self._ensure_loaded()
        self._add(password)

    @property
    def passwords(self) -> List[str]:
        """Return the combined password list (defaults + custom, deduplicated)."""
        self._ensure_loaded()
        return list(self._passwords)  # Return copy to prevent mutation

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._passwords)

    def __iter__(self):
        self._ensure_loaded()
        return iter(self._passwords)

    def __contains__(self, password: str) -> bool:
        self._ensure_loaded()
        return password in self._seen

    def __repr__(self) -> str:
        self._ensure_loaded()
        return f"PasswordDictionary({len(self._passwords)} passwords)"
