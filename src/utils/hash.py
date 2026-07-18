"""Hash utility functions for file and content hashing."""

import hashlib
from pathlib import Path


def calculate_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a file's content.

    Args:
        file_path: Path to the file to hash
        algorithm: Hash algorithm to use (default: "sha256")

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm is not supported
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    with file_path.open("rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def calculate_content_hash(content: str, algorithm: str = "sha256") -> str:
    """Calculate hash of a string content.

    Args:
        content: String content to hash
        algorithm: Hash algorithm to use (default: "sha256")

    Returns:
        Hexadecimal hash string

    Raises:
        ValueError: If algorithm is not supported
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    # Encode to bytes using UTF-8
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()
