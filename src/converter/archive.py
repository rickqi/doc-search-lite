"""
Archive to Markdown converter.

Extracts archive contents (ZIP, 7z, RAR, tar.gz/bz2/xz) to a temporary directory,
then recursively converts supported files using ConverterCoordinator.
Generates a summary Markdown with directory tree + converted content.
"""

import logging
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

from .base import ConvertResult, Converter

logger = logging.getLogger(__name__)

# Maximum safety limits
_MAX_MEMBER_COUNT = 10_000
_MAX_UNCOMPRESSED_SIZE = 512 * 1024 * 1024  # 512 MB
_MAX_RECURSION_DEPTH = 3

# 7-Zip executable search paths (used as RAR extraction fallback)
_7Z_PATHS = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "/usr/bin/7z",
    "/usr/local/bin/7z",
    "/usr/bin/p7zip",
]


def _find_7z() -> Optional[str]:
    """Find 7-Zip executable."""
    env_path = os.getenv("SEVENZIP_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    # Check PATH
    found = shutil.which("7z")
    if found:
        return found
    for p in _7Z_PATHS:
        if os.path.isfile(p):
            return p
    return None

# Extensions we know how to extract
ARCHIVE_EXTENSIONS = {
    ".zip", ".7z", ".rar",
    ".tar", ".gz", ".tgz", ".bz2", ".xz",
}

# Compound extensions that Path.suffix won't detect
_COMPOUND_EXTENSIONS = {".tar.gz", ".tar.bz2", ".tar.xz"}


class ArchiveConverter(Converter):
    """Converter for archive files (ZIP, 7z, RAR, tar.gz/bz2/xz).

    Extracts contents to ``output_dir / "{stem}_extracted/"``, recursively
    converts supported files using the coordinator, and produces a single
    summary Markdown document.
    """

    def __init__(self, coordinator=None):
        # Lazy back-reference to ConverterCoordinator for recursive conversion.
        self._coordinator = coordinator

    # -- Converter ABC -------------------------------------------------------

    @property
    def name(self) -> str:
        return "ArchiveConverter"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def supported_formats(self) -> List[str]:
        return sorted(ARCHIVE_EXTENSIONS)

    def can_convert(self, file_path: Path) -> bool:
        """Override to handle compound extensions like .tar.gz."""
        suffix = file_path.suffix.lower()
        if suffix in ARCHIVE_EXTENSIONS:
            return True
        # Check compound extensions
        name_lower = file_path.name.lower()
        for ext in _COMPOUND_EXTENSIONS:
            if name_lower.endswith(ext):
                return True
        return False

    def convert(
        self,
        source: Path,
        output_dir: Path,
        options: Optional[Dict] = None,
    ) -> ConvertResult:
        """Convert an archive to a summary Markdown document."""
        start_time = time.time()
        options = options or {}
        errors: List[str] = []

        # Resolve coordinator reference
        coordinator = self._coordinator
        if coordinator is None:
            coordinator = options.get("_coordinator")

        # Current recursion depth
        depth = options.get("_archive_depth", 0)

        # 1. Validate source
        if not source.exists():
            return ConvertResult(
                success=False,
                markdown="",
                errors=[f"Source file not found: {source}"],
                source_file=source,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        if not os.access(source, os.R_OK):
            return ConvertResult(
                success=False,
                markdown="",
                errors=[f"Source file is not readable: {source}"],
                source_file=source,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        # 2. Detect format
        try:
            fmt = self._detect_format(source)
        except Exception as exc:
            return ConvertResult(
                success=False,
                markdown="",
                errors=[f"Cannot detect archive format: {exc}"],
                source_file=source,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        # 3. Create extraction directory
        extract_dir = output_dir / f"{source.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 4. Extract
        try:
            extracted: List[Path]
            extract_error: Optional[str] = None

            if fmt == "zip":
                extracted, extract_error = self._extract_zip(source, extract_dir)
            elif fmt == "7z":
                extracted, extract_error = self._extract_7z(source, extract_dir)
            elif fmt == "rar":
                extracted, extract_error = self._extract_rar(source, extract_dir)
            else:  # tar variants
                extracted, extract_error = self._extract_tar(source, extract_dir, fmt)

            if extract_error:
                errors.append(extract_error)
                if not extracted:
                    return ConvertResult(
                        success=False,
                        markdown="",
                        errors=errors,
                        source_file=source,
                        output_file=None,
                        converter_name=self.name,
                        converter_version=self.version,
                        convert_time=time.time() - start_time,
                    )
        except Exception as exc:
            return ConvertResult(
                success=False,
                markdown="",
                errors=[f"Extraction failed: {exc}"],
                source_file=source,
                output_file=None,
                converter_name=self.name,
                converter_version=self.version,
                convert_time=time.time() - start_time,
            )

        # 5. Classify extracted files
        supported_extensions = self._get_supported_extensions(coordinator)

        total_files = 0
        supported_files: List[Path] = []
        skipped_files: List[Path] = []

        for f in extracted:
            if f.is_file():
                total_files += 1
                if self._is_supported_file(f, supported_extensions):
                    supported_files.append(f)
                else:
                    skipped_files.append(f)

        # 6. Recursively convert supported files
        converted_contents: Dict[str, str] = {}  # relative_path -> markdown
        converted_paths: Set[Path] = set()

        if depth < _MAX_RECURSION_DEPTH and coordinator is not None:
            child_options = dict(options)
            child_options["_archive_depth"] = depth + 1
            child_options["_coordinator"] = coordinator

            for file_path in supported_files:
                try:
                    rel = file_path.relative_to(extract_dir)
                    result = coordinator.convert(
                        source=file_path,
                        output_dir=extract_dir,
                        options=child_options,
                    )
                    if result.success and result.markdown.strip():
                        converted_contents[str(rel).replace("\\", "/")] = result.markdown
                        converted_paths.add(file_path)
                except Exception as exc:
                    logger.warning("Failed to convert %s inside archive: %s", file_path, exc)
        elif depth >= _MAX_RECURSION_DEPTH:
            logger.warning("Max archive recursion depth (%d) reached, skipping nested conversion", _MAX_RECURSION_DEPTH)

        skipped_count = total_files - len(converted_contents)

        # 7. Clean up original extracted files after successful conversion.
        #    Remove source files that were converted, keeping only generated
        #    .md output and conversion failures for inspection.
        self._cleanup_extracted_originals(extract_dir, converted_paths)

        # 8. Build summary Markdown
        tree = self._build_tree(extract_dir, converted_paths)
        md = self._build_markdown(
            archive_name=source.name,
            total=total_files,
            supported_count=len(converted_contents),
            skipped_count=skipped_count,
            tree=tree,
            converted_contents=converted_contents,
        )

        # 9. Write output
        output_file = output_dir / f"{source.stem}.md"
        try:
            output_file.write_text(md, encoding="utf-8")
        except OSError as exc:
            errors.append(f"Failed to write output: {exc}")

        return ConvertResult(
            success=True,
            markdown=md,
            source_file=source,
            output_file=output_file,
            converter_name=self.name,
            converter_version=self.version,
            convert_time=time.time() - start_time,
            errors=errors,
            metadata={
                "archive_format": fmt,
                "total_files": total_files,
                "converted_files": len(converted_contents),
                "skipped_files": skipped_count,
                "extract_dir": str(extract_dir),
            },
        )

    # -- Format detection ----------------------------------------------------

    @staticmethod
    def _detect_format(source: Path) -> str:
        """Detect archive format from magic bytes."""
        with open(source, "rb") as f:
            header = f.read(8)

        if len(header) < 2:
            raise ValueError("File too small to detect archive format")

        # Check compound extensions first (tar.gz, tar.bz2, tar.xz)
        name_lower = source.name.lower()
        if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
            return "tar.gz"
        if name_lower.endswith(".tar.bz2"):
            return "tar.bz2"
        if name_lower.endswith(".tar.xz"):
            return "tar.xz"
        if name_lower.endswith(".tar"):
            return "tar"

        # Magic bytes
        if header[:4] == b"PK\x03\x04":
            return "zip"
        # RAR signature: "Rar!\x1a\x07" (6 bytes)
        # RAR4: "Rar!\x1a\x07\x00" (7 bytes) | RAR5: "Rar!\x1a\x07\x01\x00" (8 bytes)
        if header[:6] == b"Rar!\x1a\x07":
            return "rar"
        if header[:6] == b"7z\xbc\xaf\x27\x1c":
            return "7z"
        if header[:2] == b"\x1f\x8b":
            return "tar.gz"
        if header[:3] == b"BZh":
            return "tar.bz2"
        if header[:6] == b"\xfd7zXZ\x00":
            return "tar.xz"

        raise ValueError(
            f"Unknown archive format (header: {header[:8].hex()})"
        )

    # -- Extraction methods --------------------------------------------------

    def _extract_zip(self, archive: Path, dest: Path) -> tuple:
        """Extract a ZIP archive safely. Returns (extracted_paths, error)."""
        extracted: List[Path] = []
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                # Check for encrypted archive — try password dictionary
                has_encrypted = any(info.flag_bits & 0x1 for info in zf.infolist())
                pwd = None
                if has_encrypted:
                    # Try password dictionary (includes empty password)
                    from src.utils.password_dict import PasswordDictionary
                    pdict = PasswordDictionary()
                    for candidate in pdict:
                        try:
                            pwd_bytes = candidate.encode("utf-8") if candidate else b""
                            zf.testzip()
                            # testzip() succeeds with no pwd on non-encrypted,
                            # but we need to actually test with the password
                            zf.setpassword(pwd_bytes)
                            zf.testzip()
                            pwd = pwd_bytes
                            logger.info(
                                "ZIP decrypted with dictionary password (len=%d): %s",
                                len(candidate),
                                "***" if candidate else "(empty)",
                            )
                            break
                        except Exception:
                            continue

                    if pwd is None:
                        return [], (
                            "Archive is password-protected. "
                            "None of the %d dictionary passwords worked."
                            % len(pdict)
                        )

                # Safety: member count
                if len(zf.infolist()) > _MAX_MEMBER_COUNT:
                    return [], f"Too many members ({len(zf.infolist())}), max {_MAX_MEMBER_COUNT}"

                # Safety: total uncompressed size
                total_size = sum(info.file_size for info in zf.infolist())
                if total_size > _MAX_UNCOMPRESSED_SIZE:
                    return [], f"Uncompressed size too large ({total_size}), max {_MAX_UNCOMPRESSED_SIZE}"

                for info in zf.infolist():
                    # Skip symlinks and directories
                    if info.is_dir():
                        continue
                    if info.external_attr >> 28 == 0xA:
                        continue

                    target = dest / info.filename
                    if not self._safe_extract_check(target, dest):
                        logger.warning("Skipping unsafe path: %s", info.filename)
                        continue

                    zf.extract(info, dest, pwd=pwd)
                    extracted.append(target)

            return extracted, None
        except zipfile.BadZipFile as exc:
            return [], f"Bad ZIP file: {exc}"
        except Exception as exc:
            return [], f"ZIP extraction error: {exc}"

    def _extract_7z(self, archive: Path, dest: Path) -> tuple:
        """Extract a 7z archive using py7zr (lazy import)."""
        try:
            import py7zr  # noqa: lazy import
        except ImportError:
            return [], "py7zr not installed"

        extracted: List[Path] = []
        try:
            with py7zr.SevenZipFile(archive, mode="r") as zf:
                # Check password — try password dictionary
                if zf.needs_password():
                    from src.utils.password_dict import PasswordDictionary
                    pdict = PasswordDictionary()
                    found_pwd = None
                    for candidate in pdict:
                        try:
                            zf.reset()
                            with py7zr.SevenZipFile(archive, mode="r", password=candidate) as zf2:
                                zf2.extractall(path=dest)
                                # Collect extracted files
                                for f in dest.rglob("*"):
                                    if f.is_file() and not f.is_symlink():
                                        if self._safe_extract_check(f, dest):
                                            extracted.append(f)
                            logger.info(
                                "7z decrypted with dictionary password (len=%d): %s",
                                len(candidate),
                                "***" if candidate else "(empty)",
                            )
                            found_pwd = candidate
                            break
                        except Exception:
                            continue

                    if found_pwd is not None:
                        return extracted, None
                    return [], (
                        "Archive is password-protected. "
                        "None of the %d dictionary passwords worked."
                        % len(pdict)
                    )

                all_names = zf.getnames()
                if len(all_names) > _MAX_MEMBER_COUNT:
                    return [], f"Too many members ({len(all_names)}), max {_MAX_MEMBER_COUNT}"

                zf.extractall(path=dest)

            # Collect extracted files
            for f in dest.rglob("*"):
                if f.is_file() and not f.is_symlink():
                    if self._safe_extract_check(f, dest):
                        extracted.append(f)

            return extracted, None
        except Exception as exc:
            return [], f"7z extraction error: {exc}"

    def _extract_rar(self, archive: Path, dest: Path) -> tuple:
        """Extract a RAR archive. Tries rarfile first, then 7-Zip as fallback."""
        # Try rarfile + unrar first
        try:
            import rarfile  # noqa: lazy import
        except ImportError:
            # rarfile not installed, try 7-Zip directly
            return self._extract_rar_via_7z(archive, dest)

        extracted: List[Path] = []
        try:
            with rarfile.RarFile(archive, "r") as rf:
                # Check password — try password dictionary
                if rf.needs_password():
                    from src.utils.password_dict import PasswordDictionary
                    pdict = PasswordDictionary()
                    found_pwd = None
                    for candidate in pdict:
                        try:
                            with rarfile.RarFile(archive, "r", pwd=candidate) as rf2:
                                infos = rf2.infolist()
                                if len(infos) > _MAX_MEMBER_COUNT:
                                    return [], f"Too many members ({len(infos)}), max {_MAX_MEMBER_COUNT}"
                                for info in infos:
                                    if info.is_dir():
                                        continue
                                    target = dest / info.filename
                                    if not self._safe_extract_check(target, dest):
                                        logger.warning("Skipping unsafe path: %s", info.filename)
                                        continue
                                    rf2.extract(info, dest)
                                    extracted.append(target)
                            logger.info(
                                "RAR decrypted with dictionary password (len=%d): %s",
                                len(candidate),
                                "***" if candidate else "(empty)",
                            )
                            found_pwd = candidate
                            break
                        except Exception:
                            continue

                    if found_pwd is not None:
                        return extracted, None
                    return [], (
                        "Archive is password-protected. "
                        "None of the %d dictionary passwords worked."
                        % len(pdict)
                    )

                infos = rf.infolist()
                if len(infos) > _MAX_MEMBER_COUNT:
                    return [], f"Too many members ({len(infos)}), max {_MAX_MEMBER_COUNT}"

                for info in infos:
                    if info.is_dir():
                        continue

                    target = dest / info.filename
                    if not self._safe_extract_check(target, dest):
                        logger.warning("Skipping unsafe path: %s", info.filename)
                        continue

                    rf.extract(info, dest)
                    extracted.append(target)

            return extracted, None
        except rarfile.NeedFirstVolume:
            return [], "RAR: multi-volume archive not supported"
        except rarfile.BadRarFile as exc:
            return [], f"Bad RAR file: {exc}"
        except rarfile.RarCannotExec:
            # unrar not installed — try 7-Zip fallback
            logger.info("rarfile backend (unrar) not found, trying 7-Zip")
            return self._extract_rar_via_7z(archive, dest)
        except Exception as exc:
            error_msg = str(exc)
            if any(kw in error_msg.lower() for kw in ("unrar", "nonetype", "cannot find working tool")):
                logger.info("rarfile failed, trying 7-Zip: %s", exc)
                return self._extract_rar_via_7z(archive, dest)
            return [], f"RAR extraction error: {exc}"

    def _extract_rar_via_7z(self, archive: Path, dest: Path) -> tuple:
        """Extract RAR using 7-Zip command-line tool (fallback)."""
        sevenz = _find_7z()
        if not sevenz:
            return [], (
                "RAR extraction requires either unrar or 7-Zip. "
                "Install one: winget install 7zip.7zip"
            )

        try:
            result = subprocess.run(
                [sevenz, "x", str(archive), f"-o{dest}", "-y", "-bso0", "-bse0"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Check for password-protected — try password dictionary
                if "password" in stderr.lower() or "wrong password" in stderr.lower():
                    # Try common passwords via 7z command
                    from src.utils.password_dict import PasswordDictionary
                    pdict = PasswordDictionary()
                    for candidate in pdict:
                        try:
                            result2 = subprocess.run(
                                [sevenz, "x", str(archive), f"-o{dest}", "-y",
                                 "-bso0", "-bse0", f"-p{candidate}"],
                                capture_output=True, text=True, timeout=120,
                            )
                            if result2.returncode == 0:
                                logger.info(
                                    "7z RAR decrypted with dictionary password (len=%d): %s",
                                    len(candidate),
                                    "***" if candidate else "(empty)",
                                )
                                break
                        except Exception:
                            continue
                    else:
                        return [], (
                            "Archive is password-protected. "
                            "None of the %d dictionary passwords worked."
                            % len(pdict)
                        )
                    # If we got here, one of the passwords worked — collect extracted files
                    # Fall through to collection below
                else:
                    return [], f"7-Zip RAR extraction failed (exit {result.returncode}): {stderr}"

            # Collect extracted files
            extracted: List[Path] = []
            count = 0
            for f in dest.rglob("*"):
                if f.is_file() and not f.is_symlink():
                    if self._safe_extract_check(f, dest):
                        extracted.append(f)
                        count += 1
                    if count > _MAX_MEMBER_COUNT:
                        return [], f"Too many members ({count}), max {_MAX_MEMBER_COUNT}"

            return extracted, None
        except subprocess.TimeoutExpired:
            return [], "7-Zip RAR extraction timed out"
        except Exception as exc:
            return [], f"7-Zip RAR extraction error: {exc}"

    def _extract_tar(self, archive: Path, dest: Path, fmt: str) -> tuple:
        """Extract a tar archive (gz/bz2/xz/plain)."""
        extracted: List[Path] = []
        try:
            # Map detected format to tarfile mode
            mode_map = {
                "tar": "r:",
                "tar.gz": "r:gz",
                "tar.bz2": "r:bz2",
                "tar.xz": "r:xz",
            }
            mode = mode_map.get(fmt, "r:*")

            with tarfile.open(archive, mode) as tf:
                members = tf.getmembers()
                if len(members) > _MAX_MEMBER_COUNT:
                    return [], f"Too many members ({len(members)}), max {_MAX_MEMBER_COUNT}"

                # Safety: total uncompressed size
                total_size = sum(m.size for m in members if m.isfile())
                if total_size > _MAX_UNCOMPRESSED_SIZE:
                    return [], f"Uncompressed size too large ({total_size}), max {_MAX_UNCOMPRESSED_SIZE}"

                for member in members:
                    # Skip directories
                    if not member.isfile():
                        continue
                    # Skip symlinks and hardlinks
                    if member.issym() or member.islnk():
                        continue

                    target = dest / member.name
                    if not self._safe_extract_check(target, dest):
                        logger.warning("Skipping unsafe path: %s", member.name)
                        continue

                    tf.extract(member, dest, set_attrs=False)
                    extracted.append(target)

            return extracted, None
        except tarfile.TarError as exc:
            return [], f"Tar extraction error: {exc}"
        except Exception as exc:
            return [], f"Tar extraction error: {exc}"

    # -- Safety --------------------------------------------------------------

    @staticmethod
    def _safe_extract_check(member_path: Path, dest: Path) -> bool:
        """Reject path traversal attacks."""
        try:
            resolved = member_path.resolve()
            dest_resolved = dest.resolve()
            return str(resolved).startswith(str(dest_resolved))
        except (OSError, ValueError):
            return False

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _get_supported_extensions(coordinator) -> Set[str]:
        """Get the set of supported file extensions from the coordinator."""
        if coordinator is not None and hasattr(coordinator, "supported_extensions"):
            return set(coordinator.supported_extensions)
        # Fallback: common supported extensions
        return {
            ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xls",
            ".html", ".htm", ".csv", ".txt",
            ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif",
        }

    @staticmethod
    def _is_supported_file(f: Path, extensions: Set[str]) -> bool:
        """Check if a file has a supported extension."""
        suffix = f.suffix.lower()
        if suffix in extensions:
            return True
        # Also check compound extensions
        name_lower = f.name.lower()
        for ext in _COMPOUND_EXTENSIONS:
            if name_lower.endswith(ext):
                return True
        return False

    @staticmethod
    def _cleanup_extracted_originals(extract_dir: Path, converted_paths: Set[Path]) -> None:
        """Remove original source files after successful conversion.

        After the coordinator converts supported files inside the extract
        directory, the original files (PDF, DOCX, images, etc.) are no longer
        needed — the generated .md files contain the extracted content.
        This keeps the raw output clean and avoids leaving behind artifacts
        that were only intermediate inputs.

        Directories and any remaining files (conversion failures, unknown
        formats) are preserved for inspection.
        """
        import shutil

        for f in converted_paths:
            try:
                if f.exists() and f.is_file() and f.suffix.lower() != ".md":
                    f.unlink()
            except OSError:
                pass

        # Also clean up empty directories left after file removal
        for root, dirs, files in os.walk(str(extract_dir), topdown=False):
            if root == str(extract_dir):
                continue  # keep the extract root
            try:
                if not os.listdir(root):
                    os.rmdir(root)
            except OSError:
                pass

    @staticmethod
    def _build_tree(extract_dir: Path, converted_paths: Set[Path]) -> str:
        """Generate a directory tree string of extracted files."""
        lines: List[str] = []
        extract_resolved = extract_dir.resolve()

        def _walk(directory: Path, prefix: str = ""):
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return

            dirs = [e for e in entries if e.is_dir() and not e.is_symlink()]
            files = [e for e in entries if e.is_file() and not e.is_symlink()]
            all_entries = dirs + files

            for i, entry in enumerate(all_entries):
                is_last = (i == len(all_entries) - 1)
                connector = "└── " if is_last else "├── "
                name = entry.name

                if entry in converted_paths:
                    name = f"✓ {name}"

                lines.append(f"{prefix}{connector}{name}")

                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    _walk(entry, prefix + extension)

        _walk(extract_resolved)
        return "\n".join(lines) if lines else "(empty)"

    @staticmethod
    def _build_markdown(
        archive_name: str,
        total: int,
        supported_count: int,
        skipped_count: int,
        tree: str,
        converted_contents: Dict[str, str],
    ) -> str:
        """Build the summary Markdown document."""
        parts: List[str] = [
            f"# {archive_name}",
            "",
            f"> 解压文件: {total} 个 | 支持: {supported_count} 个 | 跳过: {skipped_count} 个",
            "",
            "## 目录结构",
            "",
            "```",
            tree,
            "```",
            "",
        ]

        if converted_contents:
            parts.append("## 文件内容")
            parts.append("")
            for rel_path in sorted(converted_contents.keys()):
                content = converted_contents[rel_path]
                parts.append(f"### {rel_path}")
                parts.append("")
                parts.append(content)
                parts.append("")

        return "\n".join(parts)
