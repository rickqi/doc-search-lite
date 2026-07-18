"""BashTool - Simulated read-only shell commands for agent file exploration.

Provides terminal-like search capabilities over raw markdown files using
pure Python (no subprocess). Implements DCI-Agent-Lite paradigm commands:
rg, grep, find, head, tail, cat, wc, ls/dir.

All operations are path-sandboxed to raw_dir and strictly read-only.
"""

import re
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.agent.base import Tool
from src.agent.tool_types import ToolResult

# Safety limits
_MAX_OUTPUT_CHARS = 8000
_MAX_SEARCH_RESULTS = 200
_MAX_FIND_RESULTS = 100
_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Allowed commands (read-only, no shell)
_ALLOWED_COMMANDS = frozenset({"rg", "grep", "find", "head", "tail", "cat", "wc", "ls", "dir"})


def _resolve_sandboxed(path_str: str, raw_dir: Path) -> Path | None:
    """Resolve a path string within the raw_dir sandbox.

    Args:
        path_str: User-provided path string (may be relative).
        raw_dir: The sandbox root directory.

    Returns:
        Resolved Path if inside sandbox, None if escape attempted.
    """
    raw_dir = raw_dir.resolve()
    candidate = (raw_dir / path_str).resolve()
    try:
        candidate.relative_to(raw_dir)
    except ValueError:
        return None
    return candidate


def _read_lines(filepath: Path) -> list[str]:
    """Read file lines, respecting size limits.

    Args:
        filepath: Path to file.

    Returns:
        List of lines, or empty list on error.
    """
    try:
        if filepath.stat().st_size > _MAX_FILE_SIZE:
            return [f"[文件过大，跳过: {filepath.name} (>5MB)]"]
    except OSError:
        return []

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return content.splitlines()


def _truncate_output(output: str) -> str:
    """Truncate output to maximum character limit.

    Args:
        output: String to potentially truncate.

    Returns:
        Original or truncated string.
    """
    if len(output) > _MAX_OUTPUT_CHARS:
        return output[:_MAX_OUTPUT_CHARS] + "\n...[输出已截断]"
    return output


def _parse_int_flag(args: list[str], flag: str, default: int) -> int:
    """Extract an integer value following a flag from args.

    Supports both `-N 10` and `-N10` forms.

    Args:
        args: Argument list.
        flag: Flag name (e.g. '-n', '--max-count').
        default: Default value if flag not found.

    Returns:
        Parsed integer value.
    """
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return default
        # Handle combined form like -n10 or -A3
        if arg.startswith(flag) and len(arg) > len(flag):
            try:
                return int(arg[len(flag):])
            except ValueError:
                pass
    return default


def _has_flag(args: list[str], flag: str) -> bool:
    """Check if a flag is present in args.

    Args:
        args: Argument list.
        flag: Flag to look for.

    Returns:
        True if flag is present.
    """
    return flag in args


def _extract_flag_value(args: list[str], flag: str, default: str) -> str:
    """Extract a string value following a flag from args.

    Args:
        args: Argument list.
        flag: Flag name (e.g. '-name').
        default: Default value if flag not found.

    Returns:
        The flag's value or default.
    """
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
    return default


def _extract_positional_args(
    args: list[str],
    *,
    bool_flags: set | None = None,
    value_flags: set | None = None,
) -> list[str]:
    """Extract positional arguments (non-flag and non-flag-value).

    Args:
        args: Full argument list.
        bool_flags: Flags that take no value (e.g. '-i', '-c', '-l').
        value_flags: Flags that take a value (e.g. '-n', '-A', '--max-count').

    Returns:
        List of positional arguments.
    """
    bool_flags = bool_flags or set()
    value_flags = value_flags or set()
    positionals: list[str] = []
    skip_next = False
    for _i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in bool_flags:
            continue
        if arg in value_flags:
            skip_next = True  # skip the value after this flag
            continue
        # Handle combined value flags like -n10, -A3, --max-count50
        is_combined_value = False
        for vf in value_flags:
            if arg.startswith(vf) and len(arg) > len(vf):
                is_combined_value = True
                break
        if is_combined_value:
            continue
        # Skip any other flags we don't recognize
        if arg.startswith("-"):
            continue
        positionals.append(arg)
    return positionals


class BashTool(Tool):
    """Read-only shell command simulator for agent file exploration.

    Implements common Unix search/read commands in pure Python,
    sandboxed to a specific directory. No subprocess calls.

    Supported commands:
        rg   — ripgrep-style regex search (most powerful)
        grep — basic regex search (delegates to rg)
        find — file discovery by name pattern
        head — first N lines of a file
        tail — last N lines of a file
        cat  — full file content
        wc   — word/line/char count
        ls   — directory listing (also as 'dir')
    """

    def __init__(
        self,
        raw_dir: Path,
        max_output: int = _MAX_OUTPUT_CHARS,
        max_file_size: int = _MAX_FILE_SIZE,
    ) -> None:
        """Initialize BashTool.

        Args:
            raw_dir: Sandbox directory for all file operations.
            max_output: Maximum output characters before truncation.
            max_file_size: Skip files larger than this (bytes).
        """
        self._raw_dir = Path(raw_dir).resolve()
        self._max_output = max_output
        self._max_file_size = max_file_size

        # Command dispatch table
        self._commands: dict[str, Callable[[list[str]], ToolResult]] = {
            "rg": self._cmd_rg,
            "grep": self._cmd_grep,
            "find": self._cmd_find,
            "head": self._cmd_head,
            "tail": self._cmd_tail,
            "cat": self._cmd_cat,
            "wc": self._cmd_wc,
            "ls": self._cmd_ls,
            "dir": self._cmd_ls,
        }

    # ------------------------------------------------------------------
    # Tool ABC implementation
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "bash"

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        return (
            "模拟只读Shell命令，在文档目录中搜索和浏览文件。"
            "支持命令: rg(正则搜索), grep, find(文件查找), "
            "head, tail, cat(文件查看), wc(统计), ls(列表)。"
            "所有操作限制在文档目录内，只读。"
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a simulated shell command.

        Args:
            command: The command string to simulate (e.g. 'rg -i 年假 .').

        Returns:
            ToolResult with command output or error.
        """
        command = kwargs.get("command", "").strip()
        if not command:
            return ToolResult.fail(error="参数 'command' 不能为空")

        # Parse command name and arguments
        parts = command.split(None, 1)
        cmd_name = parts[0].lower()

        # Parse remaining args with shlex for proper quoting
        try:
            cmd_args = shlex.split(parts[1]) if len(parts) > 1 else []
        except ValueError as exc:
            return ToolResult.fail(error=f"命令参数解析失败: {exc}")

        # Check command allowlist
        if cmd_name not in _ALLOWED_COMMANDS:
            allowed = ", ".join(sorted(_ALLOWED_COMMANDS))
            return ToolResult.fail(
                error=f"不支持的命令: {cmd_name}。允许的命令: {allowed}"
            )

        # Verify sandbox directory exists
        if not self._raw_dir.is_dir():
            return ToolResult.fail(
                error=f"文档目录不存在: {self._raw_dir}"
            )

        handler = self._commands[cmd_name]
        return handler(cmd_args)

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "要执行的Shell命令。支持: "
                                "rg [-i] [-c] [-l] [-n] [--max-count N] [-A N] [-B N] PATTERN [PATH], "
                                "grep [-i] [-c] [-l] PATTERN [PATH], "
                                "find [PATH] [-name PATTERN] [-type f|d], "
                                "head [-n N] FILE, "
                                "tail [-n N] FILE, "
                                "cat FILE, "
                                "wc [-l] [-w] FILE, "
                                "ls [PATH]"
                            ),
                        },
                    },
                    "required": ["command"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_rg(self, args: list[str]) -> ToolResult:
        """Simulate ripgrep: rg [OPTIONS] PATTERN [PATH].

        Options:
            -i             Case insensitive
            -c             Count matches per file
            -l             List filenames only
            -n             Show line numbers (always on)
            --max-count N  Max matches per file (default 50)
            -A N           Context lines after match (default 2)
            -B N           Context lines before match (default 2)
        """
        start_time = time.time()

        case_insensitive = _has_flag(args, "-i")
        count_mode = _has_flag(args, "-c")
        list_files = _has_flag(args, "-l")
        max_count = _parse_int_flag(args, "--max-count", 50)
        context_after = _parse_int_flag(args, "-A", 2)
        _parse_int_flag(args, "-B", 2)

        # Extract pattern and path from positional args
        positionals = _extract_positional_args(
            args,
            bool_flags={"-i", "-c", "-l", "-n"},
            value_flags={"--max-count", "-A", "-B"},
        )

        if not positionals:
            return ToolResult.fail(error="rg 需要提供搜索模式")

        pattern_str = positionals[0]
        search_subpath = positionals[1] if len(positionals) > 1 else "."

        # Compile regex
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern_str, flags)
        except re.error as exc:
            return ToolResult.fail(error=f"无效的正则表达式: {exc}")

        # Resolve search path within sandbox
        search_root = _resolve_sandboxed(search_subpath, self._raw_dir)
        if search_root is None:
            return ToolResult.fail(error=f"路径超出沙箱范围: {search_subpath}")

        if not search_root.exists():
            return ToolResult.fail(error=f"路径不存在: {search_subpath}")

        # Determine search scope
        if search_root.is_file():
            md_files = [search_root]
        else:
            md_files = sorted(search_root.rglob("*.md"))

        results: list[str] = []
        total_matches = 0
        files_searched = 0

        for md_file in md_files:
            if not md_file.is_file():
                continue

            # Size check
            try:
                if md_file.stat().st_size > self._max_file_size:
                    continue
            except OSError:
                continue

            files_searched += 1
            lines = _read_lines(md_file)
            matches_in_file = 0

            for i, line in enumerate(lines):
                if regex.search(line):
                    try:
                        rel = md_file.relative_to(self._raw_dir)
                    except ValueError:
                        continue

                    if count_mode:
                        matches_in_file += 1
                    elif list_files:
                        results.append(str(rel))
                        break
                    else:
                        results.append(f"{rel}:{i + 1}:{line.rstrip()}")
                        # Context after
                        for j in range(1, context_after + 1):
                            if i + j < len(lines):
                                results.append(
                                    f"{rel}-{i + 1 + j}-{lines[i + j].rstrip()}"
                                )
                        matches_in_file += 1
                        total_matches += 1

                    if matches_in_file >= max_count:
                        break

            if count_mode and matches_in_file > 0:
                try:
                    rel = md_file.relative_to(self._raw_dir)
                except ValueError:
                    continue
                results.append(f"{rel}:{matches_in_file}")

            if len(results) >= _MAX_SEARCH_RESULTS:
                break

        elapsed = time.time() - start_time
        output = "\n".join(results[:_MAX_SEARCH_RESULTS]) or "No matches found."

        return ToolResult.ok(
            data=_truncate_output(output),
            metadata={
                "command": "rg",
                "total_matches": total_matches,
                "files_searched": files_searched,
                "execution_time": round(elapsed, 3),
            },
        )

    def _cmd_grep(self, args: list[str]) -> ToolResult:
        """Simulate grep: grep [OPTIONS] PATTERN [PATH].

        Delegates to _cmd_rg for implementation.
        """
        return self._cmd_rg(args)

    def _cmd_find(self, args: list[str]) -> ToolResult:
        """Simulate find: find [PATH] [-name PATTERN] [-type f|d]."""
        start_time = time.time()

        name_pattern = _extract_flag_value(args, "-name", "*")
        file_type = _extract_flag_value(args, "-type", "f")

        # Extract search root from first positional arg
        positionals = _extract_positional_args(args, bool_flags=set(), value_flags={"-name", "-type"})
        search_subpath = positionals[0] if positionals else "."

        search_root = _resolve_sandboxed(search_subpath, self._raw_dir)
        if search_root is None:
            return ToolResult.fail(error=f"路径超出沙箱范围: {search_subpath}")

        if not search_root.exists():
            return ToolResult.fail(error=f"路径不存在: {search_subpath}")

        results: list[str] = []

        # Use rglob with name pattern, or walk everything if '*'
        if name_pattern == "*":
            paths = sorted(search_root.rglob("*"))
        else:
            paths = sorted(search_root.rglob(name_pattern))

        for p in paths:
            try:
                rel = p.relative_to(self._raw_dir)
            except ValueError:
                continue

            if file_type == "f" and p.is_file():
                results.append(str(rel))
            elif file_type == "d" and p.is_dir():
                results.append(str(rel) + "/")

            if len(results) >= _MAX_FIND_RESULTS:
                break

        elapsed = time.time() - start_time
        output = "\n".join(results) or "No files found."

        return ToolResult.ok(
            data=_truncate_output(output),
            metadata={
                "command": "find",
                "total_found": len(results),
                "execution_time": round(elapsed, 3),
            },
        )

    def _cmd_head(self, args: list[str]) -> ToolResult:
        """Simulate head: head [-n N] FILE."""
        n = _parse_int_flag(args, "-n", 10)

        # Extract filepath
        positionals = _extract_positional_args(args, bool_flags=set(), value_flags={"-n"})
        if not positionals:
            return ToolResult.fail(error="head 需要指定文件路径")

        filepath = _resolve_sandboxed(positionals[0], self._raw_dir)
        if filepath is None:
            return ToolResult.fail(error=f"路径超出沙箱范围: {positionals[0]}")

        if not filepath.is_file():
            return ToolResult.fail(error=f"文件不存在: {positionals[0]}")

        lines = _read_lines(filepath)
        output = "\n".join(lines[:n])

        return ToolResult.ok(
            data=_truncate_output(output),
            metadata={
                "command": "head",
                "lines_shown": min(n, len(lines)),
                "total_lines": len(lines),
            },
        )

    def _cmd_tail(self, args: list[str]) -> ToolResult:
        """Simulate tail: tail [-n N] FILE."""
        n = _parse_int_flag(args, "-n", 10)

        positionals = _extract_positional_args(args, bool_flags=set(), value_flags={"-n"})
        if not positionals:
            return ToolResult.fail(error="tail 需要指定文件路径")

        filepath = _resolve_sandboxed(positionals[0], self._raw_dir)
        if filepath is None:
            return ToolResult.fail(error=f"路径超出沙箱范围: {positionals[0]}")

        if not filepath.is_file():
            return ToolResult.fail(error=f"文件不存在: {positionals[0]}")

        lines = _read_lines(filepath)
        output = "\n".join(lines[-n:] if n > 0 else [])

        return ToolResult.ok(
            data=_truncate_output(output),
            metadata={
                "command": "tail",
                "lines_shown": min(n, len(lines)),
                "total_lines": len(lines),
            },
        )

    def _cmd_cat(self, args: list[str]) -> ToolResult:
        """Simulate cat: cat FILE [FILE...]."""
        if not args:
            return ToolResult.fail(error="cat 需要指定文件路径")

        outputs: list[str] = []

        for arg in args:
            filepath = _resolve_sandboxed(arg, self._raw_dir)
            if filepath is None:
                outputs.append(f"[路径超出沙箱范围: {arg}]")
                continue

            if not filepath.is_file():
                outputs.append(f"[文件不存在: {arg}]")
                continue

            # Size check
            try:
                if filepath.stat().st_size > self._max_file_size:
                    outputs.append(f"[文件过大，跳过: {arg} (>5MB)]")
                    continue
            except OSError:
                outputs.append(f"[无法读取: {arg}]")
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                outputs.append(f"[读取失败: {arg}]")
                continue

            outputs.append(content)

        combined = "\n".join(outputs)
        if not combined.strip():
            return ToolResult.ok(data="No files to display.")

        return ToolResult.ok(
            data=_truncate_output(combined),
            metadata={"command": "cat", "files_processed": len(args)},
        )

    def _cmd_wc(self, args: list[str]) -> ToolResult:
        """Simulate wc: wc [-l] [-w] FILE."""
        lines_mode = _has_flag(args, "-l")
        words_mode = _has_flag(args, "-w")

        positionals = _extract_positional_args(args, bool_flags={"-l", "-w"}, value_flags=set())
        if not positionals:
            return ToolResult.fail(error="wc 需要指定文件路径")

        filepath = _resolve_sandboxed(positionals[0], self._raw_dir)
        if filepath is None:
            return ToolResult.fail(error=f"路径超出沙箱范围: {positionals[0]}")

        if not filepath.is_file():
            return ToolResult.fail(error=f"文件不存在: {positionals[0]}")

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ToolResult.fail(error=f"读取失败: {positionals[0]}")

        line_count = len(content.splitlines())
        word_count = len(content.split())
        char_count = len(content)

        if lines_mode and not words_mode:
            result = str(line_count)
        elif words_mode and not lines_mode:
            result = str(word_count)
        else:
            # Default: lines words chars
            result = f"{line_count} {word_count} {char_count}"

        return ToolResult.ok(
            data=result,
            metadata={
                "command": "wc",
                "lines": line_count,
                "words": word_count,
                "chars": char_count,
            },
        )

    def _cmd_ls(self, args: list[str]) -> ToolResult:
        """Simulate ls: ls [PATH]."""
        start_time = time.time()

        # Extract path (default to raw_dir root)
        subpath = args[0] if args and not args[0].startswith("-") else "."

        target = _resolve_sandboxed(subpath, self._raw_dir)
        if target is None:
            return ToolResult.fail(error=f"路径超出沙箱范围: {subpath}")

        if not target.exists():
            return ToolResult.fail(error=f"路径不存在: {subpath}")

        if target.is_file():
            # Show single file info
            try:
                rel = target.relative_to(self._raw_dir)
            except ValueError:
                return ToolResult.fail(error="路径超出沙箱范围")
            size = target.stat().st_size
            return ToolResult.ok(
                data=f"{rel}  ({size} bytes)",
                metadata={"command": "ls"},
            )

        # Directory listing
        entries: list[str] = []
        try:
            for child in sorted(target.iterdir()):
                try:
                    rel = child.relative_to(self._raw_dir)
                except ValueError:
                    continue
                name = str(rel)
                if child.is_dir():
                    name += "/"
                entries.append(name)
        except OSError:
            return ToolResult.fail(error=f"无法列出目录: {subpath}")

        elapsed = time.time() - start_time
        output = "\n".join(entries) or "(空目录)"

        return ToolResult.ok(
            data=_truncate_output(output),
            metadata={
                "command": "ls",
                "entries": len(entries),
                "execution_time": round(elapsed, 3),
            },
        )
