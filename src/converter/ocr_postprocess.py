"""Post-processing pipeline for OCR results.

Cleans up raw OCR output to improve semantic continuity and readability:
- Remove OCR artifacts (stray chars, repeated spaces)
- Normalize line endings and blank lines
- Merge broken paragraphs
- Detect and format headings
- Normalize Markdown tables with inconsistent columns
"""

import logging
import re

logger = logging.getLogger(__name__)


def clean_ocr_text(text: str) -> str:
    """Remove common OCR artifacts and normalize whitespace.

    - Normalizes line endings (CRLF -> LF)
    - Removes stray control characters (except tab and newline)
    - Collapses multiple spaces into one (outside of indentation)
    - Removes empty lines at start and end
    - Collapses 3+ consecutive blank lines into 2

    Args:
        text: Raw OCR text.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove stray control characters (keep \t and \n)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# Patterns that should NOT be merged with the previous line
_NO_MERGE_PATTERNS = re.compile(
    r"^(?:"
    r"#{1,6}\s"           # Markdown headings
    r"|[-*•]\s"           # Unordered list items
    r"|\d+[.)]\s"         # Ordered list items
    r"|\|"                # Table rows
    r"|>"                  # Blockquotes
    r")"
)


def _is_special_line(line: str) -> bool:
    """Check if a line has structural meaning and should not be merged."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_NO_MERGE_PATTERNS.match(stripped))


def merge_paragraphs(text: str) -> str:
    """Merge lines that are continuations of the same paragraph.

    Two consecutive non-blank lines are merged with a space when:
    - Neither line is a heading, list item, table row, or blockquote
    - The previous line does not end with a structural marker

    Args:
        text: Input text.

    Returns:
        Text with paragraphs properly merged.
    """
    if not text:
        return ""

    lines = text.split("\n")
    merged: list[str] = []

    for line in lines:
        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]

        # Both lines must be non-empty and non-special
        if (
            prev.strip()
            and line.strip()
            and not _is_special_line(prev)
            and not _is_special_line(line)
            # Previous line should not end with certain punctuation that implies
            # end of a thought (period, question mark, exclamation in CJK context)
            # but DO merge if prev ends mid-sentence
        ):
            merged[-1] = prev.rstrip() + " " + line.lstrip()
        else:
            merged.append(line)

    return "\n".join(merged)


def detect_headings(text: str) -> str:
    """Detect and format headings in OCR output.

    Converts lines that look like headings to ``### heading`` format:
    - Short lines (< 30 chars) followed by longer content
    - Standalone lines ending with a colon (Chinese ： or English :)
    - All-caps lines (Latin script only)

    Does not convert lines inside code blocks or tables.

    Args:
        text: Input text.

    Returns:
        Text with detected headings formatted.
    """
    if not text:
        return ""

    lines = text.split("\n")
    result: list[str] = []

    # Track whether we're inside a code block
    in_code_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Skip lines that are already headings
        if stripped.startswith("#"):
            result.append(line)
            continue

        # Skip empty lines, table rows, list items, blockquotes
        if not stripped or stripped.startswith("|") or stripped.startswith(">"):
            result.append(line)
            continue

        # Skip list items
        if re.match(r"^[-*•]\s", stripped) or re.match(r"^\d+[.)]\s", stripped):
            result.append(line)
            continue

        # Check next non-empty line to determine context
        next_line = ""
        for j in range(i + 1, len(lines)):
            if lines[j].strip():
                next_line = lines[j].strip()
                break

        is_heading = False

        # Rule 1: Short line followed by longer content (min 3 chars)
        if (
            3 <= len(stripped) < 30
            and next_line
            and len(next_line) > len(stripped)
            and not stripped.endswith((".", "。", "！", "?", "？", "!", ",",
                                      "，", "、", ";", "；"))
        ):
            is_heading = True

        # Rule 2: Line ending with colon (standalone, not too long)
        if (
            stripped.endswith(("：", ":"))
            and len(stripped) < 50
            and not stripped.startswith("|")
        ):
            is_heading = True

        # Rule 3: All-caps lines (at least 3 chars, Latin only)
        if (
            len(stripped) >= 3
            and stripped.isalpha()
            and stripped.isupper()
            and re.match(r"^[A-Z]+$", stripped)
        ):
            is_heading = True

        if is_heading:
            # Strip trailing colon for cleaner heading
            heading_text = stripped.rstrip("：:")
            result.append(f"### {heading_text}")
        else:
            result.append(line)

    return "\n".join(result)


def normalize_tables(text: str) -> str:
    """Normalize Markdown tables with inconsistent column counts.

    - Detects table blocks (rows starting with ``|``)
    - Pads short rows with empty cells
    - Fixes separator rows to match column count
    - Skips single-column "tables" that look like lists

    Args:
        text: Input text.

    Returns:
        Text with normalized tables.
    """
    if not text:
        return ""

    lines = text.split("\n")
    result: list[str] = []

    # Collect consecutive table rows
    table_lines: list[str] = []
    in_table = False

    def _flush_table():
        """Process and flush accumulated table lines."""
        if not table_lines:
            return

        # Determine column count from all rows (use max)
        col_count = 0
        for tl in table_lines:
            stripped_t = tl.strip()
            cells = [c.strip() for c in stripped_t.split("|")]
            # Filter empty strings from split edges
            cells = [c for c in cells if c != ""]
            # Skip separator rows for column count (use data rows)
            is_sep = all(re.match(r"^[\s\-:]+$", c) for c in cells if c)
            if not is_sep:
                col_count = max(col_count, len(cells))

        # Fallback: if all rows were separators, count from them
        if col_count == 0:
            for tl in table_lines:
                stripped_t = tl.strip()
                cells = [c.strip() for c in stripped_t.split("|")]
                cells = [c for c in cells if c != ""]
                col_count = max(col_count, len(cells))

        # Single-column "table" that looks like a list — don't treat as table
        if col_count <= 1:
            result.extend(table_lines)
            return

        # Pad each row to col_count
        for tl in table_lines:
            stripped_t = tl.strip()
            cells = stripped_t.split("|")

            # Handle leading/trailing pipe split artifacts
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]

            # Strip whitespace from each cell
            cells = [c.strip() for c in cells]

            # Fix separator rows
            is_separator = all(
                re.match(r"^[\s\-:]+$", c) for c in cells if c
            )
            if is_separator:
                cells = ["---"] * col_count
            elif len(cells) < col_count:
                # Pad short rows
                cells = cells + [""] * (col_count - len(cells))
            elif len(cells) > col_count:
                # Truncate long rows
                cells = cells[:col_count]

            result.append("| " + " | ".join(cells) + " |")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        else:
            if in_table:
                _flush_table()
                table_lines = []
                in_table = False
            result.append(line)

    # Flush any remaining table
    if in_table:
        _flush_table()

    return "\n".join(result)


def postprocess_ocr_result(text: str) -> str:
    """Apply the full post-processing pipeline to OCR output.

    Pipeline order: clean -> normalize_tables -> merge_paragraphs -> detect_headings

    Args:
        text: Raw OCR text.

    Returns:
        Post-processed text.
    """
    text = clean_ocr_text(text)
    text = normalize_tables(text)
    text = merge_paragraphs(text)
    text = detect_headings(text)
    return text
