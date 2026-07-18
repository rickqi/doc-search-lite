"""Document chunker for splitting long Markdown documents into indexable chunks.

Splits by H2/H3 headings (primary split points), with H4/H5/H6 fallback
and paragraph-boundary hard-splitting for oversized sections.

Used by:
- ``src/cli.py`` build-index command (``--chunk-mode``)
- ``src/watch/index_watcher.py`` IndexWatcher (``chunk_mode=True``)

Zero schema changes — chunk info is encoded in ``doc_id`` (``#c{N}`` suffix)
and ``title`` (``{filename} § {heading_text}``).
"""

from typing import Dict, List, Tuple

# Chunks smaller than this get merged into the previous chunk.
MIN_MERGE_CHARS = 500


def split_into_chunks(
    content: str,
    headings: List[Dict],
    max_chunk_chars: int = 50_000,
) -> List[Tuple[str, str]]:
    """Split Markdown *content* into chunks delimited by heading boundaries.

    Args:
        content: Full Markdown text.
        headings: Heading list (from :func:`extract_headings`), each item is
            ``{"level": int, "text": str, "line": int}`` (1-based line).
        max_chunk_chars: Soft upper bound on chunk length in characters.

    Returns:
        List of ``(chunk_title, chunk_content)`` tuples.  Always returns **at
        least one** entry.  When the document is short enough or has no
        suitable headings the whole content is returned as
        ``("全文", content)``.
    """
    if not content.strip():
        return [("全文", content)]

    # Short document → single chunk (regardless of headings)
    if len(content) <= max_chunk_chars:
        return [("全文", content)]

    # Long document with no headings → hard-split at paragraph boundaries
    if not headings:
        return _hard_split_paragraphs(content, max_chunk_chars)

    lines = content.split("\n")

    # ── Primary split: H2 / H3 ──────────────────────────────
    primary = [h for h in headings if 2 <= h["level"] <= 3]

    if primary:
        raw_chunks = _split_by_headings(lines, primary)
    else:
        # No H2/H3 — fall back to H4-H6 if available
        secondary = [h for h in headings if h["level"] >= 4]
        if secondary:
            raw_chunks = _split_by_headings(lines, secondary)
        else:
            # Only H1 or unrecognised — hard split at paragraph boundaries
            return _hard_split_paragraphs(content, max_chunk_chars)

    # ── Refine oversized chunks ─────────────────────────────
    refined: List[Tuple[str, str]] = []
    for title, text, start_line, end_line in raw_chunks:
        if len(text) <= max_chunk_chars:
            refined.append((title, text))
            continue

        # Try sub-headings (H4+) within this chunk's line range
        sub_headings = [
            h for h in headings
            if h["level"] >= 4 and start_line <= h["line"] <= end_line
        ]
        if len(sub_headings) >= 2:
            sub_raw = _split_by_headings(lines, sub_headings)
            sub_refined: List[Tuple[str, str]] = []
            for st, stext, _, _ in sub_raw:
                if len(stext) > max_chunk_chars:
                    parts = _hard_split_paragraphs(stext, max_chunk_chars)
                    if len(parts) <= 1:
                        sub_refined.append((st, stext))
                    else:
                        for idx, (_, part) in enumerate(parts):
                            sub_refined.append((f"{st}（{idx + 1}）", part))
                else:
                    sub_refined.append((st, stext))
            refined.extend(sub_refined)
        else:
            # No sub-headings — hard split at paragraph boundaries
            parts = _hard_split_paragraphs(text, max_chunk_chars)
            if len(parts) <= 1:
                refined.append((title, text))
            else:
                for idx, (_, part) in enumerate(parts):
                    refined.append((f"{title}（{idx + 1}）", part))

    # ── Merge tiny chunks ───────────────────────────────────
    merged = _merge_tiny_chunks(refined, MIN_MERGE_CHARS, max_chunk_chars)

    if not merged:
        return [("全文", content)]
    return merged


# ── Internal helpers ────────────────────────────────────────


def _split_by_headings(
    lines: List[str],
    split_points: List[Dict],
) -> List[Tuple[str, str, int, int]]:
    """Build ``(title, text, start_line, end_line)`` tuples from *lines*.

    *split_points* must be sorted by ``line`` (ascending).  Content before
    the first split point becomes a chunk titled ``"开头"``.
    """
    chunks: List[Tuple[str, str, int, int]] = []
    total = len(lines)

    # Ensure split_points are sorted by line number
    sorted_sp = sorted(split_points, key=lambda h: h["line"])

    first_line = sorted_sp[0]["line"]
    if first_line > 1:
        pre_text = "\n".join(lines[: first_line - 1]).strip()
        if pre_text:
            chunks.append(("开头", pre_text, 1, first_line - 1))

    for i, h in enumerate(sorted_sp):
        start = h["line"]
        end = sorted_sp[i + 1]["line"] - 1 if i + 1 < len(sorted_sp) else total
        text = "\n".join(lines[start - 1 : end]).strip()
        if text:
            chunks.append((h["text"], text, start, end))

    return chunks


def _hard_split_paragraphs(
    text: str,
    max_chars: int,
) -> List[Tuple[str, str]]:
    """Split *text* at ``\\n\\n`` paragraph boundaries to stay under *max_chars*.

    If a single paragraph exceeds *max_chars* it is force-split at the
    character boundary.
    """
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= max_chars:
            current += "\n\n" + para
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    # Force-split any remaining oversized chunk (single huge paragraph)
    result: List[Tuple[str, str]] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(("（续）", chunk))
        else:
            for i in range(0, len(chunk), max_chars):
                result.append(("（续）", chunk[i : i + max_chars]))

    return result


def _merge_tiny_chunks(
    chunks: List[Tuple[str, str]],
    min_chars: int,
    max_chars: int = 0,
) -> List[Tuple[str, str]]:
    """Merge chunks shorter than *min_chars* into the previous chunk.

    If *max_chars* > 0, merging only occurs when the combined result would
    not exceed *max_chars*.  This prevents merging from undoing hard-splits.
    """
    merged: List[Tuple[str, str]] = []
    for title, text in chunks:
        if merged and len(text) < min_chars:
            prev_title, prev_text = merged[-1]
            combined = len(prev_text) + 2 + len(text)
            if max_chars <= 0 or combined <= max_chars:
                merged[-1] = (prev_title, prev_text + "\n\n" + text)
                continue
        merged.append((title, text))
    return merged
