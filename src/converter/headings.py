"""Markdown 标题提取工具。

从转换后的 Markdown 内容中提取标题层级结构，
用于 Agent 的文档结构感知导航。

零 LLM 成本，纯正则提取。
"""

import re
from typing import List, Dict

# 最大提取标题数，防止超长文档消耗过多 token
MAX_HEADINGS = 30


def extract_headings(markdown: str) -> List[Dict]:
    """从 Markdown 内容中提取标题层级结构。

    仅提取 ``#`` 开头的标准 Markdown 标题行，
    不提取代码块内的 ``#`` 注释。

    Args:
        markdown: Markdown 文本内容

    Returns:
        标题列表，每项含:
        - level: int (1-6, 对应 ``#`` 到 ``######``)
        - text: str (标题文本，去除 ``#`` 前缀)
        - line: int (行号，从 1 开始)
    """
    if not markdown:
        return []

    headings: List[Dict] = []
    in_code_block = False

    for line_num, line in enumerate(markdown.split("\n"), start=1):
        stripped = line.rstrip()

        # Track code blocks — skip # inside ```
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if match:
            headings.append({
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
                "line": line_num,
            })
            if len(headings) >= MAX_HEADINGS:
                break

    return headings
