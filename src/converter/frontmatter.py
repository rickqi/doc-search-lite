"""YAML Frontmatter 注入与剥离工具。

在转换后的 .md 文件头部注入 OKF 风格的 YAML frontmatter，
使文档自带结构化元数据（type/title/tags/headings），
同时保证索引和 Agent 工具能正确剥离 frontmatter 再处理。

设计原则:
- inject_frontmatter() 幂等：先剥离已有 frontmatter 再注入
- strip_frontmatter() 安全：无 frontmatter 时原样返回
- parse_frontmatter() 轻量：不依赖 PyYAML，纯字符串解析
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# frontmatter 块正则：---\n ... \n---\n
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# frontmatter 注入时最多保留的 headings 数量
MAX_HEADINGS_IN_FRONTMATTER = 20


def inject_frontmatter(content: str, metadata: Dict[str, Any]) -> str:
    """在 Markdown 内容头部注入 YAML frontmatter。

    如果内容已有 frontmatter，先剥离再重新注入（幂等）。

    Args:
        content: Markdown 正文内容（可能已有 frontmatter）
        metadata: 元数据字典，支持:
            - title (str): 文档标题
            - doc_type (str): 文档类型 (policy/process/report/manual/data/other)
            - source (str): 源文件名
            - tags (list): 标签列表
            - converted_at (str): 转换时间 ISO 格式
            - headings (list): 标题结构 [{level, text, line}]

    Returns:
        带 YAML frontmatter 的 Markdown 内容
    """
    # 先剥离已有 frontmatter
    _, body = strip_frontmatter(content)

    lines: List[str] = ["---"]

    title = metadata.get("title", "")
    lines.append(f"title: {_yaml_escape(title)}")

    doc_type = metadata.get("doc_type") or metadata.get("type") or "document"
    lines.append(f"type: {_yaml_escape(doc_type)}")

    source = metadata.get("source", "")
    if source:
        lines.append(f"source: {_yaml_escape(source)}")

    tags = metadata.get("tags", [])
    if tags:
        tag_str = ", ".join(_yaml_escape(str(t)) for t in tags)
        lines.append(f"tags: [{tag_str}]")

    ts = metadata.get("converted_at", "")
    if ts:
        lines.append(f"converted_at: {_yaml_escape(ts)}")

    headings = metadata.get("headings", [])
    if headings:
        lines.append("headings:")
        for h in headings[:MAX_HEADINGS_IN_FRONTMATTER]:
            if isinstance(h, dict):
                level = h.get("level", 1)
                text = h.get("text", "")
            else:
                level = 1
                text = str(h)
            lines.append(f"  - level: {level}")
            lines.append(f"    text: {_yaml_escape(text)}")

    lines.append("---")
    lines.append("")

    return "\n".join(lines) + body


def strip_frontmatter(content: str) -> Tuple[bool, str]:
    """剥离 YAML frontmatter。

    Args:
        content: 可能包含 frontmatter 的文本

    Returns:
        (has_frontmatter, body) 元组:
        - has_frontmatter: 是否检测到并剥离了 frontmatter
        - body: 剥离后的正文（无 frontmatter 时等于 content）
    """
    match = _FRONTMATTER_RE.match(content)
    if match:
        return True, content[match.end() :]
    return False, content


def has_frontmatter(content: str) -> bool:
    """检查内容是否以 YAML frontmatter 开头。

    Args:
        content: 文本内容

    Returns:
        True 如果内容以 ---\\n 开头且包含闭合 ---\\n
    """
    return bool(_FRONTMATTER_RE.match(content))


def parse_frontmatter(content: str) -> Optional[Dict[str, Any]]:
    """简单解析 YAML frontmatter 为字典（不依赖 PyYAML）。

    仅支持简单 ``key: value`` 和 ``key: [a, b]`` 格式。
    headings 等嵌套结构做轻量解析。

    Args:
        content: 可能包含 frontmatter 的文本

    Returns:
        解析后的字典，无 frontmatter 时返回 None
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None

    raw = match.group(1)
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[List[Any]] = None

    for line in raw.split("\n"):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Check for list item under current_key (e.g., "  - level: 1")
        if current_key and stripped.startswith("- ") and current_list is not None:
            item_text = stripped[2:].strip()
            # Try to parse "key: value" within list item
            if ":" in item_text:
                # Nested dict in list (e.g., headings entries)
                # Just store the text value if it's "text: xxx"
                sub_key, _, sub_val = item_text.partition(":")
                if sub_key.strip() == "text":
                    current_list.append(sub_val.strip().strip("'\""))
            else:
                current_list.append(item_text.strip("'\""))
            continue

        # Flush pending list
        if current_key and current_list is not None:
            result[current_key] = current_list
            current_key = None
            current_list = None

        # Parse "key: value" line
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if not value:
                # Start of a nested list block (e.g., "headings:")
                current_key = key
                current_list = []
            elif value.startswith("[") and value.endswith("]"):
                # Inline list: [a, b, c]
                inner = value[1:-1].strip()
                if inner:
                    result[key] = [
                        v.strip().strip("'\"") for v in inner.split(",")
                    ]
                else:
                    result[key] = []
            else:
                result[key] = value.strip("'\"")

    # Flush any remaining list
    if current_key and current_list is not None:
        result[current_key] = current_list

    return result


def _yaml_escape(value: str) -> str:
    """简单 YAML 值转义。

    包含特殊字符（: # [ ] { } 换行 引号）时加双引号包裹。

    Args:
        value: 原始字符串

    Returns:
        YAML 安全的值字符串
    """
    value = str(value)
    # 空字符串不需要引号
    if not value:
        return '""'
    if any(c in value for c in [":", "#", "[", "]", "{", "}", "\n", '"', "'"]):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value
